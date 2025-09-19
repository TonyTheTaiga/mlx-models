import argparse
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from PIL import Image
from tora import Tora
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


class ImageFolderCaptionDataset(Dataset):
    """Loads .jpeg images and applies the same caption to each."""

    def __init__(self, root_dir: str, caption: str, resolution: int):
        self.paths = list(Path(root_dir).glob("**/*.jpeg"))
        if not self.paths:
            raise ValueError(f"No JPEG images found in {root_dir}")
        self.caption = caption
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=Image.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        pv = self.transform(img)
        return {"pixel_values": pv, "text": self.caption}


def parse_args():
    p = argparse.ArgumentParser("LoRA+Accelerate fine-tune SD v1.5")
    p.add_argument("--train_data_dir", type=str, required=True)
    p.add_argument("--caption", type=str, required=True)
    p.add_argument("--resolution", type=int, default=520)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument(
        "--tora_experiment_name",
        type=str,
        default="LoRA_SD_FineTune",
        help="Name for the Tora experiment.",
    )
    p.add_argument(
        "--tora_workspace_id",
        type=str,
        default="API_DEFAULT",
        help="Workspace ID for Tora experiment.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    hyperparams = {
        "resolution": args.resolution,
        "train_batch_size": args.train_batch_size,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "train_data_dir": args.train_data_dir,
        "caption": args.caption,
    }
    tora_client = Tora.create_experiment(
        name=args.tora_experiment_name,
        description="LoRA fine-tuning of Stable Diffusion v1.5",
        hyperparams=hyperparams,
        tags=["stable-diffusion", "lora", "fine-tuning"],
        workspace_id="a45ac854-f741-474d-aa5c-c6cfdf59b65a",
    )
    print(f"Tora experiment created: {tora_client._experiment_id}")

    accelerator = Accelerator()
    ds = ImageFolderCaptionDataset(args.train_data_dir, args.caption, args.resolution)
    dl = DataLoader(
        ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
    noise_sched = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
    )

    print(unet)

    for model in (text_encoder, vae, unet):
        model.requires_grad_(False)

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none",
        init_lora_weights="gaussian",
    )
    unet = get_peft_model(unet, peft_config)

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=args.learning_rate)

    unet, optimizer, dl, text_encoder, vae = accelerator.prepare(
        unet, optimizer, dl, text_encoder, vae
    )

    for epoch in range(args.num_train_epochs):
        unet.train()
        total_loss = 0.0
        prog = tqdm(
            dl,
            disable=not accelerator.is_main_process,
            desc=f"Epoch {epoch + 1}/{args.num_train_epochs}",
        )
        for batch in prog:
            with torch.no_grad():
                enc = tokenizer(
                    batch["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).to(accelerator.device)
                txt_emb = text_encoder(**enc).last_hidden_state

            with torch.no_grad():
                lat = vae.encode(batch["pixel_values"]).latent_dist.sample()
                lat = lat * vae.config.scaling_factor

            noise = torch.randn_like(lat)
            timesteps = torch.randint(
                0,
                noise_sched.config.num_train_timesteps,
                (lat.shape[0],),
                device=accelerator.device,
            ).long()
            noisy = noise_sched.add_noise(lat, noise, timesteps)

            pred = unet(noisy, timesteps, encoder_hidden_states=txt_emb).sample
            loss = torch.nn.functional.mse_loss(pred, noise)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            prog.set_postfix(loss=loss.item())
            total_loss += loss.item()

        if accelerator.is_main_process:
            avg_epoch_loss = total_loss / len(dl)
            tora_client.log(name="train_loss", value=avg_epoch_loss, step=epoch)

        if accelerator.is_main_process:
            ckpt_dir = os.path.join(args.output_dir, f"lora-epoch{epoch + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            unet.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("✅ Fine-tuning complete! LoRA weights in", args.output_dir)
        unet.save_pretrained(args.output_dir)
        tora_client.shutdown()


if __name__ == "__main__":
    main()
