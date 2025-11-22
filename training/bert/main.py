from pathlib import Path
from uuid import uuid4
import math
import json
import polars as pl

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import tiktoken
from mlx.utils import tree_flatten
from tora import Tora
from tqdm import tqdm

from networks.transformers.bert.model import Bert

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "bookcorpus-refined" / "extracted" / "BookCorpus3.csv"
DATASET_NAME = DATASET_PATH.stem
CHECKPOINT_DIR = ROOT / "training" / "bert" / "checkpoints"
DESCRIPTION = f"BERT MLM pretraining on {DATASET_NAME}"
TEXT_COL = "0"


class BertMLMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.dense(x)
        x = self.gelu(x)
        x = self.norm(x)
        return self.decoder(x)


class BertForMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        n_layers: int,
    ):
        super().__init__()
        self.bert = Bert(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
        )
        self.lm_head = BertMLMHead(d_model=d_model, vocab_size=vocab_size)

    def __call__(self, tokens: mx.array, mask: mx.array) -> mx.array:
        hidden = self.bert(tokens, mask)
        return self.lm_head(hidden)


def stream_batches(
    path: Path,
    tokenizer,
    seq_len: int,
    pad_id: int,
    batch_size: int,
    chunk_batches: int,
):
    chunk_size = chunk_batches * batch_size
    sequences: list[list[int]] = []
    token_buffer: list[int] = []

    def yield_batches(seq_list: list[list[int]]):
        arr = np.array(seq_list, dtype=np.int32)
        idx = np.random.permutation(len(arr))
        arr = arr[idx]
        for start in range(0, len(arr), batch_size):
            yield mx.array(arr[start : start + batch_size], dtype=mx.int32)

    reader = pl.read_csv_batched(str(path), batch_size=10000)

    while True:
        try:
            batches = reader.next_batches(1)
            if not batches:
                break
            df = batches[0]

            texts = df[TEXT_COL].to_list()
            for text in texts:
                text = text or ""
                token_ids = tokenizer.encode_ordinary(text)
                token_buffer.extend(token_ids)
                while len(token_buffer) >= seq_len:
                    sequences.append(token_buffer[:seq_len])
                    token_buffer = token_buffer[seq_len:]
                    if len(sequences) >= chunk_size:
                        yield from yield_batches(sequences)
                        sequences = []
        except StopIteration:
            break

    if token_buffer:
        remainder = len(token_buffer) % seq_len
        if remainder:
            token_buffer.extend([pad_id] * (seq_len - remainder))
        for start in range(0, len(token_buffer), seq_len):
            sequences.append(token_buffer[start : start + seq_len])

    if sequences:
        yield from yield_batches(sequences)


def count_total_sequences(path: Path, tokenizer, seq_len: int) -> int:
    total_tokens = 0
    reader = pl.read_csv_batched(str(path), batch_size=10000)

    while True:
        try:
            batches = reader.next_batches(1)
            if not batches:
                break
            df = batches[0]
            texts = df[TEXT_COL].to_list()

            for text in texts:
                text = text or ""
                total_tokens += len(tokenizer.encode_ordinary(text))
        except StopIteration:
            break

    return (total_tokens + seq_len - 1) // seq_len


def create_attention_mask(tokens: mx.array, pad_id: int) -> mx.array:
    attention_mask = tokens != pad_id
    attention_mask = attention_mask.astype(mx.float32)
    attn_mask = attention_mask[:, None, :, None] * attention_mask[:, None, None, :]
    return mx.where(attn_mask, 0.0, -1e9).astype(mx.float32)


def prepare_mlm_batch(
    batch: mx.array,
    mask_token_id: int,
    pad_id: int,
    base_vocab_size: int,
    mask_prob: float,
) -> tuple[mx.array, mx.array, mx.array]:
    mask_selector = (mx.random.uniform(shape=batch.shape) < mask_prob) & (
        batch != pad_id
    )

    rand_vals = mx.random.uniform(shape=batch.shape)
    mask_token_positions = mask_selector & (rand_vals < 0.8)
    random_token_positions = mask_selector & (rand_vals >= 0.8) & (rand_vals < 0.9)

    random_tokens = mx.random.randint(
        low=0, high=base_vocab_size, shape=batch.shape, dtype=batch.dtype
    )
    mask_token_value = mx.array(mask_token_id, dtype=batch.dtype)

    masked_inputs = mx.where(mask_token_positions, mask_token_value, batch)
    masked_inputs = mx.where(random_token_positions, random_tokens, masked_inputs)

    prediction_mask = mask_selector.astype(mx.float32)
    labels = batch
    return masked_inputs, labels, prediction_mask


def masked_language_modeling_loss(
    logits: mx.array, labels: mx.array, prediction_mask: mx.array
) -> mx.array:
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)
    flat_mask = prediction_mask.reshape(-1)

    losses = nn.losses.cross_entropy(flat_logits, flat_labels, reduction="none")
    masked_loss = mx.sum(losses * flat_mask) / mx.maximum(mx.sum(flat_mask), 1.0)
    return masked_loss


def decode_tokens(tokenizer, token_ids: list[int], pad_id: int) -> str:
    filtered = [
        int(t) for t in token_ids if int(t) != pad_id and int(t) < tokenizer.n_vocab
    ]
    if not filtered:
        return ""
    return tokenizer.decode(filtered)


def decode_tokens_with_mask(
    tokenizer, token_ids: list[int], pad_id: int, mask_id: int
) -> str:
    pieces: list[str] = []
    buffer: list[int] = []
    for token in token_ids:
        tid = int(token)
        if tid == pad_id:
            continue
        if tid == mask_id:
            if buffer:
                pieces.append(tokenizer.decode(buffer))
                buffer = []
            pieces.append(" [MASK]")
        elif tid < tokenizer.n_vocab:
            buffer.append(tid)
    if buffer:
        pieces.append(tokenizer.decode(buffer))
    decoded = "".join(pieces).strip()
    return decoded.replace("  ", " ")


def compute_loss(
    model: BertForMLM,
    batch: mx.array,
    pad_token_id: int,
    mask_token_id: int,
    base_vocab_size: int,
    mask_prob: float,
) -> mx.array:
    input_ids, labels, prediction_mask = prepare_mlm_batch(
        batch, mask_token_id, pad_token_id, base_vocab_size, mask_prob
    )
    attention_mask = create_attention_mask(input_ids, pad_token_id)
    logits = model(input_ids, attention_mask)
    return masked_language_modeling_loss(logits, labels, prediction_mask)


def sample_generation(
    model: BertForMLM,
    tokenizer,
    pad_token_id: int,
    mask_token_id: int,
    base_vocab_size: int,
    mask_prob: float,
    seq_len: int,
    batch_size: int,
    chunk_batches: int,
) -> dict | None:
    batches = stream_batches(
        DATASET_PATH,
        tokenizer,
        seq_len,
        pad_token_id,
        batch_size,
        chunk_batches,
    )
    try:
        batch = next(batches)
    except StopIteration:
        return None

    masked_inputs, labels, prediction_mask = prepare_mlm_batch(
        batch, mask_token_id, pad_token_id, base_vocab_size, mask_prob
    )
    attention_mask = create_attention_mask(masked_inputs, pad_token_id)
    logits = model(masked_inputs, attention_mask)
    predictions = mx.argmax(logits, axis=-1)
    prediction_mask_bool = prediction_mask > 0
    filled = mx.where(prediction_mask_bool, predictions, labels)

    example_idx = 0
    original_tokens = labels[example_idx].tolist()
    masked_tokens = masked_inputs[example_idx].tolist()
    filled_tokens = filled[example_idx].tolist()

    record = {
        "original": decode_tokens(tokenizer, original_tokens, pad_token_id),
        "masked": decode_tokens_with_mask(
            tokenizer, masked_tokens, pad_token_id, mask_token_id
        ),
        "predicted": decode_tokens(tokenizer, filled_tokens, pad_token_id),
    }

    return record


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    base_vocab_size = tokenizer.n_vocab
    mask_token_id = base_vocab_size
    pad_token_id = base_vocab_size + 1
    vocab_size = base_vocab_size + 2

    seq_len = 128
    batch_size = 32
    d_model = 256
    n_heads = 4
    n_layers = 4
    epochs = 50
    learning_rate = 1e-4
    mask_prob = 0.15

    model = BertForMLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        max_seq_len=seq_len,
        n_layers=n_layers,
    )
    optimizer = optim.AdamW(learning_rate=learning_rate)
    mx.eval(model.parameters())

    loss_fn = lambda m, batch: compute_loss(  # type: ignore
        m,
        batch,
        pad_token_id,
        mask_token_id,
        base_vocab_size,
        mask_prob,
    )
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    tora = Tora.create_experiment(
        name=f"BERT_MLM_{uuid4().hex[:3]}",
        description=DESCRIPTION,
        hyperparams={
            "seq_len": seq_len,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "mask_prob": mask_prob,
            "num_params": num_params,
        },
        workspace_id="da377350-b7dc-416d-a2fc-8c232396e476",
    )
    tora.max_buffer_len = 1

    chunk_batches = 64
    total_sequences = count_total_sequences(DATASET_PATH, tokenizer, seq_len)
    batches_per_epoch = max(1, math.ceil(total_sequences / batch_size))

    generation_log_path = CHECKPOINT_DIR / f"bert_mlm_samples_{DATASET_NAME}.jsonl"
    generation_log_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        culm_loss = 0.0
        num_samples = 0

        batches = stream_batches(
            DATASET_PATH,
            tokenizer,
            seq_len,
            pad_token_id,
            batch_size,
            chunk_batches,
        )

        progress = tqdm(
            batches,
            total=batches_per_epoch,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
        )
        for step, batch in enumerate(progress, start=1):
            loss, grads = loss_and_grad_fn(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            culm_loss += loss.item() * batch.shape[0]
            num_samples += batch.shape[0]
            avg = culm_loss / num_samples if num_samples else float("nan")
            progress.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg:.4f}")

            if step % 100 == 0:
                progress.write(
                    f"Epoch {epoch + 1} Step {step}: batch_loss={loss.item():.4f}, avg_loss={avg:.4f}"
                )

            if step % 200 == 0:
                record = sample_generation(
                    model,
                    tokenizer,
                    pad_token_id,
                    mask_token_id,
                    base_vocab_size,
                    mask_prob,
                    seq_len,
                    batch_size,
                    chunk_batches,
                )
                if record:
                    record["epoch"] = epoch + 1
                    record["step"] = step

                    with generation_log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

        progress.close()
        avg_loss = culm_loss / num_samples
        tora.metric("train_loss", step_or_epoch=epoch, value=avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} - train loss: {avg_loss:.4f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_id = uuid4().hex[:8]
    checkpoint_path = CHECKPOINT_DIR / f"bert_mlm_{DATASET_NAME}_{checkpoint_id}.npz"
    model.save_weights(str(checkpoint_path))
    print(f"Saved checkpoint to {checkpoint_path}")

    record = sample_generation(
        model,
        tokenizer,
        pad_token_id,
        mask_token_id,
        base_vocab_size,
        mask_prob,
        seq_len,
        batch_size,
        chunk_batches,
    )
    if record:
        with generation_log_path.open("a", encoding="utf-8") as f:
            record |= {"epoch": "final", "step": "final"}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
