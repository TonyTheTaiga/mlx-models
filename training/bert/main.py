import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import polars as pl
import tiktoken
import yaml
from mlx.utils import tree_flatten
from tora import Tora
from tqdm import tqdm

from networks.transformers.bert.model import Bert

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def resolve_path(value: str | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def build_config(config_path: Path = CONFIG_PATH) -> SimpleNamespace:
    cfg = load_yaml_config(config_path)
    base = config_path.parent

    dataset_path = resolve_path(cfg.get("dataset_path"), base) or (
        ROOT / "data" / "bookcorpus-refined" / "extracted" / "BookCorpus3.csv"
    )
    dataset_name = dataset_path.stem
    checkpoint_dir = resolve_path(cfg.get("checkpoint_dir"), base) or (
        ROOT / "training" / "bert" / "checkpoints"
    )
    description = cfg.get("description", f"BERT MLM pretraining on {dataset_name}")
    text_col = str(cfg.get("text_column", "0"))
    workspace_id = cfg.get("workspace_id", "da377350-b7dc-416d-a2fc-8c232396e476")
    tokenizer = cfg.get("tokenizer", "gpt2")
    seq_len = int(cfg.get("seq_len", 128))
    batch_size = int(cfg.get("batch_size", 32))
    epochs = int(cfg.get("epochs", 50))
    learning_rate = float(cfg.get("learning_rate", 1e-4))
    d_model = int(cfg.get("d_model", 256))
    n_heads = int(cfg.get("n_heads", 4))
    n_layers = int(cfg.get("n_layers", 4))
    mask_prob = float(cfg.get("mask_prob", 0.15))
    max_sequences = cfg.get("max_sequences")
    chunk_batches = int(cfg.get("chunk_batches", 64))

    sample_log_value = cfg.get("sample_log")
    if sample_log_value is None:
        sample_log_path = checkpoint_dir / f"bert_mlm_samples_{dataset_name}.jsonl"
    else:
        sample_log_path = resolve_path(sample_log_value, checkpoint_dir or base)

    return SimpleNamespace(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        checkpoint_dir=checkpoint_dir,
        description=description,
        text_col=text_col,
        workspace_id=workspace_id,
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        mask_prob=mask_prob,
        max_sequences=max_sequences,
        chunk_batches=chunk_batches,
        sample_log_path=sample_log_path,
    )


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
    text_col: str,
    max_sequences: int | None = None,
):
    chunk_size = chunk_batches * batch_size
    sequences: list[list[int]] = []
    token_buffer: list[int] = []
    total_yielded = 0

    def yield_batches(seq_list: list[list[int]]):
        nonlocal total_yielded
        arr = np.array(seq_list, dtype=np.int32)
        idx = np.random.permutation(len(arr))
        arr = arr[idx]
        for start in range(0, len(arr), batch_size):
            if max_sequences is not None and total_yielded >= max_sequences:
                return
            batch = arr[start : start + batch_size]

            # Truncate batch if it exceeds max_sequences
            if max_sequences is not None and total_yielded + len(batch) > max_sequences:
                remaining = max_sequences - total_yielded
                batch = batch[:remaining]

            yield mx.array(batch, dtype=mx.int32)
            total_yielded += len(batch)

    reader = pl.read_csv_batched(str(path), batch_size=10000)

    while True:
        if max_sequences is not None and total_yielded >= max_sequences:
            break
        try:
            batches = reader.next_batches(1)
            if not batches:
                break
            df = batches[0]

            texts = df[text_col].to_list()
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
                        if max_sequences is not None and total_yielded >= max_sequences:
                            break
        except StopIteration:
            break

    if token_buffer and (max_sequences is None or total_yielded < max_sequences):
        remainder = len(token_buffer) % seq_len
        if remainder:
            token_buffer.extend([pad_id] * (seq_len - remainder))
        for start in range(0, len(token_buffer), seq_len):
            sequences.append(token_buffer[start : start + seq_len])

    if sequences and (max_sequences is None or total_yielded < max_sequences):
        yield from yield_batches(sequences)


def count_total_sequences(
    path: Path,
    tokenizer,
    seq_len: int,
    text_col: str,
    max_sequences: int | None = None,
) -> int:
    if max_sequences is not None:
        return max_sequences

    total_tokens = 0
    # Use Polars to read CSV in batches to avoid OOM on large files
    reader = pl.read_csv_batched(str(path), batch_size=10000)

    while True:
        try:
            batches = reader.next_batches(1)
            if not batches:
                break
            df = batches[0]
            texts = df[text_col].to_list()

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
    mask_selector = (mx.random.uniform(shape=batch.shape) < mask_prob) & (batch != pad_id)

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
    filtered = [int(t) for t in token_ids if int(t) != pad_id and int(t) < tokenizer.n_vocab]
    if not filtered:
        return ""
    return tokenizer.decode(filtered)


def decode_tokens_with_mask(tokenizer, token_ids: list[int], pad_id: int, mask_id: int) -> str:
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
    dataset_path: Path,
    text_col: str,
    max_sequences: int | None,
) -> dict | None:
    batches = stream_batches(
        dataset_path,
        tokenizer,
        seq_len,
        pad_token_id,
        batch_size,
        chunk_batches,
        text_col=text_col,
        max_sequences=max_sequences,
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
        "masked": decode_tokens_with_mask(tokenizer, masked_tokens, pad_token_id, mask_token_id),
        "predicted": decode_tokens(tokenizer, filled_tokens, pad_token_id),
    }

    return record


def main(cfg: SimpleNamespace | None = None):
    cfg = cfg or build_config()

    tokenizer = tiktoken.get_encoding(cfg.tokenizer)
    base_vocab_size = tokenizer.n_vocab
    mask_token_id = base_vocab_size
    pad_token_id = base_vocab_size + 1
    vocab_size = base_vocab_size + 2

    model = BertForMLM(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        max_seq_len=cfg.seq_len,
        n_layers=cfg.n_layers,
    )
    optimizer = optim.AdamW(learning_rate=cfg.learning_rate)
    mx.eval(model.parameters())

    loss_fn = lambda m, batch: compute_loss(  # type: ignore
        m,
        batch,
        pad_token_id,
        mask_token_id,
        base_vocab_size,
        cfg.mask_prob,
    )
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    total_sequences = count_total_sequences(
        cfg.dataset_path,
        tokenizer,
        cfg.seq_len,
        cfg.text_col,
        max_sequences=cfg.max_sequences,
    )
    if cfg.max_sequences is None:
        print(f"max_sequences=None, streaming all {total_sequences:,} sequences in the dataset.")
    tora = Tora.create_experiment(
        name=f"BERT_MLM_{uuid4().hex[:3]}",
        description=cfg.description,
        hyperparams={
            "seq_len": cfg.seq_len,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "mask_prob": cfg.mask_prob,
            "num_params": num_params,
            "max_sequences": total_sequences,
        },
        workspace_id=cfg.workspace_id,
    )
    tora.max_buffer_len = 1

    chunk_batches = cfg.chunk_batches
    batches_per_epoch = max(1, math.ceil(total_sequences / cfg.batch_size))

    generation_log_path = cfg.sample_log_path
    generation_log_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        culm_loss = 0.0
        num_samples = 0

        batches = stream_batches(
            cfg.dataset_path,
            tokenizer,
            cfg.seq_len,
            pad_token_id,
            cfg.batch_size,
            chunk_batches,
            text_col=cfg.text_col,
            max_sequences=total_sequences,
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
                    cfg.mask_prob,
                    cfg.seq_len,
                    cfg.batch_size,
                    chunk_batches,
                    cfg.dataset_path,
                    cfg.text_col,
                    cfg.max_sequences,
                )
                if record:
                    record["epoch"] = epoch + 1
                    record["step"] = step

                    with generation_log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

        progress.close()
        avg_loss = culm_loss / num_samples
        tora.metric("train_loss", step_or_epoch=epoch, value=avg_loss)
        print(f"Epoch {epoch + 1}/{cfg.epochs} - train loss: {avg_loss:.4f}")

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_id = uuid4().hex[:8]
    checkpoint_path = cfg.checkpoint_dir / f"bert_mlm_{cfg.dataset_name}_{checkpoint_id}.npz"
    model.save_weights(str(checkpoint_path))
    print(f"Saved checkpoint to {checkpoint_path}")

    record = sample_generation(
        model,
        tokenizer,
        pad_token_id,
        mask_token_id,
        base_vocab_size,
        cfg.mask_prob,
        cfg.seq_len,
        cfg.batch_size,
        chunk_batches,
        cfg.dataset_path,
        cfg.text_col,
        cfg.max_sequences,
    )
    if record:
        with generation_log_path.open("a", encoding="utf-8") as f:
            record |= {"epoch": "final", "step": "final"}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
