"""Train a tiny Vision Transformer and export the weights for the browser explainer.

The block is the standard pre-norm ViT block: class token, learned positions,
multi-head self-attention, and a GELU MLP. Images are the synthetic shapes
from tao_lab. The label is color plus shape, nine classes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tao_lab.data import COLORS, SHAPES, sample_one

IMAGE = 32
PATCH = 8
DIM = 32
HEADS = 4
DEPTH = 4
MLP = 4
CLASSES = [f"{color} {shape}" for color in COLORS for shape in SHAPES]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "model.json"


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mlp)
        self.fc2 = nn.Linear(dim * mlp, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.ln1(tokens)
        batch, length, dim = hidden.shape
        head_dim = dim // self.heads
        qkv = self.qkv(hidden).reshape(batch, length, 3, self.heads, head_dim)
        query, key, value = qkv.unbind(2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) * (head_dim ** -0.5)
        weights = scores.softmax(dim=-1)
        mixed = (weights @ value).transpose(1, 2).reshape(batch, length, dim)
        tokens = tokens + self.proj(mixed)
        hidden = self.ln2(tokens)
        tokens = tokens + self.fc2(F.gelu(self.fc1(hidden)))
        return tokens


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, DIM, kernel_size=PATCH, stride=PATCH)
        self.cls = nn.Parameter(torch.zeros(1, 1, DIM))
        n_tokens = 1 + (IMAGE // PATCH) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, DIM))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(DIM, HEADS, MLP) for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, len(CLASSES))

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch(images).flatten(2).transpose(1, 2)
        cls = self.cls.expand(images.shape[0], -1, -1)
        return torch.cat([cls, patches], dim=1) + self.pos

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(images)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(self.norm(hidden[:, 0]))


def flat(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().contiguous().reshape(-1).tolist()


def linear(layer: nn.Linear) -> dict[str, list[float]]:
    return {"w": flat(layer.weight), "b": flat(layer.bias)}


def norm(layer: nn.LayerNorm) -> dict[str, list[float]]:
    return {"w": flat(layer.weight), "b": flat(layer.bias)}


def render(shape: str, color: str, cx: int, cy: int, radius: int = 7) -> torch.Tensor:
    size = IMAGE
    ys = torch.arange(size).view(size, 1).expand(size, size)
    xs = torch.arange(size).view(1, size).expand(size, size)
    if shape == "circle":
        mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2
    elif shape == "square":
        mask = ((ys - cy).abs() <= radius) & ((xs - cx).abs() <= radius)
    else:
        top = cy - radius
        height = 2 * radius
        rel = (ys.float() - top) / height
        half_width = rel * radius
        mask = (rel >= 0) & (rel <= 1) & ((xs.float() - cx).abs() <= half_width)
    rgb = {"red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0)}[color]
    image = torch.zeros(3, size, size)
    image[:, mask] = torch.tensor(rgb).view(3, 1)
    return image


def first_maps(model: TinyViT, image: torch.Tensor) -> tuple[list[float], list[float]]:
    block = model.blocks[0]
    tokens = model.embed(image.unsqueeze(0))
    hidden = block.ln1(tokens)
    batch, length, dim = hidden.shape
    head_dim = dim // block.heads
    qkv = block.qkv(hidden).reshape(batch, length, 3, block.heads, head_dim)
    query, key, _value = qkv.unbind(2)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    weights = ((query @ key.transpose(-2, -1)) * (head_dim ** -0.5)).softmax(dim=-1)
    return flat(tokens[0]), flat(weights[0, 0])


def batch_of(n: int, seed: int, noise: float) -> tuple[torch.Tensor, torch.Tensor]:
    images = []
    labels = []
    for index in range(n):
        generator = torch.Generator().manual_seed(seed + index)
        sample = sample_one(IMAGE, noise, generator)
        images.append(sample["image"])
        labels.append(int(sample["color"]) * 3 + int(sample["shape"]))
    return torch.stack(images), torch.tensor(labels)


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyViT().to(device)
    images, labels = batch_of(6000, 0, 0.08)
    val_images, val_labels = batch_of(800, 100_000, 0.08)
    images, labels = images.to(device), labels.to(device)
    val_images, val_labels = val_images.to(device), val_labels.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    batch = 128
    accuracy = 0.0
    for epoch in range(1, 41):
        model.train()
        order = torch.randperm(images.shape[0], device=device)
        total = 0.0
        for start in range(0, images.shape[0], batch):
            pick = order[start : start + batch]
            logits = model(images[pick])
            loss = F.cross_entropy(logits, labels[pick])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * pick.shape[0]
        model.eval()
        with torch.no_grad():
            pred = model(val_images).argmax(dim=-1)
            accuracy = float((pred == val_labels).float().mean())
        print(f"epoch {epoch:02d}  loss {total / images.shape[0]:.3f}  val {accuracy:.3f}", flush=True)
        if accuracy >= 0.97 and epoch >= 8:
            break

    if accuracy < 0.9:
        raise SystemExit(f"model did not learn the shapes (val accuracy {accuracy:.3f})")

    model.eval()
    examples = []
    positions = [("left", 11), ("center", 16), ("right", 21)]
    gallery = []
    for color in COLORS:
        for shape in SHAPES:
            gallery.append((f"{color}-{shape}", f"{color} {shape}", color, shape, 16))
    gallery.append(("red-circle-left", "red circle", "red", "circle", 11))
    gallery.append(("blue-square-right", "blue square", "blue", "square", 21))
    gallery.append(("green-triangle-left", "green triangle", "green", "triangle", 11))

    with torch.no_grad():
        for index, (example_id, label, color, shape, cx) in enumerate(gallery):
            image = render(shape, color, cx, 16).to(device)
            logits = model(image.unsqueeze(0))[0].cpu()
            item = {
                "id": example_id,
                "label": label,
                "pixels": flat(image.cpu()),
                "logits": flat(logits),
            }
            if index == 0:
                after_pos, attn = first_maps(model.cpu(), image.cpu())
                model.to(device)
                item["debug"] = {"afterPos": after_pos, "attnHead0": attn}
            examples.append(item)

    payload = {
        "config": {
            "imageSize": IMAGE,
            "patchSize": PATCH,
            "dim": DIM,
            "heads": HEADS,
            "depth": DEPTH,
            "mlpRatio": MLP,
            "eps": 1e-5,
            "classes": CLASSES,
            "valAccuracy": round(accuracy, 4),
        },
        "patchW": flat(model.patch.weight),
        "patchB": flat(model.patch.bias),
        "cls": flat(model.cls[0, 0]),
        "pos": flat(model.pos[0]),
        "blocks": [
            {
                "ln1": norm(block.ln1),
                "qkv": linear(block.qkv),
                "proj": linear(block.proj),
                "ln2": norm(block.ln2),
                "fc1": linear(block.fc1),
                "fc2": linear(block.fc2),
            }
            for block in model.blocks
        ],
        "norm": norm(model.norm),
        "head": linear(model.head),
        "examples": examples,
        "positions": [{"name": name, "cx": cx} for name, cx in positions],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload))
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
