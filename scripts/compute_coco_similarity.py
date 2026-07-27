#!/usr/bin/env python3
import argparse
import csv
import importlib.machinery
import os
import sys
import types
import warnings
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

# The local CLIP module imports transformers through model/bert.py. In this
# environment, importing TensorFlow via transformers can fail because of an
# h5py/numpy ABI mismatch, while this script only needs PyTorch inference.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
warnings.filterwarnings("ignore", message=r"A NumPy version .* is required for this version of SciPy")
if "ftfy" not in sys.modules:
    ftfy_fallback = types.ModuleType("ftfy")
    ftfy_fallback.__spec__ = importlib.machinery.ModuleSpec("ftfy", loader=None)
    ftfy_fallback.fix_text = lambda text: text
    sys.modules["ftfy"] = ftfy_fallback

from clip import load, tokenize  # noqa: E402
from utils import image_transform  # noqa: E402


def parse_coco_txt(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 2)
            if len(parts) != 3:
                raise ValueError(f"Bad TSV format at {path}:{line_no}: {line}")
            image_id, image_path, caption = parts
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "caption": caption,
                }
            )
    return rows


def resolve_image_path(image_path: str) -> str:
    if os.path.isabs(image_path):
        return image_path
    candidates = [
        os.path.join(PROJECT_ROOT, image_path),
        os.path.join(PROJECT_ROOT, "data", "coco_data", image_path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


def select_examples(
    rows: List[Dict[str, str]],
    image_id: Optional[str],
    image_path: Optional[str],
    caption: Optional[str],
) -> List[Dict[str, str]]:
    if caption:
        if not image_path:
            raise ValueError("--caption requires --image_path or --image")
        return [
            {
                "image_id": image_id or "",
                "image_path": image_path,
                "caption": caption,
            }
        ]

    if image_id is not None:
        selected = [row for row in rows if row["image_id"] == str(image_id)]
    elif image_path is not None:
        selected = [
            row for row in rows
            if row["image_path"] == image_path or os.path.basename(row["image_path"]) == os.path.basename(image_path)
        ]
    else:
        first_path = rows[0]["image_path"]
        selected = [row for row in rows if row["image_path"] == first_path]

    if not selected:
        raise ValueError("No matching image/caption rows found in the dataset file.")
    return selected


def load_clip_from_checkpoint(checkpoint_path: str, device: torch.device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    clip_model, _ = load("ViT-B/32", jit=False)

    loaded_clip_state = checkpoint.get("clip_model") is not None
    if loaded_clip_state:
        missing, unexpected = clip_model.load_state_dict(checkpoint["clip_model"], strict=False)
        if missing:
            print(f"[warn] Missing CLIP keys when loading checkpoint: {len(missing)}")
        if unexpected:
            print(f"[warn] Unexpected CLIP keys when loading checkpoint: {len(unexpected)}")
    else:
        saved_args = checkpoint.get("args", {})
        uses_adapter = bool(saved_args.get("use_clip_lora") or saved_args.get("unfreeze_clip_projection"))
        print(
            "[warn] checkpoint does not contain `clip_model`; using base ViT-B/32 CLIP weights "
            "for cosine similarity."
        )
        if uses_adapter:
            print(
                "[warn] checkpoint args indicate CLIP adapter/projection training, but adapter weights "
                "are not stored under `clip_model` in this checkpoint."
            )

    clip_model = clip_model.to(device).eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)

    return clip_model, checkpoint, loaded_clip_state


@torch.no_grad()
def compute_similarities(clip_model, image_path: str, captions: List[str], device: torch.device, batch_size: int):
    transform = image_transform(is_train=False)
    image = Image.open(resolve_image_path(image_path)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    image_features = clip_model.encode_image(image_tensor)
    image_features = F.normalize(image_features.float(), dim=-1)

    results = []
    for start in range(0, len(captions), batch_size):
        batch_captions = captions[start:start + batch_size]
        text_tokens = torch.cat([tokenize(text, truncate=True) for text in batch_captions]).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features.float(), dim=-1)
        scores = (image_features @ text_features.T).squeeze(0).detach().cpu().tolist()
        results.extend(scores)
    return results


def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = ["rank", "image_id", "image_path", "caption", "cosine_similarity"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute image-text cosine similarity for one COCO image with a CLIP checkpoint."
    )
    parser.add_argument("--checkpoint", default="checkpoints/train_version2_pairwise_plugin.pt")
    parser.add_argument("--data", default="data/coco_dataset_test.txt")
    parser.add_argument("--image_id", default=None, help="COCO txt first-column id, e.g. 0")
    parser.add_argument("--image_path", "--image", dest="image_path", default=None, help="Relative or absolute image path")
    parser.add_argument("--caption", default=None, help="Optional single caption. If omitted, use captions from --data.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--topk", type=int, default=0, help="If >0, print only top-k captions by cosine similarity.")
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dataset_path = os.path.join(PROJECT_ROOT, args.data) if not os.path.isabs(args.data) else args.data
    checkpoint_path = os.path.join(PROJECT_ROOT, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint

    rows = parse_coco_txt(dataset_path)
    selected = select_examples(rows, args.image_id, args.image_path, args.caption)
    image_path = selected[0]["image_path"]
    captions = [row["caption"] for row in selected]

    clip_model, checkpoint, loaded_clip_state = load_clip_from_checkpoint(checkpoint_path, device)
    scores = compute_similarities(clip_model, image_path, captions, device, args.batch_size)

    output_rows = []
    for row, score in zip(selected, scores):
        output_rows.append(
            {
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "caption": row["caption"],
                "cosine_similarity": f"{score:.6f}",
            }
        )
    output_rows.sort(key=lambda item: float(item["cosine_similarity"]), reverse=True)
    for rank, row in enumerate(output_rows, start=1):
        row["rank"] = rank

    printed_rows = output_rows[: args.topk] if args.topk > 0 else output_rows
    print(f"checkpoint: {args.checkpoint}")
    print(f"checkpoint_epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"loaded_clip_model_from_checkpoint: {loaded_clip_state}")
    print(f"image: {image_path}")
    print("")
    for row in printed_rows:
        print(
            f"{row['rank']:>2}. cosine={row['cosine_similarity']} "
            f"id={row['image_id']} caption={row['caption']}"
        )

    if args.output:
        output_path = os.path.join(PROJECT_ROOT, args.output) if not os.path.isabs(args.output) else args.output
        write_csv(output_path, output_rows)
        print(f"\nsaved: {output_path}")


if __name__ == "__main__":
    main()
