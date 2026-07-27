import argparse
import csv
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
from clip import load
from dataloader_downstream import VG_Attribution, VG_Relation
from model import triple_Transformer
from utils import image_transform


class StructureResidualHead(nn.Module):
    def __init__(
        self,
        dim=512,
        hidden_dim=768,
        init_scale=0.1,
        max_scale=0.5,
        gate_tau=0.05,
        gate_max=0.5,
        direct_score_prediction=False,
        uniform_residual_weighting=False,
    ):
        super().__init__()
        init_ratio = min(max(init_scale / max_scale, 1e-4), 1 - 1e-4)
        self.raw_scale = nn.Parameter(torch.tensor(torch.logit(torch.tensor(init_ratio)).item()))
        self.max_scale = max_scale
        self.gate_tau = gate_tau
        self.gate_max = gate_max
        self.direct_score_prediction = direct_score_prediction
        self.uniform_residual_weighting = uniform_residual_weighting
        self.mlp = nn.Sequential(
            nn.Linear(dim * 7, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 7 + 1, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def scale(self):
        return torch.sigmoid(self.raw_scale) * self.max_scale

    def forward(self, image_features, text_features, knowledge_features, clip_score, clip_confidence=None):
        image_features = F.normalize(image_features.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)
        knowledge_features = F.normalize(knowledge_features.float(), dim=-1)
        clip_score = clip_score.float()
        x = torch.cat([
            image_features,
            text_features,
            knowledge_features,
            image_features * knowledge_features,
            text_features * knowledge_features,
            image_features * text_features,
            torch.abs(text_features - knowledge_features),
        ], dim=-1)
        delta = self.mlp(x).squeeze(-1)
        if self.direct_score_prediction:
            gate = torch.zeros_like(delta)
            return delta, delta, self.scale(), gate
        if self.uniform_residual_weighting:
            gate = torch.full_like(delta, float(self.gate_max))
        else:
            raw_gate = torch.sigmoid(self.gate_mlp(torch.cat([x, clip_score.unsqueeze(-1)], dim=-1))).squeeze(-1)
            if clip_confidence is None:
                confidence_gate = torch.ones_like(raw_gate)
            else:
                confidence_gate = torch.exp(-clip_confidence.float().clamp(min=0.0) / max(self.gate_tau, 1e-6))
            gate = self.gate_max * raw_gate * confidence_gate
        score = clip_score + self.scale() * gate * delta
        return score, delta, self.scale(), gate


def _get_arg(saved_args, name, default):
    return saved_args.get(name, default)


def _to_float(tensor):
    return float(tensor.detach().cpu().reshape(-1)[0].item())


def build_dataset(task, data_path):
    if task == "relation":
        return VG_Relation(transform=image_transform(is_train=False))
    return VG_Attribution(data_path=data_path, transform=image_transform(is_train=False))


def score_caption_pair(clip_model, triple_model, structure_head, item):
    image = item["image_options"][0].unsqueeze(0).cuda()
    false_caption, true_caption = item["caption_options"]

    true_tokens = clip.tokenize(true_caption, truncate=True).cuda()
    false_tokens = clip.tokenize(false_caption, truncate=True).cuda()

    image_features = F.normalize(clip_model.encode_image(image).float(), dim=-1)
    true_text_features = F.normalize(clip_model.encode_text(true_tokens).float(), dim=-1)
    false_text_features = F.normalize(clip_model.encode_text(false_tokens).float(), dim=-1)

    true_knowledge = F.normalize(
        triple_model(
            item["head_inputs"], item["relation_inputs"], item["tail_inputs"],
            0, item["attention_mask"].cuda()
        ).float(),
        dim=-1,
    )
    false_knowledge = F.normalize(
        triple_model(
            item["reversed_head_inputs"], item["reversed_relation_inputs"], item["reversed_tail_inputs"],
            0, item["reversed_attention_mask"].cuda()
        ).float(),
        dim=-1,
    )

    clip_true = (image_features * true_text_features).sum(dim=-1)
    clip_false = (image_features * false_text_features).sum(dim=-1)
    clip_confidence = torch.abs(clip_true.detach() - clip_false.detach())

    model_true, delta_true, scale, gate_true = structure_head(
        image_features, true_text_features, true_knowledge, clip_true,
        clip_confidence=clip_confidence,
    )
    model_false, delta_false, _, gate_false = structure_head(
        image_features, false_text_features, false_knowledge, clip_false,
        clip_confidence=clip_confidence,
    )

    return {
        "true_caption": true_caption,
        "false_caption": false_caption,
        "clip_true": _to_float(clip_true),
        "clip_false": _to_float(clip_false),
        "model_true": _to_float(model_true),
        "model_false": _to_float(model_false),
        "delta_true": _to_float(delta_true),
        "delta_false": _to_float(delta_false),
        "gate_true": _to_float(gate_true),
        "gate_false": _to_float(gate_false),
        "residual_scale": _to_float(scale),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/train_version2_pairwise_plugin.pt", type=str)
    parser.add_argument("--task", choices=["attribution", "relation"], default="attribution")
    parser.add_argument("--data_path", default="data/visual_genome_attribution_aug.json", type=str)
    parser.add_argument("--output", default="outputs/qualitative_scores_attribution.csv", type=str)
    parser.add_argument("--num_examples", default=30, type=int)
    parser.add_argument("--only_rescued", action="store_true", default=False)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_args = checkpoint.get("args", {})

    clip_model, _ = load("ViT-B/32", jit=False)
    clip_model = clip_model.cuda().eval()
    if checkpoint.get("clip_model") is not None:
        clip_model.load_state_dict(checkpoint["clip_model"], strict=False)
    for param in clip_model.parameters():
        param.requires_grad_(False)

    triple_model = triple_Transformer().cuda().eval()
    if checkpoint.get("triple_transformer") is not None:
        triple_model.load_state_dict(checkpoint["triple_transformer"], strict=False)
    for param in triple_model.parameters():
        param.requires_grad_(False)

    structure_head = StructureResidualHead(
        init_scale=_get_arg(saved_args, "structure_residual_scale", 0.1),
        max_scale=_get_arg(saved_args, "structure_residual_max_scale", 0.2),
        gate_tau=_get_arg(saved_args, "structure_gate_tau", 0.1),
        gate_max=_get_arg(saved_args, "structure_gate_max", 0.3),
        direct_score_prediction=_get_arg(saved_args, "direct_score_prediction", False),
        uniform_residual_weighting=_get_arg(saved_args, "uniform_residual_weighting", False),
    ).cuda().eval()
    structure_head.load_state_dict(checkpoint["structure_head"], strict=True)
    for param in structure_head.parameters():
        param.requires_grad_(False)

    dataset = build_dataset(args.task, args.data_path)
    rows = []
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            row = score_caption_pair(clip_model, triple_model, structure_head, item)
            raw_item = getattr(dataset, "dataset", [{}])[index]
            row["index"] = index
            row["task"] = args.task
            row["image_path"] = raw_item.get("image_path", "")
            row["clip_margin"] = row["clip_true"] - row["clip_false"]
            row["model_margin"] = row["model_true"] - row["model_false"]
            row["margin_improvement"] = row["model_margin"] - row["clip_margin"]
            row["clip_correct"] = int(row["clip_margin"] > 0)
            row["model_correct"] = int(row["model_margin"] > 0)
            if args.only_rescued and not (row["clip_correct"] == 0 and row["model_correct"] == 1):
                continue
            rows.append(row)

    rows.sort(key=lambda row: (row["model_correct"], row["margin_improvement"]), reverse=True)
    rows = rows[: args.num_examples]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.output.endswith(".json"):
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    else:
        fieldnames = [
            "index", "task", "image_path",
            "true_caption", "false_caption",
            "clip_true", "clip_false", "clip_margin", "clip_correct",
            "model_true", "model_false", "model_margin", "model_correct",
            "margin_improvement", "delta_true", "delta_false",
            "gate_true", "gate_false", "residual_scale",
        ]
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("Exported", len(rows), "examples to", args.output)


if __name__ == "__main__":
    main()
