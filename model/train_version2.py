import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BertTokenizer
import os
from contextlib import nullcontext

from tqdm import tqdm
from utils import get_args, set_manualSeed, image_transform, WinoLoss, CLIPLoss, MarginLoss
from dataloader import CoCoDataset, CoCoDataset_aug, CoCoDataset_aug_object_attribute, CoCoDataset_aug_update, \
    CoCoDataset_aug_update_withneg
from dataloader_downstream import VG_Relation, VG_Attribution
from clip import load
import clip
from model import myTransformer, bert_Transformer, triple_Transformer          # 保留 import
import eval_version2 as eval_utils
import wandb, json
from model_zoo import get_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCO_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "coco_data")
VG_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "visual_genome_data", "vg_image")


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
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

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


def _add_projection_lora(owner, base_name, rank, alpha):
    base = getattr(owner, base_name)
    if base is None:
        return []
    in_dim, out_dim = base.shape
    dtype = base.dtype
    device = base.device
    scale = float(alpha) / float(rank)
    lora_a = nn.Parameter(torch.empty(in_dim, rank, device=device, dtype=dtype))
    lora_b = nn.Parameter(torch.zeros(rank, out_dim, device=device, dtype=dtype))
    nn.init.normal_(lora_a, std=0.01)
    setattr(owner, f"{base_name}_lora_A", lora_a)
    setattr(owner, f"{base_name}_lora_B", lora_b)
    setattr(owner, f"{base_name}_lora_scale", scale)
    base.requires_grad_(False)
    return [lora_a, lora_b]


def add_clip_projection_lora(clip_model, rank=8, alpha=16.0, train_text=True, train_visual=True):
    lora_params = []
    for p in clip_model.parameters():
        p.requires_grad_(False)
    if train_text and hasattr(clip_model, "text_projection"):
        lora_params.extend(_add_projection_lora(clip_model, "text_projection", rank, alpha))
    if train_visual and hasattr(clip_model.visual, "proj"):
        lora_params.extend(_add_projection_lora(clip_model.visual, "proj", rank, alpha))
    for p in lora_params:
        p.requires_grad_(True)
    return lora_params


def compute_grad_norm(params):
    grads = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack(grads)).item()


class LocalStructureMatcher(nn.Module):
    def __init__(self, clip_model, dim=512, num_slots=8, init_scale=0.05, max_scale=0.5):
        super().__init__()
        self.clip = clip_model
        self.num_slots = num_slots
        self.dim = dim
        patch_dim = clip_model.visual.conv1.out_channels
        self.patch_proj = nn.Linear(patch_dim, dim)
        self.slot_keys = nn.Parameter(torch.randn(num_slots, dim))
        self.role_embed = nn.Parameter(torch.zeros(3, dim))
        self.triple_proj = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.xavier_uniform_(self.slot_keys.unsqueeze(0))
        nn.init.trunc_normal_(self.role_embed, std=0.02)
        init_ratio = min(max(init_scale / max_scale, 1e-4), 1 - 1e-4)
        self.raw_scale = nn.Parameter(torch.tensor(torch.logit(torch.tensor(init_ratio)).item()))
        self.max_scale = max_scale

    def scale(self):
        return torch.sigmoid(self.raw_scale) * self.max_scale

    @torch.no_grad()
    def extract_patch_tokens(self, images):
        visual = self.clip.visual
        x = visual.conv1(images.type(self.clip.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        patch_tokens = visual.ln_post(x[:, 1:, :])
        return patch_tokens.float()

    def image_slots(self, images):
        patch_tokens = self.extract_patch_tokens(images)
        patch_tokens = F.normalize(self.patch_proj(patch_tokens), dim=-1)
        queries = F.normalize(self.slot_keys, dim=-1)
        attn = torch.einsum("kd,bpd->bkp", queries, patch_tokens)
        attn = F.softmax(attn / (self.dim ** 0.5), dim=-1)
        slots = torch.einsum("bkp,bpd->bkd", attn, patch_tokens)
        return F.normalize(slots, dim=-1)

    @torch.no_grad()
    def encode_triple_parts(self, triple_parts, device):
        tokens = torch.cat([clip.tokenize(text, truncate=True) for text in triple_parts]).to(device)
        feats = self.clip.encode_text(tokens)
        return F.normalize(feats.float(), dim=-1)

    def triple_slots(self, candidates, device):
        max_triples = max(1, max(len(candidate.get("triples", [])) for candidate in candidates))
        flat_parts, positions = [], []
        for cand_idx, candidate in enumerate(candidates):
            triples = candidate.get("triples", [])
            for triple_idx, triple in enumerate(triples):
                if len(triple) < 3:
                    continue
                flat_parts.extend([str(triple[0]), str(triple[1]), str(triple[2])])
                positions.append((cand_idx, triple_idx))
        slots = torch.zeros(len(candidates), max_triples, self.dim, device=device)
        mask = torch.zeros(len(candidates), max_triples, dtype=torch.bool, device=device)
        if flat_parts:
            part_feats = self.encode_triple_parts(flat_parts, device).view(-1, 3, self.dim)
            role_feats = part_feats + self.role_embed.unsqueeze(0)
            triple_feats = self.triple_proj(role_feats.reshape(role_feats.size(0), -1))
            triple_feats = F.normalize(triple_feats, dim=-1)
            for feat, (cand_idx, triple_idx) in zip(triple_feats, positions):
                slots[cand_idx, triple_idx] = feat
                mask[cand_idx, triple_idx] = True
        return slots, mask

    def local_scores(self, images, candidates):
        device = images.device
        obj_slots = self.image_slots(images)
        triple_slots, triple_mask = self.triple_slots(candidates, device)
        if obj_slots.size(0) == 1 and triple_slots.size(0) > 1:
            obj_slots = obj_slots.expand(triple_slots.size(0), -1, -1)
        sim = torch.einsum("bkd,bmd->bkm", obj_slots, triple_slots)
        triple_best = sim.max(dim=1).values
        denom = triple_mask.float().sum(dim=1).clamp(min=1.0)
        scores = (triple_best * triple_mask.float()).sum(dim=1) / denom
        scores = torch.where(triple_mask.any(dim=1), scores, torch.zeros_like(scores))
        return scores


class TopKRerankTrainDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        with open(data_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def _resolve_image_path(self, image_path):
        if os.path.isabs(image_path):
            return image_path
        if "/" in image_path:
            return os.path.join(COCO_IMAGE_DIR, image_path)
        return os.path.join(VG_IMAGE_DIR, image_path)

    def __getitem__(self, index):
        item = self.dataset[index]
        image = Image.open(self._resolve_image_path(item["image_path"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "image_id": item["image_id"],
            "candidates": item["candidates"],
        }


def collate_single(batch):
    return batch[0]


def _unordered_phrase(parts):
    words = []
    for part in parts:
        if part:
            words.extend(str(part).split())
    return " ".join(sorted(words))


def make_unordered_sro_inputs(head_inputs, relation_inputs, tail_inputs, pad_id=0, cls_id=101, sep_id=102):
    head_ids = head_inputs["input_ids"]
    rel_ids = relation_inputs["input_ids"]
    tail_ids = tail_inputs["input_ids"]
    original_shape = head_ids.shape
    length = original_shape[-1]
    flat_head = head_ids.reshape(-1, length)
    flat_rel = rel_ids.reshape(-1, length)
    flat_tail = tail_ids.reshape(-1, length)
    merged_rows = []
    for h_row, r_row, t_row in zip(flat_head, flat_rel, flat_tail):
        tokens = []
        for row in (h_row, r_row, t_row):
            valid = row[(row != pad_id) & (row != cls_id) & (row != sep_id)]
            tokens.extend(valid.detach().cpu().tolist())
        tokens = sorted(tokens)[: max(length - 2, 0)]
        row = torch.full((length,), pad_id, dtype=head_ids.dtype, device=head_ids.device)
        if length > 0:
            row[0] = cls_id
        if tokens:
            row[1:1 + len(tokens)] = torch.tensor(tokens, dtype=head_ids.dtype, device=head_ids.device)
        if length > 1:
            row[min(len(tokens) + 1, length - 1)] = sep_id
        merged_rows.append(row)

    merged_ids = torch.stack(merged_rows, dim=0).view(original_shape)
    token_type_ids = torch.zeros_like(merged_ids)
    token_attention = (merged_ids != pad_id).long()

    def build_like(inputs):
        out = {}
        for key, value in inputs.items():
            if key == "input_ids":
                out[key] = merged_ids
            elif key == "token_type_ids":
                out[key] = token_type_ids
            elif key == "attention_mask":
                out[key] = token_attention
            else:
                out[key] = value
        return out

    unordered = build_like(head_inputs)
    return unordered, build_like(relation_inputs), build_like(tail_inputs)


def encode_candidate_triples(candidates, tokenizer, padding_num=6, length=5, unordered_sro=False):
    head_words, relation_words, tail_words, attention_rows = [], [], [], []
    for candidate in candidates:
        triples = candidate.get("triples", [])[:padding_num]
        cur_head, cur_rel, cur_tail = [], [], []
        for triple in triples:
            if unordered_sro:
                unordered = _unordered_phrase(triple[:3])
                cur_head.append(unordered)
                cur_rel.append(unordered)
                cur_tail.append(unordered)
            else:
                cur_head.append(triple[0])
                cur_rel.append(triple[1])
                cur_tail.append(triple[2])
        valid_len = len(cur_head)
        cur_head += [""] * (padding_num - valid_len)
        cur_rel += [""] * (padding_num - valid_len)
        cur_tail += [""] * (padding_num - valid_len)
        head_words.extend(cur_head)
        relation_words.extend(cur_rel)
        tail_words.extend(cur_tail)
        attention_rows.append([1] * valid_len + [0] * (padding_num - valid_len))

    batch_size = len(candidates)
    head_inputs = tokenizer.batch_encode_plus(
        head_words, max_length=length, add_special_tokens=True,
        padding="max_length", return_tensors="pt", truncation=True
    )
    relation_inputs = tokenizer.batch_encode_plus(
        relation_words, max_length=length, add_special_tokens=True,
        padding="max_length", return_tensors="pt", truncation=True
    )
    tail_inputs = tokenizer.batch_encode_plus(
        tail_words, max_length=length, add_special_tokens=True,
        padding="max_length", return_tensors="pt", truncation=True
    )
    for inputs in (head_inputs, relation_inputs, tail_inputs):
        for key, value in inputs.items():
            inputs[key] = value.view(batch_size, 1, padding_num, length)
    attention_mask = torch.tensor(attention_rows, dtype=torch.long).view(batch_size, 1, padding_num)
    return head_inputs, relation_inputs, tail_inputs, attention_mask


# ─────────────────────────────────────────────────────────────────────
# ① 开关：True = 走 triple-transformer（默认）；False = 关闭 KG 模块
# ─────────────────────────────────────────────────────────────────────
USE_TRIPLE_TRANSFORMER = True
# ─────────────────────────────────────────────────────────────────────

args = get_args()
for name, value in {
    "structure_residual_scale": 0.1,
    "structure_residual_max_scale": 0.2,
    "structure_lr": 1e-4,
    "kg_lr": 1e-5,
    "freeze_triple_bert": False,
    "structure_loss_weight": 1.0,
    "structure_ce_weight": 1.0,
    "structure_margin": 0.1,
    "structure_batch_ce_weight": 0.0,
    "structure_delta_l2_weight": 1e-2,
    "structure_delta_gap_weight": 1e-2,
    "structure_gate_tau": 0.05,
    "structure_gate_max": 0.3,
    "direct_score_prediction": False,
    "uniform_residual_weighting": False,
    "unordered_sro": False,
    "structure_gate_l1_weight": 1e-2,
    "structure_easy_gap": 0.05,
    "structure_easy_gate_weight": 0.05,
    "rerank_train_path": "data/train_coco_topk_rerank.json",
    "rerank_train_topk": 16,
    "rerank_loss_weight": 1.0,
    "rerank_pair_weight": 1.0,
    "semantic_pair_weight": 1.0,
    "cross_pair_weight": 0.05,
    "use_local_structure": True,
    "local_num_slots": 8,
    "local_score_scale": 0.05,
    "local_score_max_scale": 0.5,
    "unfreeze_clip_projection": False,
    "clip_projection_lr": 1e-6,
    "unfreeze_clip_all": False,
    "clip_lr": 1e-7,
    "temp": 0.07,
    "topk_rerank_k": 10,
    "topk_relation_path": "data/visual_genome_relation_topk_rerank.json",
    "topk_attribution_path": "data/visual_genome_attribution_topk_rerank.json",
    "skip_topk_eval": False,
    "eval_interval": 200,
    "save_path": "",
}.items():
    if not hasattr(args, name):
        setattr(args, name, value)
eval_utils.USE_KNOWLEDGE = USE_TRIPLE_TRANSFORMER
wandb.init(project=args.project,
           name=args.name + "_lr" + str(args.lr) + "_weight" + str(args.neg_loss_weight) + "_weight" + str(
               args.knowledge_weight) + "_layernum" + str(args.transformer_layer_num) +
               ("_kg" if USE_TRIPLE_TRANSFORMER else "_clip_only"),
           config={**vars(args), "use_triple_transformer": USE_TRIPLE_TRANSFORMER})
set_manualSeed(args)

# ── 2. 暂时没有数据集格式
idx2id = dict()
with open("data/test_coco_aug_havezero.json", "r") as f:
    infomation = json.load(f)
for idx, item in enumerate(infomation):
    idx2id[idx] = item['id']

# CLIP
clip_model, preprocess = load("ViT-B/32", jit=False)
clip_model = clip_model.cuda()
for p in clip_model.parameters():
    p.requires_grad_(False)
clip_full_params = []
clip_projection_params = []
clip_lora_params = []
if args.unfreeze_clip_all:
    for p in clip_model.parameters():
        p.requires_grad_(True)
    clip_full_params = [p for p in clip_model.parameters() if p.requires_grad]
    print("Unfreeze full CLIP params:", sum(p.numel() for p in clip_full_params), "lr:", args.clip_lr)
elif args.use_clip_lora:
    clip_lora_params = add_clip_projection_lora(
        clip_model,
        rank=args.clip_lora_rank,
        alpha=args.clip_lora_alpha,
        train_text=args.clip_lora_text,
        train_visual=args.clip_lora_visual,
    )
    print(
        "Using CLIP projection LoRA params:",
        sum(p.numel() for p in clip_lora_params),
        "text:", args.clip_lora_text,
        "visual:", args.clip_lora_visual,
        "rank:", args.clip_lora_rank,
        "alpha:", args.clip_lora_alpha,
        "lr:", args.clip_lora_lr,
    )
elif args.unfreeze_clip_projection:
    if hasattr(clip_model, "text_projection") and clip_model.text_projection is not None:
        clip_model.text_projection.requires_grad_(True)
        clip_projection_params.append(clip_model.text_projection)
    if hasattr(clip_model.visual, "proj") and clip_model.visual.proj is not None:
        clip_model.visual.proj.requires_grad_(True)
        clip_projection_params.append(clip_model.visual.proj)
    print("Unfreeze CLIP projection params:", sum(p.numel() for p in clip_projection_params))
else:
    print("Using frozen CLIP backbone: plug-in structure module only.")
clip_model.train() if args.unfreeze_clip_all else clip_model.eval()

# Transformer — USE_TRIPLE_TRANSFORMER=False 时用轻量占位，不加载 BERT
myTransformer = triple_Transformer().cuda() if USE_TRIPLE_TRANSFORMER else nn.Identity().cuda()
for p in myTransformer.parameters():
    p.requires_grad_(USE_TRIPLE_TRANSFORMER)
if USE_TRIPLE_TRANSFORMER and args.freeze_triple_bert and hasattr(myTransformer, "model"):
    for p in myTransformer.model.parameters():
        p.requires_grad_(False)
    if hasattr(myTransformer, "linear"):
        for p in myTransformer.linear.parameters():
            p.requires_grad_(True)
    myTransformer.model.eval()
    print("Freeze triple BERT encoder; train triple projection only.")


def set_triple_encoder_mode(train=True):
    if not USE_TRIPLE_TRANSFORMER:
        myTransformer.eval()
        return
    if train:
        myTransformer.train()
        if args.freeze_triple_bert and hasattr(myTransformer, "model"):
            myTransformer.model.eval()
    else:
        myTransformer.eval()


set_triple_encoder_mode(train=True)

structure_head = StructureResidualHead(
    init_scale=args.structure_residual_scale,
    max_scale=args.structure_residual_max_scale,
    gate_tau=args.structure_gate_tau,
    gate_max=args.structure_gate_max,
    direct_score_prediction=args.direct_score_prediction,
    uniform_residual_weighting=args.uniform_residual_weighting,
).cuda()
if args.direct_score_prediction:
    for p in structure_head.gate_mlp.parameters():
        p.requires_grad_(False)
    print("Use direct score prediction: structural MLP directly predicts the final matching score.")
elif args.uniform_residual_weighting:
    for p in structure_head.gate_mlp.parameters():
        p.requires_grad_(False)
    print("Use uniform residual weighting: fixed residual gate = {:.4f}".format(args.structure_gate_max))
if args.unordered_sro:
    print("Use unordered SRO: subject/relation/object role order is removed before triple encoding.")
eval_utils.STRUCTURE_RESIDUAL_HEAD = structure_head

local_matcher = None
if args.use_local_structure:
    local_matcher = LocalStructureMatcher(
        clip_model,
        num_slots=args.local_num_slots,
        init_scale=args.local_score_scale,
        max_scale=args.local_score_max_scale,
    ).cuda()
eval_utils.LOCAL_STRUCTURE_MATCHER = local_matcher

# dataloader
train_vg_dataset = VG_Attribution(data_path=args.train_path, transform=image_transform())
train_vg_dataloader = DataLoader(train_vg_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True)
rerank_train_dataloader = None
rerank_tokenizer = None
if os.path.exists(args.rerank_train_path):
    rerank_train_dataset = TopKRerankTrainDataset(data_path=args.rerank_train_path, transform=image_transform())
    rerank_train_dataloader = DataLoader(
        rerank_train_dataset, num_workers=4, batch_size=1, shuffle=True, collate_fn=collate_single
    )
    rerank_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("Using grouped rerank training data:", args.rerank_train_path, "num_images:", len(rerank_train_dataset))
else:
    print("Grouped rerank training data not found, fallback to pairwise training:", args.rerank_train_path)
    if local_matcher is not None:
        print("Disable local structure matcher for pairwise fallback because it is only trained by grouped rerank loss.")
        local_matcher = None
        eval_utils.LOCAL_STRUCTURE_MATCHER = None
test_vg_dataset = VG_Attribution(data_path=args.test_path, transform=image_transform(is_train=False))
test_vg_dataloader = DataLoader(test_vg_dataset, num_workers=8, batch_size=args.batch_size, shuffle=False)

test_dataset = CoCoDataset_aug_update(data_path="data/test_coco_aug_havezero.json",
                                      transform=image_transform(is_train=False))
test_dataloader = DataLoader(test_dataset, num_workers=8, batch_size=args.batch_size, shuffle=False)
vg_relation_dataset = VG_Relation(transform=image_transform(is_train=False))
vg_relation_dataloader = DataLoader(vg_relation_dataset, batch_size=128, num_workers=8)

# loss
loss = MarginLoss(margin=0.1)
loss_hingo = WinoLoss(margin=0.1)

def trainable_params(module):
    return [p for p in module.parameters() if p.requires_grad]

# ③ optimizer：默认冻结 CLIP；可选只训练 text_projection / visual.proj。
if USE_TRIPLE_TRANSFORMER:
    param_groups = [
        {'params': trainable_params(myTransformer), 'lr': args.kg_lr},
        {'params': structure_head.parameters(), 'lr': args.structure_lr},
    ]
    if local_matcher is not None:
        param_groups.append({'params': local_matcher.parameters(), 'lr': args.structure_lr})
    if clip_full_params:
        param_groups.append({'params': clip_full_params, 'lr': args.clip_lr, 'weight_decay': 0.0})
    if clip_projection_params:
        param_groups.append({'params': clip_projection_params, 'lr': args.clip_projection_lr, 'weight_decay': 0.0})
    if clip_lora_params:
        param_groups.append({'params': clip_lora_params, 'lr': args.clip_lora_lr, 'weight_decay': 0.0})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
else:
    param_groups = [
        {'params': structure_head.parameters(), 'lr': args.structure_lr},
    ]
    if local_matcher is not None:
        param_groups.append({'params': local_matcher.parameters(), 'lr': args.structure_lr})
    if clip_full_params:
        param_groups.append({'params': clip_full_params, 'lr': args.clip_lr, 'weight_decay': 0.0})
    if clip_projection_params:
        param_groups.append({'params': clip_projection_params, 'lr': args.clip_projection_lr, 'weight_decay': 0.0})
    if clip_lora_params:
        param_groups.append({'params': clip_lora_params, 'lr': args.clip_lora_lr, 'weight_decay': 0.0})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

total_params = sum(p.numel() for p in clip_model.parameters()) + sum(p.numel() for p in myTransformer.parameters()) + sum(p.numel() for p in structure_head.parameters())
trainable_total = sum(p.numel() for group in param_groups for p in group["params"] if p.requires_grad)
trainable_clip = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
trainable_triple = sum(p.numel() for p in myTransformer.parameters() if p.requires_grad)
trainable_structure = sum(p.numel() for p in structure_head.parameters() if p.requires_grad)
print(
    "Parameter count - total:{:.2f}M, trainable:{:.2f}M, clip_trainable:{:.2f}M, triple_trainable:{:.2f}M, structure_trainable:{:.2f}M".format(
        total_params / 1e6,
        trainable_total / 1e6,
        trainable_clip / 1e6,
        trainable_triple / 1e6,
        trainable_structure / 1e6,
    )
)

best_acc_test_attribution, best_acc_test_relation = 0, 0


def save_training_checkpoint(path, epoch):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "args": vars(args),
            "structure_head": structure_head.state_dict(),
            "triple_transformer": myTransformer.state_dict() if USE_TRIPLE_TRANSFORMER else None,
            "clip_model": clip_model.state_dict() if args.unfreeze_clip_all else None,
            "clip_lora": {
                "use_clip_lora": args.use_clip_lora,
                "clip_lora_text": args.clip_lora_text,
                "clip_lora_visual": args.clip_lora_visual,
                "clip_lora_rank": args.clip_lora_rank,
                "clip_lora_alpha": args.clip_lora_alpha,
            },
            "clip_projection": args.unfreeze_clip_projection,
        },
        path,
    )
    print("Saved checkpoint:", path)


def set_clip_trainable(model, trainable):
    for p in model.parameters():
        p.requires_grad_(False)
    if args.unfreeze_clip_all:
        for p in clip_full_params:
            p.requires_grad_(True)
        model.train() if trainable else model.eval()
        return
    if args.unfreeze_clip_projection:
        for p in clip_projection_params:
            p.requires_grad_(True)
    if args.use_clip_lora:
        for p in clip_lora_params:
            p.requires_grad_(True)
    model.eval()


def encode_clip_pair(model, images, text_tokens):
    clip_context = nullcontext() if (args.unfreeze_clip_all or args.unfreeze_clip_projection or args.use_clip_lora) else torch.no_grad()
    with clip_context:
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features.float(), dim=-1)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features.float(), dim=-1)
    return text_features, image_features


def encode_clip_pairwise(model, images, true_text_tokens, false_text_tokens):
    clip_context = nullcontext() if (args.unfreeze_clip_all or args.unfreeze_clip_projection or args.use_clip_lora) else torch.no_grad()
    with clip_context:
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features.float(), dim=-1)
        true_text_features = model.encode_text(true_text_tokens)
        true_text_features = F.normalize(true_text_features.float(), dim=-1)
        false_text_features = model.encode_text(false_text_tokens)
        false_text_features = F.normalize(false_text_features.float(), dim=-1)
    return true_text_features, false_text_features, image_features


def multipositive_ce(scores, labels, temp):
    positive_mask = labels.bool()
    if positive_mask.sum() == 0:
        return None
    logits = scores / temp
    return -(torch.logsumexp(logits[positive_mask], dim=0) - torch.logsumexp(logits, dim=0))


def compute_grouped_rerank_loss(batch, tokenizer):
    image = batch["image"].unsqueeze(0).cuda()
    candidates = batch["candidates"]
    if len(candidates) < 2:
        return None

    labels_all = torch.tensor([candidate["label"] for candidate in candidates], device="cuda", dtype=torch.long)
    captions = [candidate["caption"] for candidate in candidates]

    clip_context = nullcontext() if (args.unfreeze_clip_all or args.unfreeze_clip_projection or args.use_clip_lora) else torch.no_grad()
    with clip_context:
        image_features = clip_model.encode_image(image)
        image_features = F.normalize(image_features.float(), dim=-1)
        text_tokens = torch.cat([clip.tokenize(caption, truncate=True) for caption in captions]).cuda()
        text_features = clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features.float(), dim=-1)
        clip_scores = (image_features * text_features).sum(dim=-1)

    k = min(int(args.rerank_train_topk), clip_scores.numel())
    _, top_idx = torch.topk(clip_scores, k=k, largest=True)
    if labels_all[top_idx].sum().item() == 0 and labels_all.sum().item() > 0:
        best_true_idx = torch.where(labels_all == 1)[0][torch.argmax(clip_scores[labels_all == 1])]
        top_idx[-1] = best_true_idx
        top_idx = torch.unique(top_idx, sorted=False)

    top_candidates = [candidates[idx] for idx in top_idx.detach().cpu().tolist()]
    top_labels = labels_all[top_idx]
    if top_labels.sum().item() == 0 or (top_labels == 0).sum().item() == 0:
        return None

    head_inputs, relation_inputs, tail_inputs, attention_mask = encode_candidate_triples(
        top_candidates, tokenizer, unordered_sro=args.unordered_sro
    )
    if USE_TRIPLE_TRANSFORMER:
        knowledge_features = myTransformer(
            head_inputs, relation_inputs, tail_inputs, 0, attention_mask.cuda()
        )
        knowledge_features = F.normalize(knowledge_features.float(), dim=-1)
    else:
        knowledge_features = torch.zeros_like(text_features[top_idx])

    image_top = image_features.expand(top_idx.numel(), -1)
    text_top = text_features[top_idx]
    clip_top = clip_scores[top_idx]
    clip_confidence = (clip_top.max().detach() - clip_top).clamp(min=0.0)
    scores, deltas, residual_scale, gates = structure_head(
        image_top, text_top, knowledge_features, clip_top, clip_confidence=clip_confidence
    )
    local_score = torch.zeros_like(scores)
    local_scale = torch.zeros((), device=scores.device)
    if local_matcher is not None:
        local_score = local_matcher.local_scores(image, top_candidates)
        local_scale = local_matcher.scale()
        scores = scores + local_scale * gates * local_score

    listwise_loss = multipositive_ce(scores, top_labels, args.temp)
    if listwise_loss is None:
        return None

    pos_scores = scores[top_labels == 1]
    sources = [candidate.get("source", "") for candidate in top_candidates]
    hard_neg_mask = torch.tensor([source == "false" for source in sources], device=scores.device)
    cross_neg_mask = torch.tensor([source == "cross_image" for source in sources], device=scores.device)
    all_neg_mask = top_labels == 0
    if hard_neg_mask.any():
        hard_neg_scores = scores[hard_neg_mask]
    else:
        hard_neg_scores = scores[all_neg_mask]
    semantic_pairwise_loss = F.relu(
        args.structure_margin - (pos_scores[:, None] - hard_neg_scores[None, :])
    ).mean()
    if cross_neg_mask.any():
        cross_neg_scores = scores[cross_neg_mask]
        cross_pairwise_loss = F.relu(
            args.structure_margin - (pos_scores[:, None] - cross_neg_scores[None, :])
        ).mean()
    else:
        cross_pairwise_loss = torch.zeros((), device=scores.device)
    pairwise_loss = (
        args.semantic_pair_weight * semantic_pairwise_loss
        + args.cross_pair_weight * cross_pairwise_loss
    )
    delta_l2_loss = deltas.pow(2).mean()
    if args.direct_score_prediction or args.uniform_residual_weighting:
        gate_l1_loss = torch.zeros((), device=scores.device)
        easy_gate_loss = torch.zeros((), device=scores.device)
    else:
        gate_l1_loss = gates.mean()
        easy_mask = clip_confidence > args.structure_easy_gap
        if easy_mask.any():
            easy_gate_loss = gates[easy_mask].mean()
        else:
            easy_gate_loss = torch.zeros((), device=scores.device)
    total = (
        listwise_loss
        + args.rerank_pair_weight * pairwise_loss
        + args.structure_delta_l2_weight * delta_l2_loss
        + args.structure_gate_l1_weight * gate_l1_loss
        + args.structure_easy_gate_weight * easy_gate_loss
    )

    with torch.no_grad():
        order = torch.argsort(scores, descending=True)
        top1_true = top_labels[order[0]].float()
        hard_false_count = sum(candidate.get("source") == "false" for candidate in top_candidates)
        cross_false_count = sum(candidate.get("source") == "cross_image" for candidate in top_candidates)

    return {
        "loss": total,
        "listwise_loss": listwise_loss.detach(),
        "pairwise_loss": pairwise_loss.detach(),
        "semantic_pairwise_loss": semantic_pairwise_loss.detach(),
        "cross_pairwise_loss": cross_pairwise_loss.detach(),
        "delta_l2_loss": delta_l2_loss.detach(),
        "gate_l1_loss": gate_l1_loss.detach(),
        "easy_gate_loss": easy_gate_loss.detach(),
        "top1_true": top1_true.detach(),
        "num_candidates": torch.tensor(float(len(top_candidates)), device="cuda"),
        "num_hard_false": torch.tensor(float(hard_false_count), device="cuda"),
        "num_cross_false": torch.tensor(float(cross_false_count), device="cuda"),
        "residual_scale": residual_scale.detach(),
        "gate_mean": gates.mean().detach(),
        "local_score": local_score.mean().detach(),
        "local_scale": local_scale.detach(),
    }


# 训练
for epoch in range(args.epoch):
    set_clip_trainable(clip_model, True)
    set_triple_encoder_mode(train=True)
    structure_head.train()
    local_matcher.train() if local_matcher is not None else None

    active_train_dataloader = rerank_train_dataloader if rerank_train_dataloader is not None else train_vg_dataloader
    for i, batch in enumerate(tqdm(active_train_dataloader, total=len(active_train_dataloader))):
        if i % args.eval_interval == 0:
            set_clip_trainable(clip_model, False)
            set_triple_encoder_mode(train=False)
            structure_head.eval()
            local_matcher.eval() if local_matcher is not None else None
            # eval task1
            t1, t5, t10, i1, i5, i10 = eval_utils.eval_coco_large(
                clip_model, myTransformer, test_dataloader, idx2id, args
            )
            wandb.log({
                "TextRank1": t1,
                "TextRank5": t5,
                "TextRank10": t10,
                "ImageRank1": i1,
                "ImageRank5": i5,
                "ImageRank10": i10,
            })

            acc_test_attribution = eval_utils.test_vg_attribution(
                clip_model, myTransformer, test_vg_dataloader, test_vg_dataset, args
            )
            wandb.log({"acc_test_attribution": acc_test_attribution})

            acc_test_relation = eval_utils.test_vg_relation(
                clip_model, myTransformer, vg_relation_dataloader, vg_relation_dataset, args
            )
            wandb.log({"acc_test_relation": acc_test_relation})

            if not args.skip_topk_eval:
                topk_attr_metrics = eval_utils.test_vg_topk_rerank(
                    clip_model, myTransformer, args.topk_attribution_path, args, task_name="attribute"
                )
                if topk_attr_metrics:
                    wandb.log(topk_attr_metrics)
                topk_rel_metrics = eval_utils.test_vg_topk_rerank(
                    clip_model, myTransformer, args.topk_relation_path, args, task_name="relation"
                )
                if topk_rel_metrics:
                    wandb.log(topk_rel_metrics)
            set_triple_encoder_mode(train=True)
            set_clip_trainable(clip_model, True)
            structure_head.train()
            local_matcher.train() if local_matcher is not None else None

        if rerank_train_dataloader is not None:
            set_clip_trainable(clip_model, True)
            set_triple_encoder_mode(train=True)
            structure_head.train()
            local_matcher.train() if local_matcher is not None else None
            optimizer.zero_grad()

            grouped_loss = compute_grouped_rerank_loss(batch, rerank_tokenizer)
            if grouped_loss is None:
                optimizer.zero_grad(set_to_none=True)
                continue

            total_loss = args.rerank_loss_weight * grouped_loss["loss"]
            if not torch.isfinite(total_loss):
                print("Skip non-finite grouped loss at epoch {}, step {}: {}".format(epoch, i, total_loss.item()))
                optimizer.zero_grad(set_to_none=True)
                continue
            total_loss.backward()
            clip_lora_grad = compute_grad_norm(clip_lora_params)
            grad_norm_value = torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
                max_norm=1.0
            )
            if not torch.isfinite(grad_norm_value):
                print("Skip non-finite grouped grad at epoch {}, step {}: grad_norm={}".format(epoch, i, grad_norm_value))
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

            if i % 10 == 0:
                print(
                    'Epoch:{}, step:{}, grouped_loss:{:.4f}, listwise:{:.4f}, pair:{:.4f}, semantic_pair:{:.4f}, cross_pair:{:.4f}, top1:{:.4f}, cand:{:.0f}, hard:{:.0f}, cross:{:.0f}, scale:{:.4f}, gate:{:.4f}, local:{:.4f}, local_scale:{:.4f}'.format(
                        epoch, i, total_loss.item(),
                        grouped_loss["listwise_loss"].item(),
                        grouped_loss["pairwise_loss"].item(),
                        grouped_loss["semantic_pairwise_loss"].item(),
                        grouped_loss["cross_pairwise_loss"].item(),
                        grouped_loss["top1_true"].item(),
                        grouped_loss["num_candidates"].item(),
                        grouped_loss["num_hard_false"].item(),
                        grouped_loss["num_cross_false"].item(),
                        grouped_loss["residual_scale"].item(),
                        grouped_loss["gate_mean"].item(),
                        grouped_loss["local_score"].item(),
                        grouped_loss["local_scale"].item(),
                    )
                )
                wandb.log({
                    "Loss": total_loss.item(),
                    "grouped_rerank_loss": grouped_loss["loss"].item(),
                    "grouped_listwise_loss": grouped_loss["listwise_loss"].item(),
                    "grouped_pairwise_loss": grouped_loss["pairwise_loss"].item(),
                    "grouped_semantic_pairwise_loss": grouped_loss["semantic_pairwise_loss"].item(),
                    "grouped_cross_pairwise_loss": grouped_loss["cross_pairwise_loss"].item(),
                    "grouped_delta_l2_loss": grouped_loss["delta_l2_loss"].item(),
                    "grouped_gate_l1_loss": grouped_loss["gate_l1_loss"].item(),
                    "grouped_easy_gate_loss": grouped_loss["easy_gate_loss"].item(),
                    "grouped_train_top1_true": grouped_loss["top1_true"].item(),
                    "grouped_num_candidates": grouped_loss["num_candidates"].item(),
                    "grouped_num_hard_false": grouped_loss["num_hard_false"].item(),
                    "grouped_num_cross_false": grouped_loss["num_cross_false"].item(),
                    "structure_residual_scale": grouped_loss["residual_scale"].item(),
                    "structure_gate_mean": grouped_loss["gate_mean"].item(),
                    "local_structure_score": grouped_loss["local_score"].item(),
                    "local_structure_scale": grouped_loss["local_scale"].item(),
                })
            continue

        set_clip_trainable(clip_model, True)
        set_triple_encoder_mode(train=True)
        structure_head.train()
        local_matcher.train() if local_matcher is not None else None
        optimizer.zero_grad()

        img        = batch["image_options"][0].cuda()
        text_true  = batch["caption_options"][1].squeeze(1).cuda()
        text_false = batch["caption_options"][0].squeeze(1).cuda()

        # ⑤ KG 模块：USE=False 时传 zero
        if USE_TRIPLE_TRANSFORMER:
            head_inputs = batch["head_inputs"]
            relation_inputs = batch["relation_inputs"]
            tail_inputs = batch["tail_inputs"]
            reversed_head_inputs = batch["reversed_head_inputs"]
            reversed_relation_inputs = batch["reversed_relation_inputs"]
            reversed_tail_inputs = batch["reversed_tail_inputs"]
            if args.unordered_sro:
                head_inputs, relation_inputs, tail_inputs = make_unordered_sro_inputs(
                    head_inputs, relation_inputs, tail_inputs
                )
                reversed_head_inputs, reversed_relation_inputs, reversed_tail_inputs = make_unordered_sro_inputs(
                    reversed_head_inputs, reversed_relation_inputs, reversed_tail_inputs
                )
            knowledge_emb = myTransformer(
                head_inputs, relation_inputs,
                tail_inputs, 0, batch["attention_mask"].cuda()
            )
            knowledge_emb = F.normalize(knowledge_emb, dim=1)
            reversed_knowledge_emb = myTransformer(
                reversed_head_inputs, reversed_relation_inputs,
                reversed_tail_inputs, 0, batch["reversed_attention_mask"].cuda()
            )
            reversed_knowledge_emb = F.normalize(reversed_knowledge_emb, dim=1)
        else:
            # 取 text_true 的 batch size 来对齐零向量
            knowledge_emb = torch.zeros(
                text_true.size(0), 512, device=text_true.device
            )
            reversed_knowledge_emb = knowledge_emb

        text_features, reversed_text_features, image_features = encode_clip_pairwise(
            clip_model, img, text_true, text_false
        )

        clip_score_true = (image_features * text_features).sum(dim=-1)
        clip_score_false = (image_features * reversed_text_features).sum(dim=-1)
        pair_clip_confidence = torch.abs(clip_score_true.detach() - clip_score_false.detach())
        score_true, delta_true, residual_scale, gate_true = structure_head(
            image_features, text_features, knowledge_emb, clip_score_true,
            clip_confidence=pair_clip_confidence
        )
        score_false, delta_false, _, gate_false = structure_head(
            image_features, reversed_text_features, reversed_knowledge_emb, clip_score_false,
            clip_confidence=pair_clip_confidence
        )

        structure_margin_loss = F.relu(args.structure_margin - (score_true - score_false)).mean()
        structure_logits = torch.stack([score_false, score_true], dim=1) / args.temp
        structure_labels = torch.ones(structure_logits.size(0), device=structure_logits.device, dtype=torch.long)
        structure_ce_loss = F.cross_entropy(structure_logits, structure_labels)
        delta_l2_loss = 0.5 * (delta_true.pow(2).mean() + delta_false.pow(2).mean())
        delta_gap_loss = (delta_true - delta_false).pow(2).mean()
        if args.direct_score_prediction or args.uniform_residual_weighting:
            gate_l1_loss = torch.zeros((), device=structure_margin_loss.device)
            easy_gate_loss = torch.zeros((), device=structure_margin_loss.device)
        else:
            gate_l1_loss = 0.5 * (gate_true.mean() + gate_false.mean())
            easy_pair_mask = pair_clip_confidence > args.structure_easy_gap
            if easy_pair_mask.any():
                easy_gate_loss = 0.5 * (
                    gate_true[easy_pair_mask].mean() + gate_false[easy_pair_mask].mean()
                )
            else:
                easy_gate_loss = torch.zeros((), device=structure_margin_loss.device)

        if args.structure_batch_ce_weight > 0:
            batch_size = image_features.size(0)
            image_matrix = image_features[:, None, :].expand(batch_size, batch_size, -1).reshape(-1, image_features.size(-1))
            text_matrix = text_features[None, :, :].expand(batch_size, batch_size, -1).reshape(-1, text_features.size(-1))
            knowledge_matrix = knowledge_emb[None, :, :].expand(batch_size, batch_size, -1).reshape(-1, knowledge_emb.size(-1))
            clip_score_matrix = (image_matrix * text_matrix).sum(dim=-1)
            batch_clip_confidence = torch.zeros_like(clip_score_matrix)
            batch_score_matrix, _, _, _ = structure_head(
                image_matrix, text_matrix, knowledge_matrix, clip_score_matrix,
                clip_confidence=batch_clip_confidence
            )
            batch_score_matrix = batch_score_matrix.view(batch_size, batch_size)
            batch_labels = torch.arange(batch_size, device=batch_score_matrix.device)
            batch_ce_loss = 0.5 * (
                F.cross_entropy(batch_score_matrix / args.temp, batch_labels) +
                F.cross_entropy(batch_score_matrix.t() / args.temp, batch_labels)
            )
        else:
            batch_ce_loss = torch.zeros((), device=structure_margin_loss.device)

        structure_loss = (
            structure_margin_loss
            + args.structure_ce_weight * structure_ce_loss
            + args.structure_batch_ce_weight * batch_ce_loss
            + args.structure_delta_l2_weight * delta_l2_loss
            + args.structure_delta_gap_weight * delta_gap_loss
            + args.structure_gate_l1_weight * gate_l1_loss
            + args.structure_easy_gate_weight * easy_gate_loss
        )

        total_loss = args.structure_loss_weight * structure_loss
        if not torch.isfinite(total_loss):
            print("Skip non-finite loss at epoch {}, step {}: {}".format(epoch, i, total_loss.item()))
            optimizer.zero_grad(set_to_none=True)
            continue
        total_loss.backward()
        clip_lora_grad = compute_grad_norm(clip_lora_params)
        grad_norm_value = torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
            max_norm=1.0
        )
        if not torch.isfinite(grad_norm_value):
            print("Skip non-finite grad at epoch {}, step {}: grad_norm={}".format(epoch, i, grad_norm_value))
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()

        if i % 10 == 0:
            score_gap = (score_true - score_false).mean().item()
            clip_gap = (clip_score_true - clip_score_false).mean().item()
            delta_gap = (delta_true - delta_false).mean().item()
            gate_mean = 0.5 * (gate_true.mean().item() + gate_false.mean().item())
            pairwise_acc = (score_true > score_false).float().mean().item()
            print(
                'Epoch:{}, step:{}, loss:{:.4f}, structure:{:.4f}, margin:{:.4f}, hard_ce:{:.4f}, batch_ce:{:.4f}, gate_l1:{:.4f}, easy_gate:{:.4f}, pair_acc:{:.4f}, gap:{:.4f}, clip_gap:{:.4f}, delta_gap:{:.4f}, scale:{:.4f}, gate:{:.4f}, lora_grad:{:.6f}'.format(
                    epoch, i, total_loss.item(), structure_loss.item(),
                    structure_margin_loss.item(), structure_ce_loss.item(), batch_ce_loss.item(),
                    gate_l1_loss.item(), easy_gate_loss.item(),
                    pairwise_acc, score_gap, clip_gap, delta_gap, residual_scale.item(), gate_mean,
                    clip_lora_grad
                )
            )
            wandb.log({
                "Loss": total_loss.item(),
                "clip_trainable": 0.0,
                "structure_loss": structure_loss.item(),
                "structure_margin_loss": structure_margin_loss.item(),
                "structure_ce_loss": structure_ce_loss.item(),
                "structure_batch_ce_loss": batch_ce_loss.item(),
                "structure_delta_l2_loss": delta_l2_loss.item(),
                "structure_delta_gap_loss": delta_gap_loss.item(),
                "structure_gate_l1_loss": gate_l1_loss.item(),
                "structure_easy_gate_loss": easy_gate_loss.item(),
                "train_pairwise_acc": pairwise_acc,
                "structure_score_gap": score_gap,
                "clip_score_gap": clip_gap,
                "structure_delta_gap": delta_gap,
                "structure_residual_scale": residual_scale.item(),
                "structure_gate_mean": gate_mean,
                "clip_lora_grad_norm": clip_lora_grad,
            })

    print('----------------------this is {}_th epoch----------------------------'.format(epoch))
    save_training_checkpoint(args.save_path, epoch)
