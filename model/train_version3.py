import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import get_args, set_manualSeed, image_transform
from dataloader import CoCoDataset_aug_update
from dataloader_downstream import VG_Relation, VG_Attribution
from clip import load, tokenize
from new_model import PatchGraphCLIP
from eval_version3 import eval_coco_large, test_vg_relation, test_vg_attribution

import wandb
import json


def to_token(x):
    """caption_options 的元素可能是 Tensor / list[str] / list[Tensor]"""
    if isinstance(x, torch.Tensor):
        t = x.cuda()
        if t.dim() == 3 and t.shape[1] == 1:
            t = t.squeeze(1)
        return t
    if isinstance(x, list):
        if len(x) == 0:
            raise ValueError("Empty caption list")
        first = x[0]
        if isinstance(first, str):
            return tokenize([s.strip() for s in x]).cuda()
        if isinstance(first, torch.Tensor):
            return torch.stack(
                [t.squeeze(0) if t.dim() == 2 else t for t in x], dim=0
            ).cuda()
    raise TypeError(f"Unknown caption type: {type(x)}")


def build_sg_input(head_enc, relation_enc, tail_enc, attn_mask, device):
    """
    head_enc/relation_enc/tail_enc: BatchEncoding，['input_ids'] shape 是 [B, N, W] = [B, 6, 5]
    attn_mask: [B, 1, N] tensor，1=真实三元组，0=padding三元组（数据集自带）
    返回: {'input_ids': [B,N,3,W], 'padding_mask': [B,N] bool, True=padding}
    """
    head_ids = head_enc['input_ids'].to(device)
    rel_ids = relation_enc['input_ids'].to(device)
    tail_ids = tail_enc['input_ids'].to(device)
    stacked = torch.stack([head_ids, rel_ids, tail_ids], dim=2)  # [B, N, 3, W]
    valid_mask = attn_mask.squeeze(1).to(device)                 # [B, N], 1=valid
    padding_mask = (valid_mask == 0)                              # True=padding
    return {'input_ids': stacked, 'padding_mask': padding_mask}


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


def grad_norm(params):
    grads = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack(grads)).item()


def compute_loss(out, args, out_false=None):
    TEMP = getattr(args, 'temp', 0.07)
    W_G2G = getattr(args, 'weight_g2g', 1.0)
    W_FUSED = getattr(args, 'weight_fused', 1.0)
    W_L2L = getattr(args, 'weight_l2l', 1.0)
    W_DIV = getattr(args, 'weight_diversity', 0.1)
    W_HARDNEG = getattr(args, 'weight_hardneg', 1.0)
    W_HARDNEG_CE = getattr(args, 'weight_hardneg_ce', 1.0)
    MARGIN = getattr(args, 'hardneg_margin', 0.2)

    def infonce(a, b, temp):
        a = F.normalize(a.float(), dim=-1)
        b = F.normalize(b.float(), dim=-1)
        B = a.size(0)
        logits = a @ b.T / temp
        labels = torch.arange(B, device=a.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    log = {
        'loss_g2g': 0.0,
        'loss_fused': 0.0,
        'loss_l2l': 0.0,
        'loss_diversity': out['diversity_loss'].item(),
        'residual_scale': out.get('residual_scale', torch.tensor(0.0)).item(),
        'graph_scale': out.get('graph_scale', torch.tensor(0.0)).item(),
        'graph_score': out.get('graph_score', torch.tensor(0.0)).float().mean().item(),
        'gate_mean': out['gate_mean'].item(),
    }
    total_loss = out['residual_score'].sum() * 0.0

    if W_G2G > 0:
        loss_g2g = infonce(out['v_global'], out['t_global'], TEMP)
        total_loss = total_loss + W_G2G * loss_g2g
        log['loss_g2g'] = loss_g2g.item()

    if W_FUSED > 0:
        loss_fused = infonce(out['v_fused'], out['t_fused'], TEMP)
        total_loss = total_loss + W_FUSED * loss_fused
        log['loss_fused'] = loss_fused.item()

    if W_L2L > 0:
        v = F.normalize(out['v_local_enh'].float(), dim=-1)
        t = F.normalize(out['t_local_enh'].float(), dim=-1)
        sim = torch.einsum('ikd,jnd->ijkn', v, t)
        sim_mat = (sim.max(dim=-1).values.mean(dim=-1) + sim.max(dim=-2).values.mean(dim=-1)) / 2
        labels = torch.arange(sim_mat.size(0), device=sim_mat.device)
        logits = sim_mat / TEMP
        loss_l2l = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        total_loss = total_loss + W_L2L * loss_l2l
        log['loss_l2l'] = loss_l2l.item()

    if W_DIV > 0:
        total_loss = total_loss + W_DIV * out['diversity_loss']

    # 用与评估一致的 residual score 优化强负样本排序：
    # score = CLIP_score + gamma * (fused_score - CLIP_score)
    if out_false is not None:
        score_true = out['residual_score'].float()
        score_false = out_false['residual_score'].float()

        loss_hard_neg_margin = F.relu(MARGIN - (score_true - score_false)).mean()
        hard_logits = torch.stack([score_false, score_true], dim=1) / TEMP
        hard_labels = torch.ones(hard_logits.size(0), device=hard_logits.device, dtype=torch.long)
        loss_hard_neg_ce = F.cross_entropy(hard_logits, hard_labels)
        loss_hard_neg = loss_hard_neg_margin + W_HARDNEG_CE * loss_hard_neg_ce

        total_loss = total_loss + W_HARDNEG * loss_hard_neg
        log['loss_hard_neg'] = loss_hard_neg.item()
        log['loss_hard_neg_margin'] = loss_hard_neg_margin.item()
        log['loss_hard_neg_ce'] = loss_hard_neg_ce.item()
        log['hard_score_gap'] = (score_true - score_false).mean().item()
        log['clip_score_gap'] = (out['clip_score'].float() - out_false['clip_score'].float()).mean().item()
        log['fused_score_gap'] = (out['fused_score'].float() - out_false['fused_score'].float()).mean().item()
        if 'graph_score' in out and 'graph_score' in out_false:
            log['graph_score_gap'] = (out['graph_score'].float() - out_false['graph_score'].float()).mean().item()

    return total_loss, log


def apply_train_defaults(args):
    """Defaults tuned for the true/false attribution objective."""
    defaults = {
        'temp': 0.07,
        'weight_g2g': 0.0,
        'weight_fused': 0.0,
        'weight_l2l': 0.0,
        'weight_diversity': 0.0,
        'weight_hardneg': 1.0,
        'weight_hardneg_ce': 1.0,
        'hardneg_margin': 0.1,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    return args


args = get_args()
args = apply_train_defaults(args)
wandb.init(
    project=args.project,
    name=args.name + "_DualTower",
    config=vars(args)
)
set_manualSeed(args)


idx2id = dict()
with open("data/test_coco_aug_havezero.json", "r") as f:
    infomation = json.load(f)
for idx, item in enumerate(infomation):
    idx2id[idx] = item['id']


train_vg_dataset = VG_Attribution(data_path=args.train_path, transform=image_transform())
train_vg_dataloader = DataLoader(
    train_vg_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True
)
test_vg_dataset = VG_Attribution(data_path=args.test_path, transform=image_transform(is_train=False))
test_vg_dataloader = DataLoader(
    test_vg_dataset, num_workers=8, batch_size=args.batch_size, shuffle=False
)

test_dataset = CoCoDataset_aug_update(
    data_path="data/test_coco_aug_havezero.json",
    transform=image_transform(is_train=False)
)
test_dataloader = DataLoader(
    test_dataset, num_workers=8, batch_size=args.batch_size, shuffle=False
)
vg_relation_dataset = VG_Relation(transform=image_transform(is_train=False))
vg_relation_dataloader = DataLoader(
    vg_relation_dataset, batch_size=128, num_workers=8
)


clip_model, preprocess = load("ViT-B/32", jit=False)
clip_model = clip_model.cuda()

myTransformer = PatchGraphCLIP(
    clip_model=clip_model,
    clip_dim=512, slot_dim=512, num_slots=5,
    sg_word_dim=64, sg_vocab_size=30522, sg_out_dim=512,
    sg_max_triples=6, sg_pad_token_id=0, fusion_out_dim=512,
    cross_nhead=8, diversity_loss_weight=0.1, sg_max_word_len=5,
).cuda()

lora_params = []
if getattr(args, "use_clip_lora", False):
    lora_params = add_clip_projection_lora(
        clip_model,
        rank=args.clip_lora_rank,
        alpha=args.clip_lora_alpha,
        train_text=args.clip_lora_text,
        train_visual=args.clip_lora_visual,
    )
    myTransformer.train_clip_lora = len(lora_params) > 0
    print("Using CLIP projection LoRA, trainable params:", sum(p.numel() for p in lora_params))

param_groups = [
    {'params': list(myTransformer.dino_cluster.parameters()),    'lr': 5e-4},
    {'params': list(myTransformer.sg_encoder.parameters()),      'lr': 5e-4},
    {'params': list(myTransformer.cross_attn.parameters()),      'lr': 1e-4},
    {'params': list(myTransformer.slot_to_cross.parameters()),   'lr': 1e-4},
    {'params': list(myTransformer.triple_to_cross.parameters()), 'lr': 1e-4},
    {'params': list(myTransformer.part_to_cross.parameters()),   'lr': 1e-4},
    {'params': list(myTransformer.relation_from_objects.parameters()), 'lr': 1e-4},
    {'params': list(myTransformer.fusion.parameters()),          'lr': 1e-4},
    {'params': [
        myTransformer.residual_score_logit,
        myTransformer.graph_score_logit,
        myTransformer.align_temperature,
    ], 'lr': 1e-4},
]
if lora_params:
    param_groups.append({'params': lora_params, 'lr': args.clip_lora_lr, 'weight_decay': 0.0})

optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

TEMP = getattr(args, 'temp', 0.07)
best_acc_test_attribution, best_acc_test_relation = 0, 0


for epoch in range(args.epoch):
    clip_model.eval()
    myTransformer.train()
    clip_model.eval()

    for i, batch in enumerate(tqdm(train_vg_dataloader, total=len(train_vg_dataloader))):

        if i % 200 == 0:
            clip_model.eval()
            myTransformer.eval()

            t1, t5, t10, i1, i5, i10 = eval_coco_large(clip_model, myTransformer, test_dataloader, idx2id, args,
                            use_fused_scoring=True, chunk_size=64, max_eval_samples=1000)

            wandb.log({
                "TextRank1": t1, "TextRank5": t5, "TextRank10": t10,
                "ImageRank1": i1, "ImageRank5": i5, "ImageRank10": i10,
            })

            acc_test_attribution = test_vg_attribution(
                clip_model, myTransformer, test_vg_dataloader, test_vg_dataset, args
            )
            wandb.log({"acc_test_attribution": acc_test_attribution})

            acc_test_relation = test_vg_relation(
                clip_model, myTransformer, vg_relation_dataloader, vg_relation_dataset, args
            )
            wandb.log({"acc_test_relation": acc_test_relation})

            myTransformer.train()
            clip_model.eval()

        img = batch["image_options"][0].cuda()
        text_true_token = to_token(batch["caption_options"][1])
        text_false_token = to_token(batch["caption_options"][0])  # 新增：false caption

        head = batch["head_inputs"]
        relation = batch["relation_inputs"]
        tail = batch["tail_inputs"]
        attn_mask = batch["attention_mask"]
        sg_input_true = build_sg_input(head, relation, tail, attn_mask, device=img.device)

        # 新增：false triples 这一路
        rev_head = batch["reversed_head_inputs"]
        rev_relation = batch["reversed_relation_inputs"]
        rev_tail = batch["reversed_tail_inputs"]
        rev_attn_mask = batch["reversed_attention_mask"]
        sg_input_false = build_sg_input(rev_head, rev_relation, rev_tail, rev_attn_mask, device=img.device)

        out_true = myTransformer(img, text_true_token, sg_input_true)
        out_false = myTransformer(img, text_false_token, sg_input_false)

        total_loss, log = compute_loss(out_true, args, out_false=out_false)

        if not torch.isfinite(total_loss):
            print(
                "Skip non-finite loss at epoch {}, step {}: {}".format(
                    epoch, i, {k: round(v, 6) if isinstance(v, float) else v for k, v in log.items()}
                )
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad()
        total_loss.backward()
        trainable_params = [
            p for group in optimizer.param_groups for p in group["params"] if p.grad is not None
        ]
        lora_grad_norm = grad_norm(lora_params)
        grad_norm_value = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        if not torch.isfinite(grad_norm_value):
            print("Skip non-finite grad at epoch {}, step {}: grad_norm={}".format(epoch, i, grad_norm_value))
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()

        if i % 10 == 0:
            print(
                'Epoch:{}, step:{}, loss:{:.4f}, g2g:{:.4f}, fused:{:.4f}, l2l:{:.4f}, hardneg:{:.4f}, hardce:{:.4f}, gap:{:.4f}, clip_gap:{:.4f}, fused_gap:{:.4f}, graph_gap:{:.4f}, scale:{:.4f}, graph_scale:{:.4f}, div:{:.4f}, gate:{:.4f}, lora_grad:{:.6f}'.format(
                    epoch, i, total_loss.item(),
                    log['loss_g2g'], log['loss_fused'], log['loss_l2l'],
                    log.get('loss_hard_neg', 0.0), log.get('loss_hard_neg_ce', 0.0),
                    log.get('hard_score_gap', 0.0), log.get('clip_score_gap', 0.0),
                    log.get('fused_score_gap', 0.0), log.get('graph_score_gap', 0.0),
                    log.get('residual_scale', 0.0), log.get('graph_scale', 0.0),
                    log['loss_diversity'], log['gate_mean'], lora_grad_norm
                ))
            wandb.log({
                "Loss": total_loss.item(),
                "loss_g2g": log['loss_g2g'],
                "loss_fused": log['loss_fused'],
                "loss_l2l": log['loss_l2l'],
                "loss_diversity": log['loss_diversity'],
                "loss_hard_neg": log.get('loss_hard_neg', 0.0),
                "loss_hard_neg_margin": log.get('loss_hard_neg_margin', 0.0),
                "loss_hard_neg_ce": log.get('loss_hard_neg_ce', 0.0),
                "hard_score_gap": log.get('hard_score_gap', 0.0),
                "clip_score_gap": log.get('clip_score_gap', 0.0),
                "fused_score_gap": log.get('fused_score_gap', 0.0),
                "graph_score": log.get('graph_score', 0.0),
                "graph_score_gap": log.get('graph_score_gap', 0.0),
                "residual_scale": log.get('residual_scale', 0.0),
                "graph_scale": log.get('graph_scale', 0.0),
                "gate_mean": log['gate_mean'],
                "lora_grad_norm": lora_grad_norm,
            })

    print('----------------------this is {}_th epoch----------------------------'.format(epoch))
