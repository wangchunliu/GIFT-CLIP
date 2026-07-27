"""
eval_version3.py
----------------
Evaluation utilities for PatchGraphCLIP.

Key change: get_retrieval_scores_batched is rewritten to use the new
3-argument score_image_text(images, texts, sg_input) signature and the
shared build_sg_input helper.
"""
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
from utils import image_transform, compute_logits, WinoLoss
from dataloader import Mydataset
from clip import load, tokenize
from PIL import Image
from tqdm import tqdm
import numpy as np
import clip
from collections import defaultdict


# ---------------------------------------------------------------------------
# Scene-graph input builder (identical copy from train_version3.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 需要的辅助：build_sg_input（如果文件顶部已有，不用重复加）
# ---------------------------------------------------------------------------
def build_sg_input(head_enc, relation_enc, tail_enc, attn_mask, device):
    head_ids = head_enc['input_ids'].to(device)
    rel_ids = relation_enc['input_ids'].to(device)
    tail_ids = tail_enc['input_ids'].to(device)
    stacked = torch.stack([head_ids, rel_ids, tail_ids], dim=2)  # [B, N, 3, W]
    valid_mask = attn_mask.squeeze(1).to(device)                 # [B, N], 1=valid
    padding_mask = (valid_mask == 0)                              # True=padding
    return {'input_ids': stacked, 'padding_mask': padding_mask}


# ---------------------------------------------------------------------------
# get_retrieval_scores_batched：原始 5 参数签名
#   get_retrieval_scores_batched(clip_model, myTransformer, dataloader, relation, args)
# ---------------------------------------------------------------------------
def get_retrieval_scores_batched(clip_model, myTransformer, joint_loader, relation, args):
    clip_model.eval()
    myTransformer.eval()
    is_new_model = hasattr(myTransformer, 'score_image_text')

    scores = []
    for batch in tqdm(joint_loader):
        with torch.no_grad():
            if is_new_model:
                img = batch["image_options"][0].cuda()
                text_token = torch.cat([clip.tokenize(c) for c in batch["caption_options"]]).cuda()
                B_img = img.size(0)

                if "reversed_head_inputs" not in batch:
                    raise ValueError(
                        "batch 缺少 reversed_head_inputs 等字段，无法构造 false_triples 对应的 sg_input"
                    )

                # caption_options = [false_caption_batch, true_caption_batch]
                # text_token 拼接顺序 = [false_b0...false_bB-1, true_b0...true_bB-1]
                false_sg = build_sg_input(
                    batch["reversed_head_inputs"], batch["reversed_relation_inputs"],
                    batch["reversed_tail_inputs"], batch["reversed_attention_mask"],
                    device=img.device
                )
                true_sg = build_sg_input(
                    batch["head_inputs"], batch["relation_inputs"],
                    batch["tail_inputs"], batch["attention_mask"],
                    device=img.device
                )
                sg_input = {
                    'input_ids': torch.cat([false_sg['input_ids'], true_sg['input_ids']], dim=0),
                    'padding_mask': torch.cat([false_sg['padding_mask'], true_sg['padding_mask']], dim=0),
                }

                # text_token / sg_input 的顺序是 [false_batch, true_batch]。
                # 因此图片也必须按同样的分组顺序重复，避免 score_image_text 内部
                # 变成 [img0,img0,img1,img1,...] 后与文本错位配对。
                img_pair = torch.cat([img, img], dim=0)
                score = myTransformer.score_image_text(img_pair, text_token, sg_input)  # [2*B_img]
                false_scores = score[:B_img]      # [B_img]
                true_scores  = score[B_img:]      # [B_img]
                pair_scores = torch.stack([false_scores, true_scores], dim=1)  # [B_img, 2]
                scores.append(pair_scores.cpu().numpy().reshape(B_img, 1, 2))
            else:
                img_feat = clip_model.encode_image(batch["image_options"][0].cuda())
                text_feat = clip_model.encode_text(
                    torch.cat([clip.tokenize(c) for c in batch["caption_options"]]).cuda()
                )
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                B_img = img_feat.size(0)
                sim = img_feat @ text_feat.T   # [B_img, 2*B_img]
                # 同样需要按样本配对 false/true（对角线附近），保持和 is_new_model 分支一致的输出形状
                false_sim = torch.diagonal(sim[:, :B_img])
                true_sim  = torch.diagonal(sim[:, B_img:])
                pair_scores = torch.stack([false_sim, true_sim], dim=1)
                scores.append(pair_scores.cpu().numpy().reshape(B_img, 1, 2))

    return np.concatenate(scores, axis=0)   # [N_total, 1, 2]



# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_coco_large(clip_model, myTransformer, dataloader, idx2id, args,
                     use_fused_scoring=False, chunk_size=256, max_eval_samples=None):
    """
    调用方式与原脚本完全一致：
        eval_coco_large(clip_model, myTransformer, test_dataloader, idx2id, args)
    返回值不变：[TextRank1, TextRank5, TextRank10, ImageRank1, ImageRank5, ImageRank10]（比例）

    use_fused_scoring=False（默认）：
        纯 CLIP embedding 检索，速度快，不使用新模型，用于每 200 步的常规训练监控。

    use_fused_scoring=True：
        1. 先用 CLIP embedding 做全量粗排，取每个 image query 的 top-chunk_size 候选文本；
        2. 只对这些候选，用 myTransformer.score_image_text(images, texts, sg_input) 做
           真正的融合重排（cross-attention 精细打分）；
        3. 用重排后的分数覆盖对应位置，其余候选维持粗排分数。
        开销约 O(N * chunk_size)，比全量 O(N*M) 小得多，但仍明显慢于快速模式。
        建议只在整个 epoch 结束后单独调用一次，或配合 max_eval_samples 跑子集：
            eval_coco_large(clip_model, myTransformer, test_dataloader, idx2id, args,
                             use_fused_scoring=True, chunk_size=256, max_eval_samples=1000)
        注意：若正确答案本身没有进入 CLIP 粗排的 top-chunk_size，重排也无法找回它，
        R@K 的上限受粗排 recall 约束。
    """
    clip_model.eval()
    myTransformer.eval() if hasattr(myTransformer, 'eval') else None
    start = time.time()

    is_new_model = hasattr(myTransformer, 'score_image_text')

    text_embeddings, img_embeddings = [], []
    # use_fused_scoring=True 时，额外保留原始 tensor（放 cpu），供重排阶段使用
    raw_texts, raw_imgs, raw_heads, raw_rels, raw_tails, raw_masks = [], [], [], [], [], []
    seen_samples = 0

    print('loading data')
    for i, batch in enumerate(dataloader):
        img, text, head_inputs, relation_inputs, tail_inputs, token_type_ids, attention_mask = batch
        if max_eval_samples is not None:
            remaining = max_eval_samples - seen_samples
            if remaining <= 0:
                break
            if img.size(0) > remaining:
                img = img[:remaining]
                text = text[:remaining]
                head_inputs = {k: v[:remaining] for k, v in head_inputs.items()}
                relation_inputs = {k: v[:remaining] for k, v in relation_inputs.items()}
                tail_inputs = {k: v[:remaining] for k, v in tail_inputs.items()}
                attention_mask = attention_mask[:remaining]

        img_cuda = img.cuda()
        text_cuda = text.squeeze(1).cuda()

        with torch.no_grad():
            text_feat = clip_model.encode_text(text_cuda)
            text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
            img_feat = clip_model.encode_image(img_cuda)
            img_feat = img_feat / img_feat.norm(dim=1, keepdim=True)

        text_embeddings.append(text_feat.float().cpu())
        img_embeddings.append(img_feat.float().cpu())
        seen_samples += img.size(0)

        if use_fused_scoring and is_new_model:
            raw_texts.append(text.squeeze(1).cpu())
            raw_imgs.append(img.cpu())
            raw_heads.append(head_inputs['input_ids'].cpu())
            raw_rels.append(relation_inputs['input_ids'].cpu())
            raw_tails.append(tail_inputs['input_ids'].cpu())
            raw_masks.append(attention_mask.cpu())

        if max_eval_samples is not None and seen_samples >= max_eval_samples:
            break

    print("loading success")
    text_embedding = torch.cat(text_embeddings, dim=0)
    img_embedding = torch.cat(img_embeddings, dim=0)
    num = min(text_embedding.size(0), img_embedding.size(0), len(idx2id))
    if num == 0:
        raise ValueError("eval_coco_large received no samples to evaluate")
    if num < text_embedding.size(0):
        text_embedding = text_embedding[:num]
        img_embedding = img_embedding[:num]

    idx2id_eval = [idx2id[i] for i in range(num)]

    text_sim = text_embedding @ img_embedding.T
    img_sim  = img_embedding  @ text_embedding.T

    # ------------------------------------------------------------------
    # 融合重排（新模型真正被使用的地方）
    # ------------------------------------------------------------------
    if use_fused_scoring and is_new_model:
        print("[eval_coco_large] use_fused_scoring=True: retrieve-then-rerank，速度较慢。")
        device = 'cuda'
        all_texts    = torch.cat(raw_texts, dim=0)[:num]      # [N, 77]  tokenized，未 encode
        all_imgs     = torch.cat(raw_imgs,  dim=0)[:num]      # [N, C, H, W]
        all_head_ids = torch.cat(raw_heads, dim=0)[:num]      # [N, 6, 5]
        all_rel_ids  = torch.cat(raw_rels,  dim=0)[:num]
        all_tail_ids = torch.cat(raw_tails, dim=0)[:num]
        all_masks    = torch.cat(raw_masks, dim=0)[:num]      # [N, 1, 6]

        N = img_sim.size(0)
        topk = min(chunk_size, N)

        img_sim_rerank = img_sim.clone()
        for qi in tqdm(range(N), desc="fused rerank"):
            _, cand_idx = torch.topk(img_sim[qi], k=topk)   # [topk] 候选文本索引

            img_rep = all_imgs[qi:qi+1].expand(topk, -1, -1, -1).to(device)  # [topk, C,H,W]
            text_cand = all_texts[cand_idx].to(device)                       # [topk, 77]

            head_c = all_head_ids[cand_idx].to(device)   # [topk, 6, 5]
            rel_c  = all_rel_ids[cand_idx].to(device)
            tail_c = all_tail_ids[cand_idx].to(device)
            mask_c = all_masks[cand_idx].to(device)       # [topk, 1, 6]

            stacked = torch.stack([head_c, rel_c, tail_c], dim=2)  # [topk, 6, 3, 5]
            padding_mask = (mask_c.squeeze(1) == 0)                # [topk, 6] True=padding
            sg_input = {'input_ids': stacked, 'padding_mask': padding_mask}

            with torch.no_grad():
                cand_scores = myTransformer.score_image_text(img_rep, text_cand, sg_input)  # [topk]

            img_sim_rerank[qi, cand_idx] = cand_scores.cpu()

        img_sim = img_sim_rerank
        text_sim = img_sim.T  # 用重排后矩阵的转置近似 text→image 方向，避免再算一次对称重排

    # ------------------------------------------------------------------
    # R@K 计算（与原脚本完全一致，未改动）
    # ------------------------------------------------------------------
    TextRank1, TextRank5, TextRank10 = 0, 0, 0
    ImageRank1, ImageRank5, ImageRank10 = 0, 0, 0

    img_set = set()
    img_unq = []
    img_mask = []
    for i in tqdm(range(num)):
        cur_id = idx2id_eval[i]
        if cur_id in img_set:
            img_mask.append(0)
            continue
        img_set.add(cur_id)
        img_unq.append(i)
        img_mask.append(1)

        _, sorted_idx = torch.topk(img_sim[i], k=min(10, num))
        sorted_idx = sorted_idx.numpy().tolist()

        flag1, flag5, flag10 = 1, 1, 1
        for j, idx in enumerate(sorted_idx):
            if j < 1 and flag1 and idx2id_eval[idx] == cur_id:
                ImageRank1  += 1; flag1  = 0
            if j < 5 and flag5 and idx2id_eval[idx] == cur_id:
                ImageRank5  += 1; flag5  = 0
            if j < 10 and flag10 and idx2id_eval[idx] == cur_id:
                ImageRank10 += 1; flag10 = 0

    img_num = len(img_set)

    img_mask_tensor = torch.tensor(img_mask, dtype=torch.bool)
    for i in tqdm(range(num)):
        cur_id = idx2id_eval[i]
        masked_scores = text_sim[i].masked_fill(~img_mask_tensor, -float("inf"))
        _, sorted_idx = torch.topk(masked_scores, k=min(10, num))
        sorted_idx = sorted_idx.numpy().tolist()

        flag1, flag5, flag10 = 1, 1, 1
        for j, idx in enumerate(sorted_idx):
            if j < 1 and flag1 and idx2id_eval[idx] == cur_id:
                TextRank1  += 1; flag1  = 0
            if j < 5 and flag5 and idx2id_eval[idx] == cur_id:
                TextRank5  += 1; flag5  = 0
            if flag10 and idx2id_eval[idx] == cur_id:
                TextRank10 += 1; flag10 = 0

    end = time.time()
    print("Consuming {:.2f} seconds".format(end - start))
    print("TextRank1:{:.4f}, TextRank5:{:.4f}, TextRank10:{:.4f}".format(
        TextRank1 / num, TextRank5 / num, TextRank10 / num))
    print("ImageRank1:{:.4f}, ImageRank5:{:.4f}, ImageRank10:{:.4f}".format(
        ImageRank1 / img_num, ImageRank5 / img_num, ImageRank10 / img_num))

    return [TextRank1 / num, TextRank5 / num, TextRank10 / num,
            ImageRank1 / img_num, ImageRank5 / img_num, ImageRank10 / img_num]



def _recall_at_k(sim_matrix, k=1):
    """sim_matrix: [N_query, N_gallery], ground-truth is diagonal."""
    n = sim_matrix.shape[0]
    correct = 0
    for i in range(n):
        top_k = np.argsort(sim_matrix[i])[::-1][:k]
        if i in top_k:
            correct += 1
    return correct / n


def _get_caption_by_index(all_batches, global_idx):
    """Retrieve a raw caption string by its global flat index."""
    offset = 0
    for b in all_batches:
        caps = b["caption_options"]
        cap  = caps[0] if isinstance(caps, (list, tuple)) else caps
        if isinstance(cap, torch.Tensor):
            B = cap.size(0)
        else:
            B = len(cap)
        if global_idx < offset + B:
            local_idx = global_idx - offset
            if isinstance(cap, torch.Tensor):
                # Should be strings in eval mode; fall back to empty str
                return ""
            return cap[local_idx]
        offset += B
    return ""

# ---------------------------------------------------------------------------
# macroacc_evaluation / macroacc_evaluation_attribute：原始签名
#   (scores, dataset, pos_idx=1)，按 relation/attribute 类别做 macro 平均
# ---------------------------------------------------------------------------
def macroacc_evaluation(scores, dataset, pos_idx=1):
    metrics = {"Accuracy": None}
    if scores.ndim == 3:
        preds = np.argmax(np.squeeze(scores, axis=1), axis=-1)
    else:
        preds = np.argmax(scores, axis=-1)

    correct_mask = (preds == pos_idx)
    metrics["Accuracy"] = np.mean(correct_mask)

    all_relations = np.array(dataset.all_relations)
    for relation in np.unique(all_relations):
        relation_mask = (all_relations == relation)
        if relation_mask.sum() == 0:
            continue
        metrics[f"{relation}-Acc"] = float(correct_mask[relation_mask].mean())

    return metrics


def macroacc_evaluation_attribute(scores, dataset, pos_idx=1):
    metrics = {"Accuracy": None}
    if scores.ndim == 3:
        preds = np.argmax(np.squeeze(scores, axis=1), axis=-1)
    else:
        preds = np.argmax(scores, axis=-1)

    correct_mask = (preds == pos_idx)
    metrics["Accuracy"] = np.mean(correct_mask)

    all_attributes = np.array(dataset.all_attributes)
    for attribute in np.unique(all_attributes):
        attr_mask = (all_attributes == attribute)
        if attr_mask.sum() == 0:
            continue
        metrics[f"{attribute}-Acc"] = float(correct_mask[attr_mask].mean())

    return metrics


# ---------------------------------------------------------------------------
# test_vg_relation / test_vg_attribution：原始签名
#   test_vg_relation(clip_model, myTransformer, vg_relation_dataloader, vg_relation_dataset, args)
#   test_vg_attribution(clip_model, myTransformer, test_vg_dataloader, test_vg_dataset, args)
# ---------------------------------------------------------------------------
def test_vg_relation(clip_model, myTransformer, dataloader, dataset, args):
    scores = get_retrieval_scores_batched(clip_model, myTransformer, dataloader, "relation", args)
    metrics = macroacc_evaluation(scores, dataset, pos_idx=1)
    flat_scores = np.squeeze(scores, axis=1) if scores.ndim == 3 else scores
    preds = np.argmax(flat_scores, axis=-1)
    residual_scale = myTransformer.residual_scale().item() if hasattr(myTransformer, 'residual_scale') else 0.0
    print(
        "relation overall_acc {:.4f}, pred_true_rate {:.4f}, false_mean {:.4f}, true_mean {:.4f}, residual_scale {:.4f}".format(
            metrics["Accuracy"], float((preds == 1).mean()),
            float(flat_scores[:, 0].mean()), float(flat_scores[:, 1].mean()), residual_scale
        )
    )

    all_accs = [v for k, v in metrics.items() if "-Acc" in k]
    acc_test_relation = float(np.mean(all_accs))
    print("acc_test_relation", acc_test_relation)
    return acc_test_relation


def test_vg_attribution(clip_model, myTransformer, dataloader, dataset, args):
    scores = get_retrieval_scores_batched(clip_model, myTransformer, dataloader, "attribute", args)
    metrics = macroacc_evaluation_attribute(scores, dataset, pos_idx=1)
    flat_scores = np.squeeze(scores, axis=1) if scores.ndim == 3 else scores
    preds = np.argmax(flat_scores, axis=-1)
    residual_scale = myTransformer.residual_scale().item() if hasattr(myTransformer, 'residual_scale') else 0.0
    print(
        "attribute overall_acc {:.4f}, pred_true_rate {:.4f}, false_mean {:.4f}, true_mean {:.4f}, residual_scale {:.4f}".format(
            metrics["Accuracy"], float((preds == 1).mean()),
            float(flat_scores[:, 0].mean()), float(flat_scores[:, 1].mean()), residual_scale
        )
    )

    all_accs = [v for k, v in metrics.items() if "-Acc" in k]
    acc_test_attribution = float(np.mean(all_accs))
    print("acc_test_attribution", acc_test_attribution)
    return acc_test_attribution
