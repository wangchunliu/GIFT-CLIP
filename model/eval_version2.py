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
import json
import os
from transformers import BertTokenizer


def eval_coco(clip_model, dataloader):
    clip_model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start = time.time()
    text_embedding, img_embedding = torch.tensor([]), torch.tensor([])
    print('loading data')
    for i, batch in enumerate(dataloader):
        id, img, text_true = batch
        text_true = text_true.squeeze(1).to(device)
        img = img.to(device)
        with torch.no_grad():
            text = clip_model.encode_text(text_true)
            text = text / text.norm(dim=1, keepdim=True)
            img = clip_model.encode_image(img)
            img = img / img.norm(dim=1, keepdim=True)
        if i == 0:
            text_embedding, img_embedding = text, img
            continue
        text_embedding = torch.cat((text_embedding, text), 0)
        img_embedding = torch.cat((img_embedding, img), 0)
    print("loading success")
    text_embedding = text_embedding.to('cpu')
    img_embedding = img_embedding.to('cpu')
    text_sim = text_embedding @ img_embedding.T
    img_sim = img_embedding @ text_embedding.T

    TextRank1, TextRank5, TextRank10 = 0, 0, 0
    ImageRank1, ImageRank5, ImageRank10 = 0, 0, 0
    text_sim = torch.tensor(text_sim.to('cpu'))
    for i in range(1000):
        if i % 100 == 0: print(i)
        res_list = sorted(text_sim[i,], reverse=True)
        rank = res_list.index(text_sim[i][i])
        if rank < 1:
            TextRank1 += 1
        if rank < 5:
            TextRank5 += 1
        if rank < 10:
            TextRank10 += 1
    # print("text completed")

    img_sim = img_sim.to('cpu')
    for i in range(1000):
        if i % 100 == 0: print(i)
        res_list = sorted(img_sim[i,], reverse=True)
        rank = res_list.index(img_sim[i][i])
        if rank < 1:
            ImageRank1 += 1
        if rank < 5:
            ImageRank5 += 1
        if rank < 10:
            ImageRank10 += 1
    end = time.time()
    print("Consuming {:.2f} seconds".format(end - start))
    print("TextRank1:{}, TextRank5:{}, TextRank10:{}".format(TextRank1 / 1000, TextRank5 / 1000, TextRank10 / 1000))
    print(
        "ImageRank1:{}, ImageRank5:{}, ImageRank10:{}".format(ImageRank1 / 1000, ImageRank5 / 1000, ImageRank10 / 1000))
    return [TextRank1 / 1000, TextRank5 / 1000, TextRank10 / 1000, ImageRank1 / 1000, ImageRank5 / 1000,
            ImageRank10 / 1000]


def eval_coco_batch(clip_model, batch):
    clip_model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start = time.time()

    id, img, text = batch
    num = len(id)
    idx2id = dict()
    for i in range(len(id)):
        idx2id[i] = id[i]
    img = img.to(device)
    text = text.squeeze(1).to(device)
    logit_img2text, text_embedding, img_embedding = clip_model(img, text)
    print("loading success")
    text_embedding = text_embedding.to('cpu')
    img_embedding = img_embedding.to('cpu')
    text_sim = text_embedding @ img_embedding.T
    img_sim = img_embedding @ text_embedding.T

    TextRank1, TextRank5, TextRank10 = 0, 0, 0
    ImageRank1, ImageRank5, ImageRank10 = 0, 0, 0
    text_sim = torch.tensor(text_sim.to('cpu'))

    img_sim = img_sim.to('cpu')
    img_set = set()
    img_unq = []
    img_mask = []


    for i in range(num):
        if i % 1000 == 0: print(i)
        cur_id = idx2id[i]
        if cur_id in img_set:
            img_mask.append(0)
            continue
        img_set.add(cur_id)
        img_unq.append(i)
        img_mask.append(1)
        _, sorted_idx = torch.sort(img_sim[i,], descending=True)
        sorted_idx = sorted_idx.numpy().tolist()
        flag1, flag5, flag10 = 1, 1, 1
        for j in range(10):
            if j < 1 and idx2id[sorted_idx[j]] == cur_id and flag1:
                ImageRank1 += 1
                flag1 = 0
            if j < 5 and idx2id[sorted_idx[j]] == cur_id and flag5:
                ImageRank5 += 1
                flag5 = 0
            if j < 10 and idx2id[sorted_idx[j]] == cur_id and flag10:
                ImageRank10 += 1
                flag10 = 0
    img_num = len(img_set)

    for i in range(len(idx2id)):
        if i % 1000 == 0: print(i)
        cur_id = idx2id[i]
        sorted_score, sorted_idx = torch.sort(text_sim[i,].mul(torch.tensor(img_mask)), descending=True)
        sorted_idx = sorted_idx.numpy().tolist()
        flag1, flag5, flag10 = 1, 1, 1
        for j in range(10):
            if j < 1 and idx2id[sorted_idx[j]] == cur_id and flag1:
                TextRank1 += 1
                flag1 = 0
            if j < 5 and idx2id[sorted_idx[j]] == cur_id and flag5:
                TextRank5 += 1
                flag5 = 0
            if idx2id[sorted_idx[j]] == cur_id and flag10:
                TextRank10 += 1
                flag10 = 0

    # print("text completed")

    end = time.time()
    print("-----------------------------------eval_batch-------------------------------------------")
    print("Consuming {:.2f} seconds".format(end - start))
    print("TextRank1:{:.4f}, TextRank5:{:.4f}, TextRank10:{:.4f}".format(TextRank1 / num, TextRank5 / num,
                                                                         TextRank10 / num))
    print("ImageRank1:{:.4f}, ImageRank5:{:.4f}, ImageRank10:{:.4f}".format(ImageRank1 / img_num, ImageRank5 / img_num,
                                                                            ImageRank10 / img_num))
    print("-----------------------------------eval_batch-------------------------------------------")
    return [TextRank1 / num, TextRank5 / num, TextRank10 / num, ImageRank1 / img_num, ImageRank5 / img_num,
            ImageRank10 / img_num]


import time
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────
# 全局开关：True = 注入 KG（默认行为），False = 纯 CLIP
#   建议改成从 args 读：args.kg_module
# ─────────────────────────────────────────────────────────────────────
USE_KNOWLEDGE = False
USE_RESIDUAL_SCORING = True
STRUCTURE_RESIDUAL_HEAD = None
LOCAL_STRUCTURE_MATCHER = None
# ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VG_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "visual_genome_data", "vg_image")


def eval_coco_large(clip_model, myTransformer, dataloader, idx2id, args):
    """
    跨 batch 的图文检索评估（COCO / Flickr30K 风格）。

    Args:
        clip_model    : CLIP，提供 encode_text / encode_image
        myTransformer : triple-transformer（仅当 USE_KNOWLEDGE=True 时才用）
        dataloader    : 每 batch = (img, text, head, rel, tail, tti, am)
        idx2id        : list / array / tuple，长度 = N_txt
        args          : 必须包含 .knowledge_weight（即使 KG 关闭也建议保留）

    Returns:
        list: [TextR@1, TextR@5, TextR@10, ImageR@1, ImageR@5, ImageR@10]
    """
    num = len(idx2id)
    clip_model.eval()
    if USE_KNOWLEDGE:
        myTransformer.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start = time.time()

    text_embedding, clip_text_embedding, img_embedding = torch.tensor([]), torch.tensor([]), torch.tensor([])
    print('loading data')

    for i, batch in enumerate(dataloader):
        img, text, head_inputs, relation_inputs, tail_inputs, token_type_ids, attention_mask = batch

        img = img.cuda()
        text = text.squeeze(1).cuda()
        if USE_KNOWLEDGE:
            attention_mask = attention_mask.cuda()

        with torch.no_grad():
            # ── 1. CLIP 文本向量（无论是否 KG，都要做） ──────
            text_clip = clip_model.encode_text(text)
            text_clip = text_clip / text_clip.norm(dim=1, keepdim=True)
            text = text_clip

            # ── 2. KG 融合（开关关掉时直接跳过） ───────────
            if USE_KNOWLEDGE:
                if getattr(args, "unordered_sro", False):
                    head_inputs, relation_inputs, tail_inputs = make_unordered_sro_inputs(
                        head_inputs, relation_inputs, tail_inputs
                    )
                knowledge_emb = myTransformer(
                    head_inputs, relation_inputs, tail_inputs,
                    token_type_ids, attention_mask
                )
                knowledge_emb = knowledge_emb / knowledge_emb.norm(dim=1, keepdim=True)
                text = F.normalize(text_clip + knowledge_emb * args.knowledge_weight, dim=1)
            # else: text 保持纯 CLIP 输出（已归一化）

            # ── 3. CLIP 图像向量 ────────────────────────
            img = clip_model.encode_image(img)
            img = img / img.norm(dim=1, keepdim=True)

        # ── 4. 累积 ─────────────────────────────────────
        if i == 0:
            text_embedding, img_embedding = text, img
            clip_text_embedding = text_clip
        else:
            text_embedding = torch.cat((text_embedding, text), 0)
            clip_text_embedding = torch.cat((clip_text_embedding, text_clip), 0)
            img_embedding  = torch.cat((img_embedding,  img),  0)

    print("loading success")
    text_embedding = text_embedding.to('cpu')
    clip_text_embedding = clip_text_embedding.to('cpu')
    img_embedding  = img_embedding.to('cpu')

    # ── 5. 相似度矩阵（GPU 上算，省一次 host→device） ─────
    struct_text_sim = text_embedding @ img_embedding.T
    if USE_KNOWLEDGE and USE_RESIDUAL_SCORING:
        clip_text_sim = clip_text_embedding @ img_embedding.T
        scale = getattr(args, "structure_residual_scale", 0.1)
        text_sim = clip_text_sim + scale * (struct_text_sim - clip_text_sim)
    else:
        text_sim = struct_text_sim
    img_sim = text_sim.T

    # ── 6. Image-to-Text 评估（按 unique 图分母）─────────
    TextRank1, TextRank5, TextRank10 = 0, 0, 0
    ImageRank1, ImageRank5, ImageRank10 = 0, 0, 0

    img_set = set()
    img_unq = []
    img_mask = []   # 1 表示该行是 unique 图的 embedding
    for i in tqdm(range(num)):
        cur_id = idx2id[i]
        if cur_id in img_set:
            img_mask.append(0)
            continue
        img_set.add(cur_id)
        img_unq.append(i)
        img_mask.append(1)

        # 文本→图 排序：img_sim[i] = (i_th_text 与所有图的相似度)
        _, sorted_idx = torch.sort(img_sim[i], descending=True)
        sorted_idx = sorted_idx.numpy().tolist()

        flag1, flag5, flag10 = 1, 1, 1
        for j in range(10):
            if j < 1 and flag1 and idx2id[sorted_idx[j]] == cur_id:
                ImageRank1  += 1; flag1  = 0
            if j < 5 and flag5 and idx2id[sorted_idx[j]] == cur_id:
                ImageRank5  += 1; flag5  = 0
            if j < 10 and flag10 and idx2id[sorted_idx[j]] == cur_id:
                ImageRank10 += 1; flag10 = 0

    img_num = len(img_set)

    # ── 7. Text-to-Image 评估（按 caption 总数分母）──────
    img_mask_tensor = torch.tensor(img_mask)
    for i in tqdm(range(len(idx2id))):
        cur_id = idx2id[i]
        # 只在 unique 图里挑：把非 unique 图的分数 mask 成 -inf
        sorted_score, sorted_idx = torch.sort(
            text_sim[i].mul(img_mask_tensor), descending=True
        )
        sorted_idx = sorted_idx.numpy().tolist()

        flag1, flag5, flag10 = 1, 1, 1
        for j in range(10):
            if j < 1 and flag1 and idx2id[sorted_idx[j]] == cur_id:
                TextRank1  += 1; flag1  = 0
            if j < 5 and flag5 and idx2id[sorted_idx[j]] == cur_id:
                TextRank5  += 1; flag5  = 0
            if flag10 and idx2id[sorted_idx[j]] == cur_id:   # 原代码逻辑：j 不限
                TextRank10 += 1; flag10 = 0

    end = time.time()
    print("Consuming {:.2f} seconds".format(end - start))
    print("TextRank1:{:.4f}, TextRank5:{:.4f}, TextRank10:{:.4f}".format(
        TextRank1 / num, TextRank5 / num, TextRank10 / num))
    print("ImageRank1:{:.4f}, ImageRank5:{:.4f}, ImageRank10:{:.4f}".format(
        ImageRank1 / img_num, ImageRank5 / img_num, ImageRank10 / img_num))

    return [TextRank1 / num, TextRank5 / num, TextRank10 / num,
            ImageRank1 / img_num, ImageRank5 / img_num, ImageRank10 / img_num]


def eval_coco_rank1(clip_model, myTransformer, dataloader, idx2id, args):
    num = len(idx2id)
    clip_model.eval()
    start = time.time()

    text_embeddings, img_embeddings = [], []
    print('loading data')

    for batch in dataloader:
        img, text, head_inputs, relation_inputs, tail_inputs, token_type_ids, attention_mask = batch
        img = img.cuda()
        text = text.squeeze(1).cuda()

        with torch.no_grad():
            # COCO retrieval is intentionally pure CLIP here. The structure module
            # is evaluated only in VG top-k reranking, so it cannot contaminate the
            # general retrieval score.
            text_clip = clip_model.encode_text(text)
            text_clip = F.normalize(text_clip, dim=1)
            img_feat = clip_model.encode_image(img)
            img_feat = F.normalize(img_feat, dim=1)

        text_embeddings.append(text_clip.cpu())
        img_embeddings.append(img_feat.cpu())

    print("loading success")
    text_embedding = torch.cat(text_embeddings, dim=0)
    img_embedding = torch.cat(img_embeddings, dim=0)

    text_sim = text_embedding @ img_embedding.T
    img_sim = text_sim.T

    TextRank1, ImageRank1 = 0, 0
    img_set = set()
    img_mask = []
    for i in tqdm(range(num)):
        cur_id = idx2id[i]
        if cur_id in img_set:
            img_mask.append(0)
            continue
        img_set.add(cur_id)
        img_mask.append(1)

        top_idx = torch.argmax(img_sim[i]).item()
        if idx2id[top_idx] == cur_id:
            ImageRank1 += 1

    img_num = len(img_set)
    img_mask_tensor = torch.tensor(img_mask, dtype=torch.bool)
    for i in tqdm(range(num)):
        cur_id = idx2id[i]
        masked_scores = text_sim[i].masked_fill(~img_mask_tensor, -float("inf"))
        top_idx = torch.argmax(masked_scores).item()
        if idx2id[top_idx] == cur_id:
            TextRank1 += 1

    end = time.time()
    text_r1 = TextRank1 / num
    image_r1 = ImageRank1 / img_num
    print("Consuming {:.2f} seconds".format(end - start))
    print("TextRank1:{:.4f}".format(text_r1))
    print("ImageRank1:{:.4f}".format(image_r1))

    return text_r1, image_r1


class VGTopKRerankDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        with open(data_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        for item in self.dataset:
            image_path = item["image_path"]
            if not os.path.isabs(image_path):
                image_path = os.path.join(VG_IMAGE_DIR, image_path)
            item["image_path"] = image_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        image = Image.open(item["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "image_id": item["image_id"],
            "task": item.get("task", "structural"),
            "candidates": item["candidates"],
        }


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


def _encode_candidate_triples(candidates, tokenizer, padding_num=6, length=5, unordered_sro=False):
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


@torch.no_grad()
def test_vg_topk_rerank(clip_model, myTransformer, data_path, args, task_name="structural"):
    if not os.path.exists(data_path):
        print(f"skip_{task_name}_topk_rerank missing file: {data_path}")
        return {}

    clip_model.eval()
    if USE_KNOWLEDGE:
        myTransformer.eval()
    if STRUCTURE_RESIDUAL_HEAD is not None:
        STRUCTURE_RESIDUAL_HEAD.eval()
    if LOCAL_STRUCTURE_MATCHER is not None:
        LOCAL_STRUCTURE_MATCHER.eval()

    dataset = VGTopKRerankDataset(data_path, transform=image_transform(is_train=False))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda batch: batch[0])
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") if USE_KNOWLEDGE else None
    topk = int(getattr(args, "topk_rerank_k", 10))

    rerank_top1_true = 0
    pair_correct = 0
    pair_total = 0
    rerank_mrr = 0.0
    total = 0

    for item in tqdm(loader, total=len(loader)):
        image = item["image"].unsqueeze(0).cuda()
        candidates = item["candidates"]
        labels = torch.tensor([candidate["label"] for candidate in candidates], device="cuda")
        captions = [candidate["caption"] for candidate in candidates]

        image_features = clip_model.encode_image(image)
        image_features = F.normalize(image_features.float(), dim=-1)
        text_tokens = torch.cat([clip.tokenize(caption, truncate=True) for caption in captions]).cuda()
        text_features = clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features.float(), dim=-1)

        clip_scores = (image_features * text_features).sum(dim=-1)
        k = min(topk, clip_scores.numel())
        top_scores, top_idx = torch.topk(clip_scores, k=k, largest=True)

        if USE_KNOWLEDGE and STRUCTURE_RESIDUAL_HEAD is not None:
            top_candidates = [candidates[idx] for idx in top_idx.detach().cpu().tolist()]
            head_inputs, relation_inputs, tail_inputs, attention_mask = _encode_candidate_triples(
                top_candidates, tokenizer, unordered_sro=getattr(args, "unordered_sro", False)
            )
            attention_mask = attention_mask.cuda()
            knowledge_features = myTransformer(head_inputs, relation_inputs, tail_inputs, 0, attention_mask)
            knowledge_features = F.normalize(knowledge_features.float(), dim=-1)
            image_top = image_features.expand(k, -1)
            text_top = text_features[top_idx]
            clip_confidence = (top_scores.max().detach() - top_scores).clamp(min=0.0)
            struct_scores, _, _, gates = STRUCTURE_RESIDUAL_HEAD(
                image_top, text_top, knowledge_features, top_scores,
                clip_confidence=clip_confidence
            )
            if LOCAL_STRUCTURE_MATCHER is not None:
                local_scores = LOCAL_STRUCTURE_MATCHER.local_scores(image, top_candidates)
                struct_scores = struct_scores + LOCAL_STRUCTURE_MATCHER.scale() * gates * local_scores
        else:
            struct_scores = top_scores

        rerank_order_in_topk = torch.argsort(struct_scores, descending=True)
        rerank_top_idx = top_idx[rerank_order_in_topk[0]]
        rerank_hit = labels[rerank_top_idx].item() == 1
        rerank_top1_true += int(rerank_hit)

        rerank_labels = labels[top_idx[rerank_order_in_topk]]
        rerank_true_positions = torch.nonzero(rerank_labels == 1, as_tuple=False)
        if rerank_true_positions.numel() > 0:
            rerank_mrr += 1.0 / float(rerank_true_positions[0].item() + 1)

        struct_score_by_pair = defaultdict(dict)
        for rank_pos, cand_idx in enumerate(top_idx.detach().cpu().tolist()):
            candidate = candidates[cand_idx]
            struct_score_by_pair[candidate["pair_id"]][candidate["source"]] = struct_scores[rank_pos].item()
        for pair_scores in struct_score_by_pair.values():
            if "true" in pair_scores and "false" in pair_scores:
                pair_total += 1
                pair_correct += int(pair_scores["true"] > pair_scores["false"])

        total += 1

    metrics = {
        f"{task_name}_rerank_top1_true": rerank_top1_true / max(total, 1),
        f"{task_name}_rerank_mrr_at_{topk}": rerank_mrr / max(total, 1),
        f"{task_name}_pairwise_in_topk": pair_correct / max(pair_total, 1),
    }
    print(f"{task_name}_topk_rerank", metrics)
    return metrics


def get_retrieval_scores_batched(clip_model, myTransformer, joint_loader, relation, args):
    """
    跨 batch 计算 (image, caption) 相似度矩阵，返回 (N, K, L) 的分数张量。
    当 USE_KNOWLEDGE=False 时完全忽略 myTransformer，KG 部分输出 0 向量。
    """
    clip_model.eval()
    # myTransformer 只在 USE_KNOWLEDGE=True 时才被实际使用
    if USE_KNOWLEDGE:
        myTransformer.eval()
    if STRUCTURE_RESIDUAL_HEAD is not None:
        STRUCTURE_RESIDUAL_HEAD.eval()

    scores = []
    for batch in tqdm(joint_loader):
        with torch.no_grad():
            # ── 1. 图像侧 ──────────────────────────────────────────
            image_options = []
            image_tensors = []
            for i_option in batch["image_options"]:
                image_tensor = clip_model.encode_image(i_option.cuda())
                image_tensor = F.normalize(image_tensor.float(), dim=1)
                image_tensors.append(image_tensor)
                image_options.append(np.expand_dims(image_tensor.detach().cpu().numpy(), axis=1))  # (B, 1, D)

            # ── 2. 文本侧（含 KG 融合） ────────────────────────────
            caption_options = []
            clip_caption_options = []
            clip_caption_tensors = []
            knowledge_tensors = []
            for index, c_option in enumerate(batch["caption_options"]):
                # 2.1 CLIP 文本向量
                caption_tokenized  = torch.cat([clip.tokenize(c) for c in c_option])
                caption_tensor = clip_model.encode_text(caption_tokenized.cuda())
                caption_tensor = F.normalize(caption_tensor.float(), dim=1)
                clip_caption_tensor = caption_tensor

                # 2.2 KG 注入
                if USE_KNOWLEDGE:
                    # ── 旧逻辑：调用 triple-transformer ──
                    if index == 1:
                        head_inputs     = batch["head_inputs"]
                        relation_inputs  = batch["relation_inputs"]
                        tail_inputs      = batch["tail_inputs"]
                        attention_mask   = batch["attention_mask"].cuda()
                    else:  # index == 0
                        head_inputs     = batch["reversed_head_inputs"]
                        relation_inputs = batch["reversed_relation_inputs"]
                        tail_inputs     = batch["reversed_tail_inputs"]
                        attention_mask  = batch["reversed_attention_mask"].cuda()
                    if getattr(args, "unordered_sro", False):
                        head_inputs, relation_inputs, tail_inputs = make_unordered_sro_inputs(
                            head_inputs, relation_inputs, tail_inputs
                        )

                    knowledge_emb = myTransformer(
                        head_inputs, relation_inputs, tail_inputs, 0, attention_mask
                    )
                    knowledge_emb = F.normalize(knowledge_emb, dim=1)            # (B, D)
                    caption_tensor = F.normalize(
                        clip_caption_tensor + knowledge_emb * args.knowledge_weight, dim=1
                    )
                else:
                    # ── 新逻辑：KG 关掉时不做任何修改 ──
                    #   caption_embeddings 直接来自 CLIP，L2 归一化已在上方完成
                    knowledge_emb = torch.zeros_like(caption_tensor)

                clip_caption_tensors.append(clip_caption_tensor)
                knowledge_tensors.append(knowledge_emb)
                clip_caption_options.append(np.expand_dims(clip_caption_tensor.detach().cpu().numpy(), axis=1))
                caption_options.append(np.expand_dims(caption_tensor.detach().cpu().numpy(), axis=1))  # (B, 1, D)

        # ── 3. 相似度矩阵 ─────────────────────────────────────
        with torch.no_grad():
            if USE_KNOWLEDGE and USE_RESIDUAL_SCORING and STRUCTURE_RESIDUAL_HEAD is not None:
                image_scores = []
                for image_tensor in image_tensors:
                    clip_score_options = [
                        (image_tensor * clip_caption_tensor).sum(dim=-1)
                        for clip_caption_tensor in clip_caption_tensors
                    ]
                    if len(clip_score_options) == 2:
                        clip_confidence = torch.abs(
                            clip_score_options[1].detach() - clip_score_options[0].detach()
                        )
                    else:
                        clip_confidence = torch.zeros_like(clip_score_options[0])
                    caption_scores = []
                    for clip_caption_tensor, knowledge_tensor, clip_score in zip(
                        clip_caption_tensors, knowledge_tensors, clip_score_options
                    ):
                        score, _, _, _ = STRUCTURE_RESIDUAL_HEAD(
                            image_tensor, clip_caption_tensor, knowledge_tensor, clip_score,
                            clip_confidence=clip_confidence
                        )
                        caption_scores.append(score.detach().cpu().numpy())
                    image_scores.append(np.stack(caption_scores, axis=1))
                batch_scores = np.stack(image_scores, axis=1)
            else:
                image_options   = np.concatenate(image_options,   axis=1)   # (B, K, D)
                clip_caption_options = np.concatenate(clip_caption_options, axis=1)
                caption_options = np.concatenate(caption_options, axis=1)   # (B, L, D)
                struct_scores = np.einsum("nkd,nld->nkl", image_options, caption_options)
                if USE_KNOWLEDGE and USE_RESIDUAL_SCORING:
                    clip_scores = np.einsum("nkd,nld->nkl", image_options, clip_caption_options)
                    scale = getattr(args, "structure_residual_scale", 0.1)
                    batch_scores = clip_scores + scale * (struct_scores - clip_scores)
                else:
                    batch_scores = struct_scores
        scores.append(batch_scores)

    all_scores = np.concatenate(scores, axis=0)   # (N, K, L)
    return all_scores

drop_relations = ['adjusting',
                  'attached to',
                  'between',
                  'bigger than',
                  'biting',
                  'boarding',
                  'brushing',
                  'chewing',
                  'cleaning',
                  'climbing',
                  'close to',
                  'coming from',
                  'coming out of',
                  'contain',
                  'crossing',
                  'dragging',
                  'draped over',
                  'drinking',
                  'drinking from',
                  'driving',
                  'driving down',
                  'driving on',
                  'eating from',
                  'eating in',
                  'enclosing',
                  'exiting',
                  'facing',
                  'filled with',
                  'floating in',
                  'floating on',
                  'flying',
                  'flying above',
                  'flying in',
                  'flying over',
                  'flying through',
                  'full of',
                  'going down',
                  'going into',
                  'going through',
                  'grazing in',
                  'growing in',
                  'growing on',
                  'guiding',
                  'hanging from',
                  'hanging in',
                  'hanging off',
                  'hanging over',
                  'higher than',
                  'holding onto',
                  'hugging',
                  'in between',
                  'jumping off',
                  'jumping on',
                  'jumping over',
                  'kept in',
                  'larger than',
                  'leading',
                  'leaning over',
                  'leaving',
                  'licking',
                  'longer than',
                  'looking in',
                  'looking into',
                  'looking out',
                  'looking over',
                  'looking through',
                  'lying next to',
                  'lying on top of',
                  'making',
                  'mixed with',
                  'mounted on',
                  'moving',
                  'on the back of',
                  'on the edge of',
                  'on the front of',
                  'on the other side of',
                  'opening',
                  'painted on',
                  'parked at',
                  'parked beside',
                  'parked by',
                  'parked in',
                  'parked in front of',
                  'parked near',
                  'parked next to',
                  'perched on',
                  'petting',
                  'piled on',
                  'playing',
                  'playing in',
                  'playing on',
                  'playing with',
                  'pouring',
                  'reaching for',
                  'reading',
                  'reflected on',
                  'riding on',
                  'running in',
                  'running on',
                  'running through',
                  'seen through',
                  'sitting behind',
                  'sitting beside',
                  'sitting by',
                  'sitting in front of',
                  'sitting near',
                  'sitting next to',
                  'sitting under',
                  'skiing down',
                  'skiing on',
                  'sleeping in',
                  'sleeping on',
                  'smiling at',
                  'sniffing',
                  'splashing',
                  'sprinkled on',
                  'stacked on',
                  'standing against',
                  'standing around',
                  'standing behind',
                  'standing beside',
                  'standing in front of',
                  'standing near',
                  'standing next to',
                  'staring at',
                  'stuck in',
                  'surrounding',
                  'swimming in',
                  'swinging',
                  'talking to',
                  'topped with',
                  'touching',
                  'traveling down',
                  'traveling on',
                  'tying',
                  'typing on',
                  'underneath',
                  'wading in',
                  'waiting for',
                  'walking across',
                  'walking by',
                  'walking down',
                  'walking next to',
                  'walking through',
                  'working in',
                  'working on',
                  'worn on',
                  'wrapped around',
                  'wrapped in',
                  "by",
                  "of",
                  "near", "next to",
                  "with",
                  "beside",
                  "on the side of",
                  "around"]


def macroacc_evaluation(scores, dataset, drop_relations=drop_relations):
    metrics = {"Accuracy": None}
    preds = np.argmax(np.squeeze(scores, axis=1), axis=-1)
    correct_mask = (preds == 1)
    metrics["Accuracy"] = np.mean(correct_mask)

    all_relations = np.array(dataset.all_relations)
    # Log the accuracy of all relations
    for relation in np.unique(all_relations):
        if relation in drop_relations:
            continue
        relation_mask = (all_relations == relation)
        if relation_mask.sum() == 0:
            continue
        metrics[f"{relation}-Acc"] = correct_mask[relation_mask].mean()

    return metrics


def macroacc_evaluation_attribute(scores, dataset):
    metrics = {"Accuracy": None}
    preds = np.argmax(np.squeeze(scores, axis=1), axis=-1)
    correct_mask = (preds == 1)
    metrics["Accuracy"] = np.mean(correct_mask)

    all_relations = np.array(dataset.all_attributes)
    # Log the accuracy of all relations
    for relation in np.unique(all_relations):
        relation_mask = (all_relations == relation)
        if relation_mask.sum() == 0:
            continue
        metrics[f"{relation}-Acc"] = correct_mask[relation_mask].mean()

    return metrics


def test_vg_relation(clip_model, myTransformer, vg_relation_dataloader, vg_relation_dataset, args):
    scores = get_retrieval_scores_batched(clip_model, myTransformer, vg_relation_dataloader, "rel", args)
    # np.save('/root/code/clip_order/checkpoints/case_study/relation/our_score.npy',scores)
    metrics = macroacc_evaluation(scores, vg_relation_dataset)
    all_accs = []
    for k, v in metrics.items():
        if "-Acc" in k:
            all_accs.append(v)
    acc_test_relation = np.mean(all_accs)
    print("acc_test_relation", acc_test_relation)
    return acc_test_relation


def test_vg_attribution(clip_model, myTransformer, vg_attribution_dataloader, vg_attribution_dataset, args):
    scores = get_retrieval_scores_batched(clip_model, myTransformer, vg_attribution_dataloader, "attribute", args)

    metrics = macroacc_evaluation_attribute(scores, vg_attribution_dataset)
    all_accs = []
    for k, v in metrics.items():
        if "-Acc" in k:
            all_accs.append(v)
    acc_test_attribution = np.mean(all_accs)
    print("acc_test_attribution", acc_test_attribution)
    return acc_test_attribution
