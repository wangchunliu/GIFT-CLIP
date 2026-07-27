# model.py
# PatchGraphCLIP: 用 patch-level 图像 embedding（DINOv2/CLIP patch tokens 聚类）
# 与 scene graph 三元组编码融合，解决 CLIP bag-of-words 问题（ARO benchmark）。

import math
from typing import Optional, Tuple, Union, Dict
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DINOv2Cluster(nn.Module):
    def __init__(self, in_dim: int, slot_dim: int, num_slots: int = 8):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.proj = nn.Linear(in_dim, slot_dim)
        self.slot_keys = nn.Parameter(torch.randn(num_slots, slot_dim))
        nn.init.xavier_uniform_(self.slot_keys.unsqueeze(0))

    def forward(self, patch_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.proj(patch_tokens)
        queries = F.normalize(self.slot_keys, dim=-1)
        keys = F.normalize(x, dim=-1)
        attn = torch.einsum('kd,bpd->bkp', queries, keys)
        attn = F.softmax(attn / math.sqrt(self.slot_dim), dim=-1)
        objects = torch.einsum('bkp,bpd->bkd', attn, x)
        objects = F.normalize(objects, dim=-1)
        slot_normed = F.normalize(self.slot_keys, dim=-1)
        gram = slot_normed @ slot_normed.T
        K = self.num_slots
        mask = torch.triu(torch.ones(K, K, device=gram.device, dtype=torch.bool), diagonal=1)
        diversity_loss = gram[mask].mean()
        return objects, diversity_loss


class SceneGraphEncoder(nn.Module):
    def __init__(self, vocab_size: int, word_dim: int = 64, out_dim: int = 512,
                 nhead: int = 4, num_layers: int = 2, max_triples: int = 16,
                 pad_token_id: int = 0, max_word_len: int = 5):
        super().__init__()
        self.word_dim = word_dim
        self.words_per_part = 3
        self.pad_token_id = pad_token_id
        self.max_word_len = max_word_len
        self.embed = nn.Embedding(vocab_size, word_dim, padding_idx=pad_token_id)
        seq_dim = self.words_per_part * max_word_len * word_dim
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_triples * self.words_per_part * max_word_len, word_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=word_dim, nhead=nhead,
                                                     dim_feedforward=word_dim * 4,
                                                     batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.part_proj = nn.Linear(max_word_len * word_dim, out_dim)
        self.part_norm = nn.LayerNorm(out_dim)
        self.triple_proj = nn.Linear(seq_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    @staticmethod
    def _build_block_diag_mask(N: int, tokens_per_triple: int, device: torch.device) -> Tensor:
        total = N * tokens_per_triple
        mask = torch.full((total, total), float('-inf'), device=device)
        for i in range(N):
            s = i * tokens_per_triple
            e = s + tokens_per_triple
            mask[s:e, s:e] = 0.0
        return mask

    def forward(self, triples: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        return self.encode(triples, padding_mask)[0]

    def encode(self, triples: Tensor, padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        B, N, three, W = triples.shape
        assert three == self.words_per_part
        assert W <= self.max_word_len, f"triple word length {W} exceeds max_word_len {self.max_word_len}"
        device = triples.device
        x = self.embed(triples)
        x = x.reshape(B, N * self.words_per_part * W, self.word_dim)
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]
        tokens_per_triple = self.words_per_part * W
        src_mask = self._build_block_diag_mask(N, tokens_per_triple, device)
        # Do not pass triple padding into the block-diagonal self-attention here:
        # each padded triple is an isolated block, so masking all keys in that
        # block makes softmax produce NaN. Triple-level padding is applied later
        # in cross-attention, where it masks complete triple embeddings.
        x = self.transformer(x, mask=src_mask)
        x = x.reshape(B, N, tokens_per_triple, self.word_dim)
        part_embs = x.reshape(B, N, self.words_per_part, W * self.word_dim)
        part_embs = self.part_norm(self.part_proj(part_embs))
        x = x.reshape(B, N, tokens_per_triple * self.word_dim)
        triple_embs = self.norm(self.triple_proj(x))
        return triple_embs, part_embs


class CrossModalAttention(nn.Module):
    def __init__(self, dim: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim), nn.Dropout(dropout))

    def forward(self, img_objects: Tensor, text_triples: Tensor,
                key_padding_mask: Optional[Tensor] = None) -> Tensor:
        q = self.norm_q(img_objects)
        kv = self.norm_kv(text_triples)
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        out = img_objects + attn_out
        out = out + self.ffn(self.norm_out(out))
        return out


class AttentiveFusion(nn.Module):
    def __init__(self, global_dim: int, local_dim: int, out_dim: int):
        super().__init__()
        self.global_proj = nn.Linear(global_dim, out_dim)
        self.local_proj = nn.Linear(local_dim, out_dim)
        self.img_pool_q = nn.Parameter(torch.randn(1, 1, out_dim))
        self.txt_pool_q = nn.Parameter(torch.randn(1, 1, out_dim))
        nn.init.trunc_normal_(self.img_pool_q, std=0.02)
        nn.init.trunc_normal_(self.txt_pool_q, std=0.02)
        self.gate = nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.Sigmoid())
        self.out_norm = nn.LayerNorm(out_dim)

    def _attn_pool(self, query: Tensor, tokens: Tensor) -> Tensor:
        scale = math.sqrt(tokens.size(-1))
        scores = torch.bmm(query, tokens.transpose(1, 2)) / scale
        weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(weights, tokens).squeeze(1)
        return pooled

    def forward(self, v_global: Tensor, t_global: Tensor, v_local: Tensor, t_local: Tensor) -> Dict[str, Tensor]:
        v_g = self.global_proj(F.normalize(v_global, dim=-1))
        t_g = self.global_proj(F.normalize(t_global, dim=-1))
        v_l = self.local_proj(v_local)
        t_l = self.local_proj(t_local)
        B = v_g.size(0)
        img_q = self.img_pool_q.expand(B, -1, -1)
        txt_q = self.txt_pool_q.expand(B, -1, -1)
        v_l_pooled = self._attn_pool(img_q, v_l)
        t_l_pooled = self._attn_pool(txt_q, t_l)
        v_gate = self.gate(torch.cat([v_g, v_l_pooled], dim=-1))
        t_gate = self.gate(torch.cat([t_g, t_l_pooled], dim=-1))
        v_fused = self.out_norm(v_gate * v_g + (1 - v_gate) * v_l_pooled)
        t_fused = self.out_norm(t_gate * t_g + (1 - t_gate) * t_l_pooled)
        gate_mean = (v_gate.mean() + t_gate.mean()) / 2
        return {
            'v_fused': v_fused,
            't_fused': t_fused,
            'v_local_enh': v_l,
            't_local_enh': t_l,
            'gate_mean': gate_mean,
        }


class PatchGraphCLIP(nn.Module):
    def __init__(self, clip_model, clip_dim: int = 512, slot_dim: int = 512, num_slots: int = 8,
                 sg_word_dim: int = 64, sg_vocab_size: int = 10000, sg_out_dim: int = 512,
                 sg_max_triples: int = 16, sg_pad_token_id: int = 0, fusion_out_dim: int = 512,
                 cross_nhead: int = 8, diversity_loss_weight: float = 0.1,
                 sg_max_word_len: int = 5):
        super().__init__()
        self.clip = clip_model
        self.train_clip_lora = False
        self.diversity_loss_weight = diversity_loss_weight
        for p in self.clip.parameters():
            p.requires_grad_(False)
        patch_in_dim = self._get_patch_dim()
        self.dino_cluster = DINOv2Cluster(in_dim=patch_in_dim, slot_dim=slot_dim, num_slots=num_slots)
        self.sg_encoder = SceneGraphEncoder(vocab_size=sg_vocab_size, word_dim=sg_word_dim, out_dim=sg_out_dim,
                                             max_triples=sg_max_triples, pad_token_id=sg_pad_token_id,
                                             max_word_len=sg_max_word_len)
        cross_dim = max(slot_dim, sg_out_dim)
        self.cross_attn = CrossModalAttention(dim=cross_dim, nhead=cross_nhead)
        self.slot_to_cross = nn.Linear(slot_dim, cross_dim) if slot_dim != cross_dim else nn.Identity()
        self.triple_to_cross = nn.Linear(sg_out_dim, cross_dim) if sg_out_dim != cross_dim else nn.Identity()
        self.part_to_cross = nn.Linear(sg_out_dim, cross_dim) if sg_out_dim != cross_dim else nn.Identity()
        self.relation_from_objects = nn.Sequential(
            nn.Linear(cross_dim * 4, cross_dim),
            nn.GELU(),
            nn.LayerNorm(cross_dim),
            nn.Linear(cross_dim, cross_dim),
        )
        self.fusion = AttentiveFusion(global_dim=clip_dim, local_dim=cross_dim, out_dim=fusion_out_dim)
        self.residual_score_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.residual_score_max = 0.5
        self.graph_score_logit = nn.Parameter(torch.tensor(math.log(0.2 / 0.8)))
        self.graph_score_max = 0.5
        self.align_temperature = nn.Parameter(torch.tensor(0.07))

    def _get_patch_dim(self) -> int:
        visual = self.clip.visual
        if hasattr(visual, 'conv1'):
            return visual.conv1.out_channels
        if hasattr(visual, 'patch_embed'):
            return visual.patch_embed.proj.out_channels
        if hasattr(visual, 'positional_embedding'):
            return visual.positional_embedding.shape[-1]
        raise AttributeError("Cannot infer patch dim from clip.visual")

    @torch.no_grad()
    def _extract_patches(self, images: Tensor) -> Tensor:
        visual = self.clip.visual
        x = visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.unsqueeze(0)
        x = visual.ln_pre(x)
        x = visual.transformer(x)
        x = visual.ln_post(x)
        patch_tokens = x[:, 1:, :]
        return patch_tokens

    def _split_sg(self, sg_input: Union[Dict, Tensor], ref_device: torch.device) -> Tuple[Tensor, Optional[Tensor]]:
        if isinstance(sg_input, Tensor):
            triple_ids = sg_input.to(ref_device)
            padding_mask = None
        else:
            triple_ids = sg_input['input_ids'].to(ref_device)
            padding_mask = sg_input.get('padding_mask', None)
            if padding_mask is not None:
                padding_mask = padding_mask.to(ref_device)
                empty_rows = padding_mask.all(dim=1)
                if empty_rows.any():
                    padding_mask = padding_mask.clone()
                    padding_mask[empty_rows, 0] = False
        return triple_ids, padding_mask

    def residual_scale(self) -> Tensor:
        return torch.sigmoid(self.residual_score_logit) * self.residual_score_max

    def graph_scale(self) -> Tensor:
        return torch.sigmoid(self.graph_score_logit) * self.graph_score_max

    def _valid_triple_denominator(self, padding_mask: Optional[Tensor], score: Tensor) -> Tensor:
        if padding_mask is None:
            return torch.full((score.size(0),), score.size(1), device=score.device, dtype=score.dtype)
        return (~padding_mask).float().sum(dim=1).clamp_min(1.0)

    def _graph_alignment_score(self, objects_c: Tensor, part_embs: Tensor,
                               padding_mask: Optional[Tensor] = None) -> Tensor:
        parts_c = self.part_to_cross(part_embs)
        objects_n = F.normalize(objects_c.float(), dim=-1)
        parts_n = F.normalize(parts_c.float(), dim=-1)
        head_text = parts_n[:, :, 0, :]
        rel_text = parts_n[:, :, 1, :]
        tail_text = parts_n[:, :, 2, :]

        temperature = self.align_temperature.float().clamp(0.02, 0.5)
        head_logits = torch.einsum('bkd,bnd->bnk', objects_n, head_text) / temperature
        tail_logits = torch.einsum('bkd,bnd->bnk', objects_n, tail_text) / temperature
        head_attn = F.softmax(head_logits, dim=-1)
        tail_attn = F.softmax(tail_logits, dim=-1)
        head_obj = torch.einsum('bnk,bkd->bnd', head_attn, objects_n)
        tail_obj = torch.einsum('bnk,bkd->bnd', tail_attn, objects_n)

        relation_input = torch.cat(
            [head_obj, tail_obj, head_obj * tail_obj, (head_obj - tail_obj).abs()],
            dim=-1,
        )
        rel_img = F.normalize(self.relation_from_objects(relation_input), dim=-1)

        head_score = (head_obj * head_text).sum(dim=-1)
        tail_score = (tail_obj * tail_text).sum(dim=-1)
        rel_score = (rel_img * rel_text).sum(dim=-1)
        triple_score = 0.25 * head_score + 0.25 * tail_score + 0.50 * rel_score

        if padding_mask is not None:
            triple_score = triple_score.masked_fill(padding_mask, 0.0)
        denom = self._valid_triple_denominator(padding_mask, triple_score)
        return triple_score.sum(dim=1) / denom

    def _residual_scores(self, v_global: Tensor, t_global: Tensor,
                         v_fused: Tensor, t_fused: Tensor,
                         graph_score: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        clip_score = (
            F.normalize(v_global.float(), dim=-1) *
            F.normalize(t_global.float(), dim=-1)
        ).sum(dim=-1)
        fused_score = (
            F.normalize(v_fused.float(), dim=-1) *
            F.normalize(t_fused.float(), dim=-1)
        ).sum(dim=-1)
        score = clip_score + self.residual_scale() * (fused_score - clip_score)
        if graph_score is not None:
            score = score + self.graph_scale() * graph_score.float()
        return score, clip_score, fused_score

    def forward(self, images: Tensor, texts, sg_input) -> Dict[str, Tensor]:
        device = images.device
        clip_context = nullcontext() if self.train_clip_lora else torch.no_grad()
        with clip_context:
            if isinstance(texts, dict):
                v_global = self.clip.encode_image(images)
                t_global = self.clip.encode_text(texts['input_ids'].to(device) if 'input_ids' in texts else texts)
            else:
                v_global = self.clip.encode_image(images)
                t_global = self.clip.encode_text(texts)
        patch_tokens = self._extract_patches(images)
        objects, diversity_loss = self.dino_cluster(patch_tokens)
        triple_ids, padding_mask = self._split_sg(sg_input, device)
        triple_embs, part_embs = self.sg_encoder.encode(triple_ids, padding_mask)
        objects_c = self.slot_to_cross(objects)
        triples_c = self.triple_to_cross(triple_embs)
        triples_c = F.normalize(triples_c, dim=-1)
        objects_c = F.normalize(objects_c, dim=-1)
        graph_score = self._graph_alignment_score(objects_c, part_embs, padding_mask)
        v_local_enh = self.cross_attn(objects_c, triples_c, key_padding_mask=padding_mask)
        t_local_enh = self.cross_attn(triples_c, objects_c)
        fusion_out = self.fusion(v_global=v_global, t_global=t_global, v_local=v_local_enh, t_local=t_local_enh)
        residual_score, clip_score, fused_score = self._residual_scores(
            v_global, t_global, fusion_out['v_fused'], fusion_out['t_fused'], graph_score
        )
        return {
            'v_fused': fusion_out['v_fused'],
            't_fused': fusion_out['t_fused'],
            'v_local_enh': fusion_out['v_local_enh'],
            't_local_enh': fusion_out['t_local_enh'],
            'diversity_loss': diversity_loss,
            'v_global': v_global,
            't_global': t_global,
            'residual_score': residual_score,
            'clip_score': clip_score,
            'fused_score': fused_score,
            'graph_score': graph_score,
            'residual_scale': self.residual_scale(),
            'graph_scale': self.graph_scale(),
            'gate_mean': fusion_out['gate_mean'],
        }

    @torch.no_grad()
    def score_image_text(self, images: Tensor, texts, sg_input) -> Tensor:
        device = images.device
        B_img = images.size(0)
        if isinstance(texts, dict):
            txt_ids = texts.get('input_ids', texts)
            B_text = txt_ids.size(0) if isinstance(txt_ids, Tensor) else B_img
        elif isinstance(texts, Tensor):
            B_text = texts.size(0)
        else:
            B_text = B_img
        K_opts = B_text // B_img
        assert B_text % B_img == 0
        with torch.no_grad():
            v_global = self.clip.encode_image(images)
        patch_tokens = self._extract_patches(images)
        objects, _ = self.dino_cluster(patch_tokens)
        with torch.no_grad():
            if isinstance(texts, dict):
                t_global = self.clip.encode_text(texts['input_ids'].to(device) if 'input_ids' in texts else texts)
            else:
                t_global = self.clip.encode_text(texts)
        triple_ids, padding_mask = self._split_sg(sg_input, device)
        triple_embs, part_embs = self.sg_encoder.encode(triple_ids, padding_mask)
        v_global_exp = v_global.unsqueeze(1).expand(B_img, K_opts, -1).reshape(B_text, -1)
        objects_c = self.slot_to_cross(F.normalize(objects, dim=-1))
        objects_c_exp = objects_c.unsqueeze(1).expand(B_img, K_opts, -1, -1).reshape(B_text, objects_c.size(1), objects_c.size(2))
        triples_c = self.triple_to_cross(F.normalize(triple_embs, dim=-1))
        graph_score = self._graph_alignment_score(objects_c_exp, part_embs, padding_mask)
        v_local_enh = self.cross_attn(objects_c_exp, triples_c, key_padding_mask=padding_mask)
        t_local_enh = self.cross_attn(triples_c, objects_c_exp)
        fusion_out = self.fusion(v_global=v_global_exp, t_global=t_global, v_local=v_local_enh, t_local=t_local_enh)
        scores, _, _ = self._residual_scores(
            v_global_exp, t_global, fusion_out['v_fused'], fusion_out['t_fused'], graph_score
        )
        return scores
