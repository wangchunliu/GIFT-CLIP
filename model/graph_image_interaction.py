"""
graph_image_interaction.py
适配你的 triple_Transformer 输出: (B, 512), M=1
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# CrossAttentionBlock
# ----------------------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8,
                 ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        # zero-init, 训练初期完全等价原 CLIP
        nn.init.zeros_(self.ffn[-2].bias)
        nn.init.zeros_(self.ffn[-2].weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.attn.out_proj.weight)

    def forward(self, q, kv, kv_padding_mask=None, return_attn=False):
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        attn_out, attn_w = self.attn(
            qn, kvn, kvn,
            key_padding_mask=kv_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        q = q + self.dropout(attn_out)
        q = q + self.ffn(self.norm_ffn(q))
        if return_attn:
            return q, attn_w
        return q


# ----------------------------------------------------------------------
# 双向交互
# ----------------------------------------------------------------------
class GraphImageInteraction(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.img_blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model, num_heads, dropout=dropout)
             for _ in range(num_layers)])
        self.graph_blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model, num_heads, dropout=dropout)
             for _ in range(num_layers)])

    def forward(self, I, G, G_pad_mask=None, return_attn=False):
        I_star, G_star = I, G
        a_I = a_G = None

        for blk in self.img_blocks:
            if return_attn:
                I_star, a_I = blk(I_star, G,
                                  kv_padding_mask=G_pad_mask,
                                  return_attn=True)
            else:
                I_star = blk(I_star, G, kv_padding_mask=G_pad_mask)

        for blk in self.graph_blocks:
            if return_attn:
                G_star, a_G = blk(G_star, I,
                                  kv_padding_mask=None,
                                  return_attn=True)
            else:
                G_star = blk(G_star, I, kv_padding_mask=None)

        return I_star, G_star, a_I, a_G


# ----------------------------------------------------------------------
# Global -> tokens
# ----------------------------------------------------------------------
class GlobalAsTokens(nn.Module):
    def __init__(self, d_model: int, num_tokens: int = 1):
        super().__init__()
        self.K = num_tokens
        self.token = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        nn.init.trunc_normal_(self.token, std=0.02)

    def forward(self, x):
        # x: (B, d)
        B = x.size(0)
        return self.token.expand(B, -1, -1) + x.unsqueeze(1)


# ----------------------------------------------------------------------
# build_graph_tokens: 适配 BatchEncoding + (B, 512) 输出
# ----------------------------------------------------------------------
def build_graph_tokens(myTransformer,
                       head_inputs, rel_inputs, tail_inputs,
                       token_type_ids=None, attention_mask=None):
    """
    输出:
        G       : (B, M, d)        d=512, M=1 (你的 triple_Transformer 内部已聚合)
        pad_mask: (B, M)  True = padding
    """
    # 1) 整批过一次 triple_Transformer
    tti = token_type_ids if token_type_ids is not None else 0
    knowledge_emb = myTransformer(
        head_inputs, rel_inputs, tail_inputs, tti, attention_mask)

    if isinstance(knowledge_emb, (tuple, list)):
        knowledge_emb = knowledge_emb[0]

    if not isinstance(knowledge_emb, torch.Tensor):
        raise TypeError(f"triple_Transformer returned {type(knowledge_emb)}")

    # 2) 推断 batch size
    if hasattr(head_inputs, "keys"):
        B = head_inputs["input_ids"].size(0)
    elif isinstance(head_inputs, torch.Tensor):
        B = head_inputs.size(0)
    else:
        raise ValueError(f"Cannot infer batch size from {type(head_inputs)}")

    d = knowledge_emb.size(-1)

    # 3) reshape 到 (B, M, d)
    if knowledge_emb.dim() == 2:
        # (B, d) -> (B, 1, d)    ← 你的情况
        G = knowledge_emb.unsqueeze(1)
        M = 1
    elif knowledge_emb.dim() == 3:
        G = knowledge_emb
        M = G.size(1)
    elif knowledge_emb.dim() == 1:
        G = knowledge_emb.unsqueeze(0).unsqueeze(-1)
        M = 1
    else:
        flat = knowledge_emb.reshape(-1, d)
        M = flat.size(0) // B
        G = flat.reshape(B, M, d)

    device = G.device
    pad_mask = torch.zeros(B, M, dtype=torch.bool, device=device)  # False=valid
    return G, pad_mask


# ----------------------------------------------------------------------
# PositiveScale: 非负 alpha/beta
# ----------------------------------------------------------------------

class PositiveScale(nn.Module):
    """
    Non-negative bounded scaling factor.
    输出 α ∈ (0, max_val), 通过 sigmoid 平滑.
    用 log_alpha 命名以兼容外部代码 (eval.py) 直接修改 .data.
    """
    def __init__(self, init: float = 0.05, max_val: float = 0.5):
        super().__init__()
        self.max_val = max_val
        # sigmoid 反函数: logit(p) where p = init / max_val
        p = init / max_val
        p = max(1e-4, min(1 - 1e-4, p))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(p / (1 - p))))

    def forward(self):
        return torch.sigmoid(self.log_alpha) * self.max_val
