"""
把 cross-attention 权重画成热图, 适合直接放进论文.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
    return np.array(t)


def plot_attention_heatmap(attn, query_labels=None, key_labels=None,
                           title="Cross-Attention",
                           head_idx=0, save_path=None, figsize=(6, 4)):
    """
    attn: (B, h, Nq, Nk) or (h, Nq, Nk)
    """
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().cpu().numpy()
    if attn.ndim == 4:
        attn = attn[0]                            # (h, Nq, Nk)
    if attn.ndim == 3:
        attn = attn[head_idx]                     # (Nq, Nk)

    plt.figure(figsize=figsize)
    plt.imshow(attn, aspect='auto', cmap='viridis')
    plt.colorbar()
    if query_labels is not None:
        plt.yticks(range(len(query_labels)), query_labels, fontsize=8)
    if key_labels is not None:
        plt.xticks(range(len(key_labels)), key_labels, fontsize=8, rotation=45)
    plt.xlabel("Graph Nodes (Key)")
    plt.ylabel("Image Tokens (Query)")
    plt.title(f"{title} | head={head_idx}")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        return save_path
    return plt.gcf()


def plot_attention_over_image(img_tensor, attn_to_img,
                              patch_size=32, alpha=0.55,
                              head_idx=0, save_path=None):
    """
    img_tensor: (3, H, W)  (已做 normalize)
    attn_to_img: (B, h, Nq, Nk)  -- 这里用 mean over Nq 当 spatial map
    适合 "graph -> image" 的注意力, 平均到 Nk 维再 reshape.
    """
    if isinstance(attn_to_img, torch.Tensor):
        attn_to_img = attn_to_img.detach().cpu().numpy()
    attn = attn_to_img[0, head_idx].mean(0)        # (Nk,)
    # 不一定能 reshape, 这里只返回归一化后的一维向量
    norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-9)
    return norm
