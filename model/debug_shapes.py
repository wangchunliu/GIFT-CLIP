"""
诊断 triple_Transformer / dataloader 的真实输入输出形状.
"""
import torch
from torch.utils.data import DataLoader
from clip import load
from utils import image_transform
from dataloader_downstream_version1 import VG_Attribution

# 加载 dataloader
ds = VG_Attribution(data_path="data/vg_attribution_aug_train.json",
                    transform=image_transform())
loader = DataLoader(ds, batch_size=4, num_workers=0, shuffle=False)

# 加载 triple_Transformer (冻结 CLIP)
clip_model, _ = load("ViT-B/32", jit=False)
clip_model = clip_model.cuda().eval()
for p in clip_model.parameters(): p.requires_grad = False

from model import triple_Transformer
myTransformer = triple_Transformer().cuda().eval()

# 取 1 个 batch
batch = next(iter(loader))
print("=" * 60)
print("Batch keys:", list(batch.keys()))
print("=" * 60)

for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"{k:30s} Tensor  {tuple(v.shape)}")
    elif isinstance(v, (list, tuple)):
        print(f"{k:30s} list/tuple len={len(v)}")
        if len(v) and isinstance(v[0], torch.Tensor):
            print(f"{' ' * 30}    -> elem shape: {tuple(v[0].shape)}")
    elif hasattr(v, 'keys'):  # BatchEncoding
        print(f"{k:30s} BatchEncoding keys={list(v.keys())}")
        for kk, vv in v.items():
            if isinstance(vv, torch.Tensor):
                print(f"  - {kk:25s} {tuple(vv.shape)}")
    else:
        print(f"{k:30s} {type(v).__name__}")

print("=" * 60)

# 把 batch 搬到 GPU
def to_cuda(x):
    if isinstance(x, torch.Tensor): return x.cuda()
    if isinstance(x, dict):         return {k_: to_cuda(v) for k_, v in x.items()}
    if isinstance(x, (list, tuple)): return [to_cuda(v) for v in x]
    return x

batch = {k: to_cuda(v) for k, v in batch.items()}

# 单独走一次 triple_Transformer, 看输出 shape
with torch.no_grad():
    knowledge_emb = myTransformer(
        batch["head_inputs"],
        batch["relation_inputs"],
        batch["tail_inputs"],
        0,
        batch["attention_mask"])

print("knowledge_emb type :", type(knowledge_emb))
if isinstance(knowledge_emb, torch.Tensor):
    print("knowledge_emb shape:", tuple(knowledge_emb.shape))
elif isinstance(knowledge_emb, (tuple, list)):
    for i, t in enumerate(knowledge_emb):
        if isinstance(t, torch.Tensor):
            print(f"  [{i}] Tensor {tuple(t.shape)}")
        else:
            print(f"  [{i}] {type(t)}")

# CLIP 文本侧, 也确认一下 shape
text = batch["caption_options"][1].squeeze(1).cuda()
with torch.no_grad():
    T_g = clip_model.encode_text(text)
print("CLIP text feat shape:", tuple(T_g.shape))
