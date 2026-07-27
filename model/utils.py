import argparse
import random
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from PIL import Image
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm
try:
    import sng_parser
except ImportError:
    sng_parser = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/coco_dataset_train.txt', type=str)
    parser.add_argument('--test_path', default='data/coco_dataset_test.txt', type=str)
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument("--model-name", default="openai-clip:ViT-B/32", type=str)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument('--manualSeed', type=int, default=120)
    parser.add_argument('--batch_size', default='128', type=int)
    parser.add_argument('--lr', default='2e-7', type=float)
    parser.add_argument('--epoch', default='10', type=int)
    parser.add_argument('--weight_decay', default='0.1', type=float)
    parser.add_argument('--knowledge_weight', default='0.01', type=float)
    parser.add_argument('--transformer_layer_num', default='4', type=int)
    parser.add_argument('--neg_loss_weight', default='1', type=float)
    parser.add_argument('--topk_rerank_k', default=10, type=int)
    parser.add_argument('--topk_relation_path', default='data/visual_genome_relation_topk_rerank.json', type=str)
    parser.add_argument('--topk_attribution_path', default='data/visual_genome_attribution_topk_rerank.json', type=str)
    parser.add_argument('--skip_topk_eval', action='store_true', default=False)
    parser.add_argument('--eval_interval', default=200, type=int)
    parser.add_argument('--save_path', default='', type=str)
    parser.add_argument('--structure_residual_scale', default=0.1, type=float)
    parser.add_argument('--structure_residual_max_scale', default=0.2, type=float)
    parser.add_argument('--structure_lr', default=1e-4, type=float)
    parser.add_argument('--kg_lr', default=1e-5, type=float)
    parser.add_argument('--freeze_triple_bert', action='store_true', default=False)
    parser.add_argument('--structure_loss_weight', default=1.0, type=float)
    parser.add_argument('--structure_ce_weight', default=1.0, type=float)
    parser.add_argument('--structure_margin', default=0.1, type=float)
    parser.add_argument('--structure_batch_ce_weight', default=0.0, type=float)
    parser.add_argument('--structure_delta_l2_weight', default=1e-2, type=float)
    parser.add_argument('--structure_delta_gap_weight', default=1e-2, type=float)
    parser.add_argument('--structure_gate_tau', default=0.05, type=float)
    parser.add_argument('--structure_gate_max', default=0.3, type=float)
    parser.add_argument('--direct_score_prediction', action='store_true', default=False)
    parser.add_argument('--uniform_residual_weighting', action='store_true', default=False)
    parser.add_argument('--unordered_sro', action='store_true', default=False)
    parser.add_argument('--structure_gate_l1_weight', default=1e-2, type=float)
    parser.add_argument('--structure_easy_gap', default=0.05, type=float)
    parser.add_argument('--structure_easy_gate_weight', default=0.05, type=float)
    parser.add_argument('--rerank_train_path', default='data/train_coco_topk_rerank.json', type=str)
    parser.add_argument('--rerank_train_topk', default=16, type=int)
    parser.add_argument('--rerank_loss_weight', default=1.0, type=float)
    parser.add_argument('--rerank_pair_weight', default=1.0, type=float)
    parser.add_argument('--semantic_pair_weight', default=1.0, type=float)
    parser.add_argument('--cross_pair_weight', default=0.05, type=float)
    parser.add_argument('--rerank_include_best_true', action='store_true', default=True)
    parser.add_argument('--use_local_structure', action='store_true', default=True)
    parser.add_argument('--no_local_structure', dest='use_local_structure', action='store_false')
    parser.add_argument('--local_num_slots', default=8, type=int)
    parser.add_argument('--local_score_scale', default=0.05, type=float)
    parser.add_argument('--local_score_max_scale', default=0.5, type=float)
    parser.add_argument('--use_clip_lora', action='store_true', default=False)
    parser.add_argument('--clip_lora_rank', default=4, type=int)
    parser.add_argument('--clip_lora_alpha', default=8.0, type=float)
    parser.add_argument('--clip_lora_lr', default=1e-6, type=float)
    parser.add_argument('--clip_lora_text', action='store_true', default=True)
    parser.add_argument('--no_clip_lora_text', dest='clip_lora_text', action='store_false')
    parser.add_argument('--clip_lora_visual', action='store_true', default=False)
    parser.add_argument('--no_clip_lora_visual', dest='clip_lora_visual', action='store_false')
    parser.add_argument('--unfreeze_clip_projection', action='store_true', default=False)
    parser.add_argument('--clip_projection_lr', default=1e-6, type=float)
    parser.add_argument('--unfreeze_clip_all', action='store_true', default=False)
    parser.add_argument('--clip_lr', default=1e-7, type=float)
    parser.add_argument('--weight_g2g', default=0.0, type=float)
    parser.add_argument('--weight_fused', default=0.0, type=float)
    parser.add_argument('--weight_l2l', default=0.0, type=float)
    parser.add_argument('--weight_diversity', default=0.0, type=float)
    parser.add_argument('--weight_hardneg', default=1.0, type=float)
    parser.add_argument('--weight_hardneg_ce', default=1.0, type=float)
    parser.add_argument('--hardneg_margin', default=0.1, type=float)
    parser.add_argument('--temp', default=0.07, type=float)
    

    return parser.parse_args()


def set_manualSeed(args):
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

# image transform
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def image_transform(is_train = True, num_pix = 224):
    if is_train:
        return Compose([
            RandomResizedCrop(num_pix, scale=(0.9, 1.0), interpolation=BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        return Compose([
            Resize(num_pix, interpolation=BICUBIC),
            CenterCrop(num_pix),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


def swap_words(s, x, y):
    return y.join(part.replace(y, x) for part in s.split(x))

class scene_graph():
    def __init__(self, text):
        if sng_parser is None:
            raise ImportError("sng_parser is required only when using scene_graph.")
        self.graph = sng_parser.parse(text)
        self.text = text
    def swap_words(self, s, x, y):
        return y.join(part.replace(y, x) for part in s.split(x))
    def neg_gen(self):
        neg_text = []
        for relation in self.graph['relations']:
            subject = self.graph['entities'][relation['subject']]['span']

            object = self.graph['entities'][relation['object']]['span']
            temp = self.swap_words(self.text, subject, object)
            neg_text.append(temp.lower())
        if neg_text == []:
            print(self.text)
        return neg_text[0]

def compute_logits(image_features, text_true_features, text_gen_features, logit_scale):
    logit_img2text_true = logit_scale * image_features @ text_true_features.T
    logit_text_true2img = logit_scale * text_true_features @ image_features.T
    logit_img2text_gen = logit_scale * image_features @ text_gen_features.T
    logit_text_gen2img = logit_scale * text_gen_features @ image_features.T
    device = image_features.device
    num_logits = image_features.shape[0]
    logit_diag_true = torch.diag(logit_img2text_true)
    logit_diag_gen = torch.diag(logit_img2text_gen)
    x = logit_diag_true - logit_diag_gen
    Wino_label = torch.tensor([1] * num_logits, device=device)

    return logit_diag_true, logit_diag_gen, Wino_label

class WinoLoss(nn.Module):
    def __init__(self, margin):
        super(WinoLoss, self).__init__()
        self.margin = margin
        self.loss = nn.MarginRankingLoss(margin=margin, reduction='mean')
        self.relu = nn.ReLU()
    def forward(self, image_features, text_features, logit_scale, is_hard):


        # last
        batch_size = image_features.shape[0]
        cos_img2text = torch.matmul(image_features, text_features.T) # [bs,bs]

        pos_score = torch.diag(cos_img2text) #[bs]
        img_neg_score = torch.max(cos_img2text - 10 * torch.eye(batch_size, requires_grad=False).cuda(0), dim=-1)[0] # [bs]

        cos_text2img = cos_img2text.T #[bs,bs]
        if is_hard:
            text_neg_score = torch.max(cos_text2img - 10 * torch.eye(batch_size, requires_grad=False).cuda(0), dim=-1)[0] # [bs]
        else: 
            text_neg_score = torch.mean(cos_text2img, dim=-1)
        # text_neg_score = torch.max(cos_text2img - 10 * torch.eye(batch_size, requires_grad=False).cuda(0), dim=-1)[0] # [bs]
        margin = torch.ones_like(pos_score, requires_grad=False) * 0.2  #[bs]

        loss = self.relu(img_neg_score + margin - pos_score) + self.relu(text_neg_score + margin - pos_score)
        loss = torch.mean(loss)
        return loss

class MarginLoss(nn.Module):
    def __init__(self, margin):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.loss = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    def forward(self, image_features, text_true_features, text_gen_features, logit_scale):
        logit_img2text_true = logit_scale * image_features @ text_true_features.T
        logit_text_true2img = logit_scale * text_true_features @ image_features.T
        logit_img2text_gen = logit_scale * image_features @ text_gen_features.T
        logit_text_gen2img = logit_scale * text_gen_features @ image_features.T
        device = image_features.device
        num_logits = image_features.shape[0]
        logit_diag_true = torch.diag(logit_img2text_true)
        logit_diag_gen = torch.diag(logit_img2text_gen)

        clip_label = torch.arange(num_logits, device=device, dtype=torch.long)
        Wino_label = torch.tensor([1] * num_logits, device=device)
        total_loss = self.loss(logit_diag_true, logit_diag_gen, Wino_label)

        return total_loss

class MyLoss(nn.Module):
    def __init__(self, margin):
        super(MyLoss, self).__init__()
        self.margin = margin
        self.loss = nn.ReLU()
    def forward(self, image_feat, pos_GCN_emb, neg_GCN_emb, pos_text_feat, neg_text_feat):
        alpha = 1.0
        pos_GCN_emb = pos_GCN_emb / pos_GCN_emb.norm(dim=-1, keepdim=True)
        neg_GCN_emb = neg_GCN_emb / neg_GCN_emb.norm(dim=-1, keepdim=True)
        pos_text = pos_text_feat * 0.0 + alpha * pos_GCN_emb
        neg_text = neg_text_feat * 0.0 + alpha * neg_GCN_emb
        
        img2pos = image_feat @ pos_text.T
        img2neg = image_feat @ neg_text.T
        margin = 0.2
        pos_score = torch.diag(img2pos)
        neg_score = torch.diag(img2neg)
        loss = self.loss(neg_score - pos_score + margin)
        return loss

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self.scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_embed, text_embed):

        local_batch_size = image_embed.size(0)


        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(local_batch_size, device=image_embed.device)
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs
        image_embed_all, text_embed_all = all_gather_batch([image_embed, text_embed])

        # cosine similarity as logits
        logits_per_image = self.scale * image_embed @ text_embed_all.t()
        logits_per_text = self.scale * text_embed @ image_embed_all.t()

        loss = (F.cross_entropy(logits_per_image, self.labels) + F.cross_entropy(logits_per_text[:len(logits_per_image)], self.labels)) / 2

        # compute accuracy

        return loss


def triple2emb(head, rel, tail, clip_model):
    gamma = 1.0
    gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
    
    head_emb = clip_model.encode_text(head)
    rel_emb = clip_model.encode_text(rel)
    tail_emb = clip_model.encode_text(tail)
    # transe求emb
    score = head_emb + rel_emb - tail_emb
    # score = gamma.item() - torch.norm(score, p=1, dim=-1)
    
    return score
