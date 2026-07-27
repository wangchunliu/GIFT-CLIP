#!/bin/sh

model=openai-clip:ViT-B/32

CUDA_VISIBLE_DEVICES=0 python ./model/train_version3.py \
--project 712_v3_lora \
--name CocoAndVG_v3_lora \
--model-name=$model \
--train_path ./data/train_coco_aug_withneg_adjchange_merge.json \
--test_path ./data/visual_genome_attribution_aug.json \
--manualSeed 120 \
--batch_size 32 \
--lr 1e-6 \
--epoch 10 \
--weight_decay 0.1 \
--knowledge_weight 0.01 \
--transformer_layer_num 6 \
--neg_loss_weight 2 \
--temp 0.07 \
--weight_g2g 0.0 \
--weight_fused 0.0 \
--weight_l2l 0.0 \
--weight_diversity 0.0 \
--weight_hardneg 1.0 \
--weight_hardneg_ce 1.0 \
--hardneg_margin 0.1 \
--use_clip_lora \
--clip_lora_rank 8 \
--clip_lora_alpha 16 \
--clip_lora_lr 1e-5 \
--clip_lora_text \
--clip_lora_visual \
--device=cuda
