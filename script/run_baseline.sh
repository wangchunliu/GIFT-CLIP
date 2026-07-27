#!/bin/sh

model=openai-clip:ViT-B/32

CUDA_VISIBLE_DEVICES=0 python ./model/train.py \
--project 711_baseline \
--name CocoAndVG_train_py \
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
--topk_attribution_path ./data/visual_genome_attribution_topk_rerank.json \
--topk_relation_path ./data/visual_genome_relation_topk_rerank.json \
--eval_interval 200 \
--device=cuda
