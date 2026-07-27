#!/bin/sh

model=openai-clip:ViT-B/32

CUDA_VISIBLE_DEVICES=0 python ./model/train_version2.py \
--project 718_v2_clip_ft_bert_frozen_pairwise \
--name CocoAndVG_v2_clip_ft_bert_frozen_pairwise \
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
--structure_residual_scale 0.1 \
--structure_residual_max_scale 0.2 \
--structure_lr 1e-4 \
--kg_lr 1e-5 \
--unfreeze_clip_all \
--clip_lr 1e-7 \
--freeze_triple_bert \
--structure_margin 0.1 \
--structure_ce_weight 1.0 \
--structure_batch_ce_weight 0.0 \
--structure_delta_l2_weight 1e-2 \
--structure_delta_gap_weight 1e-2 \
--structure_gate_tau 0.1 \
--structure_gate_max 0.3 \
--structure_gate_l1_weight 1e-2 \
--structure_easy_gap 0.08 \
--structure_easy_gate_weight 0.05 \
--rerank_train_path ./data/disable_grouped_rerank_train.json \
--semantic_pair_weight 1.0 \
--cross_pair_weight 0.05 \
--no_local_structure \
--local_num_slots 8 \
--local_score_scale 0.05 \
--local_score_max_scale 0.5 \
--topk_rerank_k 10 \
--topk_attribution_path ./data/visual_genome_attribution_topk_rerank.json \
--topk_relation_path ./data/visual_genome_relation_topk_rerank.json \
--skip_topk_eval \
--eval_interval 1000 \
--device=cuda
