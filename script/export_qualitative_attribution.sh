#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python ./model/export_qualitative_scores_v2.py \
--checkpoint ./checkpoints/train_version2_pairwise_plugin.pt \
--task attribution \
--data_path ./data/visual_genome_attribution_aug.json \
--output ./outputs/qualitative_scores_attribution.csv \
--num_examples 30
