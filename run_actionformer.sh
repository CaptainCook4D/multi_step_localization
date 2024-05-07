#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1
# Feature Extractors = [omnivore, videomae, slowfast, 3dresnet, x3d]
## Training
python train.py configs/error_dataset_custom_feature.yaml --backbone omnivore --division_type recordings --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s_60sub --num_frames 90 --output reproduce
python train.py configs/error_dataset_custom_feature.yaml --backbone omnivore --division_type person --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s_60sub --num_frames 90 --output reproduce
python train.py configs/error_dataset_custom_feature.yaml --backbone omnivore --division_type environment --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s_60sub --num_frames 90 --output reproduce

## Evaluation
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type recordings --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s_60sub --num_frames 90 --videos_type all 2>&1 > logs/omnivore_4s_60sub_recordings_all.log
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type recordings --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s_60sub --num_frames 90 --videos_type normal 2>&1 > logs/omnivore_4s_60sub_recordings_normal.log
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type recordings --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s_60sub --num_frames 90 --videos_type error 2>&1 > logs/omnivore_4s_60sub_recordings_error.log

python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type person --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s_60sub --num_frames 90 --videos_type all 2>&1 > logs/omnivore_4s_60sub_person_all.log
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type person --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s_60sub --num_frames 90 --videos_type normal 2>&1 > logs/omnivore_4s_60sub_person_normal.log
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type person --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s_60sub --num_frames 90 --videos_type error 2>&1 > logs/omnivore_4s_60sub_person_error.log

python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type environment --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s_60sub --num_frames 90 --videos_type all 2>&1 > logs/omnivore_4s_60sub_environment_all.log
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type environment --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s_60sub --num_frames 90 --videos_type normal 2>&1 > logs/omnivore_4s_60sub_environment_normal.log
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type environment --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s_60sub --num_frames 90 --videos_type error 2>&1 > logs/omnivore_4s_60sub_environment_error.log

# Omnivore
## 3s
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type recordings --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 --videos_type all 2>&1 > logs/omnivore_3s_recordings_all.log
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type recordings --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 --videos_type normal 2>&1 > logs/omnivore_3s_recordings_normal.log
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type recordings --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 --videos_type error 2>&1 > logs/omnivore_3s_recordings_error.log

#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type person --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 --videos_type all 2>&1 > logs/omnivore_3s_person_all.log
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type person --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 --videos_type normal 2>&1 > logs/omnivore_3s_person_normal.log
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type person --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 --videos_type error 2>&1 > logs/omnivore_3s_person_error.log

#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type environment --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 --videos_type all 2>&1 > logs/omnivore_3s_environment_all.log
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type environment --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 --videos_type normal 2>&1 > logs/omnivore_3s_environment_normal.log
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type environment --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 --videos_type error 2>&1 > logs/omnivore_3s_environment_error.log

## 4s
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type recordings --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 --videos_type all 2>&1 > logs/omnivore_4s_recordings_all.log
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type recordings --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 --videos_type normal 2>&1 > logs/omnivore_4s_recordings_normal.log
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type recordings --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 --videos_type error 2>&1 > logs/omnivore_4s_recordings_error.log

#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type person --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 --videos_type all 2>&1 > logs/omnivore_4s_person_all.log
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type person --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 --videos_type normal 2>&1 > logs/omnivore_4s_person_normal.log
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type person --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 --videos_type error 2>&1 > logs/omnivore_4s_person_error.log

#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type environment --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 --videos_type all 2>&1 > logs/omnivore_4s_environment_all.log
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type environment --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 --videos_type normal 2>&1 > logs/omnivore_4s_environment_normal.log
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type environment --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 --videos_type error 2>&1 > logs/omnivore_4s_environment_error.log



