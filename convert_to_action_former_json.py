import json
import os

import numpy as np
import pandas as pd
from pytorchvideo.data.encoded_video import EncodedVideo


def get_video_durations(recording_id_paths_dict):
    video_durations = {}
    for recording_id, video_path in recording_id_paths_dict.items():
        torch_video = EncodedVideo.from_path(video_path)
        video_durations[recording_id] = float(torch_video.duration)
    return video_durations


def generate_video_durations_csv(data_subsets, dataset_dir, video_splits_dir, video_durations_path):
    recording_ids = []
    for subset in data_subsets.keys():
        data_dir_path = os.path.join(video_splits_dir, subset)
        recipe_id_dirs = os.listdir(data_dir_path)
        for recipe_dir in recipe_id_dirs:
            recipe_dir_path = os.path.join(data_dir_path, recipe_dir)
            recording_ids += [v.split('.')[0] for v in os.listdir(recipe_dir_path)]
    videos_dir = os.path.join(dataset_dir, "videos")
    recording_id_paths_dict = {rec_id: os.path.join(videos_dir, f"{rec_id}_360p.mp4") for rec_id in recording_ids}
    video_durations = get_video_durations(recording_id_paths_dict)
    df = pd.DataFrame.from_dict(video_durations, orient="index", columns=["duration"])
    df.to_csv(video_durations_path, index_label="recording_id")


def csv_to_json(split_type):
    steps_count = 353
    fps = 29.97
    error_dataset_version = f"ErrorDataset-{fps}fps-{steps_count}Steps"

    dataset_dir = "/data/error_dataset"
    annotations_dir = os.path.join(dataset_dir, "annotations")
    video_splits_dir = os.path.join(dataset_dir, "video_splits", split_type, "normal_error")

    step_idx_description_path = os.path.join(annotations_dir, "step_idx_description.csv")
    step_annotations_path = os.path.join(annotations_dir, "step_annotations.csv")
    video_durations_path = os.path.join(annotations_dir, "video_durations.csv")
    json_dataset_path = os.path.join(annotations_dir, f"annotations_actionformer_{split_type}.json")

    """
    version: {}
    database: {
        recording_id: {
            subset: {}
            duration: {}
            fps: {}
            annotations: {[
                {
                  "label": "<Step Description>",
                  "segment": [
                    18.8,
                    57.3
                  ],
                  "segment(frames)": [
                    564.0,
                    1719.0
                  ],
                  "label_id": 19
                  "has_error": False
                }
            ]},
        },
        ...
    }
    """
    step_annotations = pd.read_csv(step_annotations_path)
    step_idx_description = pd.read_csv(step_idx_description_path)
    video_durations = pd.read_csv(video_durations_path)
    data_subsets = {
        "train": "Training",
        "val": "Validation",
        "test": "Test",
    }
    # generate_video_durations_csv(data_subsets, dataset_dir, video_splits_dir, video_durations_path)

    error_dataset = {}
    for data_split_name in os.listdir(video_splits_dir):
        data_split_dir = os.path.join(video_splits_dir, data_split_name)
        subset = data_subsets[data_split_name]
        for recipe_id in os.listdir(data_split_dir):
            recipe_dir = os.path.join(data_split_dir, recipe_id)
            for video_name in os.listdir(recipe_dir):
                recording_id = video_name.split('.')[0]
                video_duration = video_durations[video_durations["recording_id"] == recording_id]["duration"].values[0]
                annotations = step_annotations[step_annotations["recording_id"] == recording_id]
                annotations_list = []
                for i in range(len(annotations)):
                    step_description = annotations.iloc[i]["step_description"]
                    step_id = annotations.iloc[i]["step_id"].astype(int)
                    start_time = annotations.iloc[i]["start_time"].astype(float)
                    end_time = annotations.iloc[i]["end_time"].astype(float)
                    has_error = annotations.iloc[i]["has_error"]

                    step_id = int(step_id)  # int64 to int as json does not support int64
                    has_error = bool(has_error)  # bool8 to bool as json does not support bool8

                    annotation = {
                        "label": step_description,
                        "segment": [
                            start_time,
                            end_time
                        ],
                        "segment(frames)": [
                            np.floor(start_time * 29.97),
                            np.ceil(end_time * 29.97)
                        ],
                        "label_id": step_id,
                        "has_error": has_error
                    }
                    annotations_list.append(annotation)

                error_dataset[recording_id] = {
                    "subset": subset,
                    "duration": video_duration,
                    "fps": fps,
                    "annotations": annotations_list,
                    "has_error": False,
                }
    json_dataset = {
        "version": error_dataset_version,
        "database": error_dataset,
    }
    with open(json_dataset_path, "w") as f:
        json.dump(json_dataset, f, indent=4)
    pass


def generate_jsons():
    split_type_list = ["recordings", "person", "environment", "recipes"]
    for split_type in split_type_list:
        print(f"Processing {split_type}...")
        csv_to_json(split_type)


if __name__ == '__main__':
    generate_jsons()
