import json
import os

import numpy as np
import pandas as pd


def csv_to_json(split_type, video_category):
    steps_count = 353
    fps = 29.97
    captaincook_dataset_version = f"CaptainCookDataset-{fps}fps-{steps_count}Steps"

    # Use the annotations from the CaptainCook annotations repository submodule
    dataset_dir = "./captaincook"
    annotations_json_dir = os.path.join(dataset_dir, "annotation_json")

    data_splits_dir = os.path.join(dataset_dir, "data_splits")
    data_split_json_files = []
    for data_split_name in os.listdir(data_splits_dir):
        if split_type in data_split_name and video_category in data_split_name:
            data_split_json_files.append(data_split_name)

    assert len(data_split_json_files) == 1, f"Expected 1 {split_type} data split file, but found {len(data_split_json_files)}"

    step_idx_description_json_path = os.path.join(annotations_json_dir, "step_idx_description.json")
    step_annotations_json_path = os.path.join(annotations_json_dir, "step_annotations.json")

    video_durations_csv_path = os.path.join("./captaincook/metadata", "video_information.csv")
    af_annotations_split_type_json_path = os.path.join("captaincook_actionformer_annotations", f"{video_category}", f"{split_type}.json")

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
    step_annotations_json = json.load(open(step_annotations_json_path, 'r'))
    step_idx_description_json = json.load(open(step_idx_description_json_path, 'r'))

    video_durations = pd.read_csv(video_durations_csv_path)
    data_subset_name_map = {
        "train": "Training",
        "val": "Validation",
        "test": "Test",
    }

    captaincook_dataset = {}

    for data_split_file in data_split_json_files:
        data_split_json_file_path = os.path.join(data_splits_dir, data_split_file)
        data_split_json = json.load(open(data_split_json_file_path, 'r'))

        # Loop through the train, val, test subsets of the data split to create the final json
        for data_subset_type in data_subset_name_map.keys():
            subset_type_recording_ids = data_split_json[data_subset_type]
            for recording_id in subset_type_recording_ids:
                video_duration = video_durations[video_durations["recording_id"] == recording_id]["duration(sec)"].values[0]
                recoding_id_step_annotations_json = step_annotations_json[recording_id]["steps"]
                annotations_list = []
                for i in range(len(recoding_id_step_annotations_json)):
                    step_details = recoding_id_step_annotations_json[i]
                    step_description = step_details["description"]
                    step_id = int(step_details["step_id"])
                    start_time = float(step_details["start_time"])
                    end_time = float(step_details["end_time"])
                    has_errors = step_details["has_errors"]

                    step_id = int(step_id)  # int64 to int as json does not support int64
                    has_error = bool(has_errors)  # bool8 to bool as json does not support bool8

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

                captaincook_dataset[recording_id] = {
                    "subset": data_subset_name_map[data_subset_type],
                    "duration": video_duration,
                    "fps": fps,
                    "annotations": annotations_list,
                    "has_error": False,
                }
    json_dataset = {
        "version": captaincook_dataset_version,
        "database": captaincook_dataset,
    }
    with open(af_annotations_split_type_json_path, "w") as f:
        json.dump(json_dataset, f, indent=4)


def generate_jsons():
    split_type_list = ["recordings", "person", "environment", "recipes"]
    video_category_types = ["normal", "combined"]
    for split_type in split_type_list:
        for video_category in video_category_types:
            print(f"Processing {split_type}...")
            csv_to_json(split_type, video_category)


if __name__ == '__main__':
    generate_jsons()
