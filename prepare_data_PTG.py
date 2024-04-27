import json
import os
import os.path as osp
import shutil

import pandas as pd


def prepare_video_directory(recording_map, recording_type, split_type, base_path='/data/ptg/demo23/video_splits'):
    os.makedirs(base_path, exist_ok=True)

    base_videos_path = osp.join(base_path, split_type, recording_type)
    os.makedirs(base_videos_path, exist_ok=True)

    train_videos_path = osp.join(base_videos_path, 'train')
    os.makedirs(train_videos_path, exist_ok=True)

    val_videos_path = osp.join(base_videos_path, 'val')
    os.makedirs(val_videos_path, exist_ok=True)

    test_videos_path = osp.join(base_videos_path, 'test')
    os.makedirs(test_videos_path, exist_ok=True)

    for activity, recording_list in recording_map.items():
        print(f"Copying {activity} videos to {recording_type} directory")
        train_set, val_set, test_set = recording_list[0][0], recording_list[0][1], recording_list[0][2]
        for recording_set, recording_set_path in zip([train_set, val_set, test_set],
                                                     [train_videos_path, val_videos_path, test_videos_path]):
            os.makedirs(osp.join(recording_set_path, f"{activity}"), exist_ok=True)
            for train_recording_num in recording_set:
                recording_id = f'{activity}_{train_recording_num}'
                raw_recording_path = osp.join(raw_videos_path, f"{recording_id}_360p.mp4")
                if not osp.exists(raw_recording_path):
                    print(f"File does not exist: {raw_recording_path}")
                    continue
                train_recording_path = osp.join(recording_set_path, f"{activity}", f"{recording_id}.mp4")
                if not osp.exists(train_recording_path):
                    shutil.copy(raw_recording_path, train_recording_path)
                else:
                    print(f"File already exists: {train_recording_path}")


def prepare_data_splits_for_splits(split_type, activity_id, recording_num_list):
    total_recordings = len(recording_num_list)
    train_recordings = int(0.6 * total_recordings)
    val_recordings = int(0.2 * total_recordings)
    test_recordings = int(0.2 * total_recordings)

    train_set, val_set, test_set = set(), set(), set()

    train_set = set(recording_num_list[:train_recordings])
    val_set = set(recording_num_list[train_recordings:train_recordings + val_recordings])
    test_set = set(recording_num_list[train_recordings + val_recordings:])

    return train_set, val_set, test_set


def prepare_recording_maps_for_splits(split_type) -> (dict, dict):
    normal_recording_map = dict()
    normal_error_combined_recording_map = dict()
    for activity_id in activity_idx_list:
        if activity_id not in normal_recording_map:
            normal_recording_map[activity_id] = []
        if activity_id not in normal_error_combined_recording_map:
            normal_error_combined_recording_map[activity_id] = []

    for activity_id in activity_idx_list:
        recording_num_list = activity_recording_map[activity_id]
        recording_num_list = [int(recording_num) for recording_num in recording_num_list]
        recording_num_list = sorted(recording_num_list)

        normal_recording_num_list = []
        error_recording_num_list = []
        # ToDo: Use the all_step_annotations.csv file to get the error recordings
        # start_time or end_time is -1 for error recordings
        # has_errors == True for error recordings
        for recording_num in recording_num_list:
            rec_id = f"{activity_id}_{recording_num}"
            rec_annotations = step_annotations[step_annotations['recording_id'] == rec_id]
            if rec_annotations['has_errors'].any():
                error_recording_num_list.append(recording_num)
            elif (len(rec_annotations[rec_annotations['start_time'] == -1]) > 0
                  or len(rec_annotations[rec_annotations['end_time'] == -1]) > 0):
                error_recording_num_list.append(recording_num)
            else:
                normal_recording_num_list.append(recording_num)

        normal_train_set, normal_val_set, normal_test_set = prepare_data_splits_for_splits(split_type, activity_id,
                                                                                           normal_recording_num_list)
        error_train_set, error_val_set, error_test_set = prepare_data_splits_for_splits(split_type, activity_id,
                                                                                        error_recording_num_list)

        normal_recording_map[activity_id].append([normal_train_set, normal_val_set, normal_test_set])

        normal_error_train_set = normal_train_set | error_train_set
        normal_error_val_set = normal_val_set | error_val_set
        normal_error_test_set = normal_test_set | error_test_set

        normal_error_combined_recording_map[activity_id].append(
            [normal_error_train_set, normal_error_val_set, normal_error_test_set])

    return normal_recording_map, normal_error_combined_recording_map


def fetch_lists():
    activity_idx_list = sorted(list(activity_idx_step_idx_df["activity_idx"].unique()))
    activity_idx_list = [str(activity_idx) for activity_idx in activity_idx_list]
    activity_recording_map = dict()
    for activity_id in activity_idx_list:
        if activity_id not in activity_recording_map:
            activity_recording_map[activity_id] = []
    recording_id_list = sorted(list(step_annotations["recording_id"].unique()))
    for recording_id in recording_id_list:
        activity_id, recording_num = recording_id.split('_')
        activity_recording_map[activity_id].append(recording_num)
    return activity_idx_list, activity_recording_map


if __name__ == '__main__':
    activity_idx_step_idx_path = osp.join('./data/ptg_dataset/annotations/activity_idx_step_idx.csv')
    step_annotations_json_path = osp.join('./data/ptg_dataset/annotations/all_step_annotations.csv')
    step_idx_description_path = osp.join('./data/ptg_dataset/annotations/step_idx_description.csv')

    activity_idx_step_idx_df = pd.read_csv(activity_idx_step_idx_path)
    step_annotations = pd.read_csv(step_annotations_json_path)
    step_idx_description = pd.read_csv(step_idx_description_path)

    # Remove MugCake
    activity_idx_step_idx_df = activity_idx_step_idx_df[activity_idx_step_idx_df['activity_name'] != 'MugCake']

    dataset_dir = "/data/ptg/demo23"
    raw_videos_path = osp.join(dataset_dir, "videos")
    activity_idx_list, activity_recording_map = fetch_lists()
    split_type_list = ["recordings"]
    save_path = osp.join(dataset_dir, "video_splits")
    for split_type in split_type_list:
        norm_recording_map, norm_error_combined_recording_map = prepare_recording_maps_for_splits(split_type)
        prepare_video_directory(norm_recording_map, "normal", split_type, save_path)
        prepare_video_directory(norm_error_combined_recording_map, "normal_error", split_type, save_path)
