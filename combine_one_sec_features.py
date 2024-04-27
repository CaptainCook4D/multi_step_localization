import os

import numpy as np


def combine_features():
    backbone = 'omnivore_swinB_epic_1s'
    features_root_dir = "/data/error_dataset/features"

    one_sec_features_dir = os.path.join(features_root_dir, "omnivore")
    features_save_dir = os.path.join(features_root_dir, backbone)
    for rec_id in sorted(os.listdir(one_sec_features_dir)):
        rec_dir = os.path.join(one_sec_features_dir, rec_id)
        # if rec_dir.endswith('.npy'):
        #     continue
        # natural sorting
        rec_directories = os.listdir(rec_dir)
        time_sorted_rec_dirs = sorted(rec_directories, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        video_feat = []
        print(f"Processing {rec_id}...")
        for feat_file in time_sorted_rec_dirs:
            feat_file_path = os.path.join(rec_dir, feat_file)
            feat = np.load(feat_file_path)
            if len(feat.shape) > 1:
                if feat.shape[0] == 1:
                    feat = feat[0]

            video_feat.append(feat[:])
        video_feat = np.stack(video_feat, axis=0)
        if os.path.exists(os.path.join(features_save_dir, rec_id, 'video_features.npy')):
            os.remove(os.path.join(features_save_dir, rec_id, 'video_features.npy'))
        video_feat_path = os.path.join(features_save_dir, rec_id, 'video_features.npy')
        os.makedirs(os.path.dirname(video_feat_path), exist_ok=True)
        np.save(video_feat_path, video_feat)
        print(f"Saved {rec_id}: {video_feat_path}")


if __name__ == '__main__':
    combine_features()
