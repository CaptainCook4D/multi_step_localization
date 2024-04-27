import csv
import os
import re

# Define a regular expression pattern to extract the values
pattern = r"tIoU = ([0-9.]+): mAP = ([0-9.]+) \(%\) Recall@1x = ([0-9.]+) \(%\) Recall@5x = ([0-9.]+) \(%\)"


def parse_results(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    results = {}
    for line in lines:
        match = re.search(pattern, line)
        if match:
            tIoU, mAP, recall1x, recall5x = map(float, match.groups())
            results[tIoU] = {
                "mAP": mAP,
                "Recall@1x": recall1x,
                "Recall@5x": recall5x
            }
    return results


def parse_actionformer_results():
    results_dir = 'logs'
    files_list = os.listdir(results_dir)
    data = {}
    for file in files_list:
        if not file.endswith('.log'):
            continue
        backbone = file.split('_')[0]
        if backbone == 'omnivore':
            backbone += '_' + file.split('_')[1]
            if 'sub' in file:
                backbone += '_' + file.split('_')[2]
        division_type = file.split('_')[-2]
        videos_type = file.split('_')[-1].split('.')[0]
        results = parse_results(os.path.join(results_dir, file))
        print(results)
        data[backbone] = data.get(backbone, {})
        data[backbone][division_type] = data[backbone].get(division_type, {})
        data[backbone][division_type][videos_type] = results
    print(data)

    # Open a CSV file for writing
    with open('metrics_data/data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(['Backbone', 'DivisionType', 'VideosType', 'IoU', 'mAP', 'Recall@1x', 'Recall@5x'])

        # Loop through your JSON data and write it to the CSV
        for category, category_data in data.items():
            for subset, subset_data in category_data.items():
                for video_type, video_type_data in subset_data.items():
                    for iou, metrics in video_type_data.items():
                        writer.writerow(
                            [category, subset, video_type, iou, metrics['mAP'], metrics['Recall@1x'],
                             metrics['Recall@5x']]
                        )
    pass


def get_pivot_table():
    import pandas as pd
    metrics_folder = os.path.join(os.path.dirname(__file__), 'metrics_data')
    df = pd.read_csv(os.path.join(metrics_folder, 'data.csv'))
    video_types = df['VideosType'].unique()
    for video_type in video_types:
        df_video_type = df[df['VideosType'] == video_type]
        df_video_type.to_csv(os.path.join(metrics_folder, f'data_{video_type}.csv'))
        pt = df_video_type.pivot_table(
            index=['Backbone', 'DivisionType'],
            columns='IoU',
            values=['mAP', 'Recall@1x', 'Recall@5x']
        )
        pt.to_csv(os.path.join(metrics_folder, f'pivot_{video_type}.csv'))


if __name__ == '__main__':
    parse_actionformer_results()
    get_pivot_table()
