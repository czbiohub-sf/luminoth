import numpy as np
import pandas as pd
import sys
import os

CONFIDENCE_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
LUMI_CSV_COLUMNS = [
    'image_id', 'xmin', 'xmax', 'ymin', 'ymax', 'label', 'prob']


if __name__ == "__main__":
    input_file = sys.argv[1]
    folder = os.path.dirname(input_file).split(os.sep)[-1]
    df = pd.read_csv(input_file)
    total_classes = len(df)
    parasite_classes = ["ring", "schizont", "troph"]
    filtered_df = df[df['label'].isin(parasite_classes)]
    parasitemia_percentages = {}
    for i in range(1, 6):
        for j in CONFIDENCE_THRESHOLDS:
            slice_total_df = df[df['image_id'].str.contains("sl{}".format(i))]
            slice_total_df = slice_total_df[slice_total_df['prob'] >= j]
            slice_filtered_df = slice_total_df[
                slice_total_df['label'].isin(parasite_classes)]
            parasitemia_percentages["sl{}_{}".format(i, j)] = (
                len(slice_filtered_df) / len(slice_total_df)) * 100
            parasitemia_percentages["sl_num{}_{}".format(i, j)] = \
                len(slice_total_df)
            parasitemia_percentages["sl_den{}_{}".format(i, j)] = \
                len(slice_filtered_df)

    for index, row in df.iterrows():
        image_path = row["image_id"]
        break
    if "Titration_point" in image_path:
        titration_point = int(
            os.path.basename(
                image_path).split("Titration_point")[-1].split("_")[0])
    elif "Titration_Point" in image_path:
        titration_point = int(
            os.path.basename(
                image_path).split("Titration_Point")[-1].split("_")[0])
    parasitemia_percentage = (len(filtered_df) / len(df)) * 100
    print(
        folder, titration_point, parasitemia_percentages,
        parasitemia_percentage)

    output_df = pd.DataFrame(columns=LUMI_CSV_COLUMNS + ['slice', 'split'])
    for index, row in df.iterrows():
        image_path = row["image_id"]
        split = os.path.basename(image_path).split("sl")
        slice_no = int(split[1][0])
        after_sl_ch_name = split[1].split('ch')[-1]
        output_df = output_df.append(
            {'image_id': row['image_id'],
             'xmin': np.int64(row['xmin']),
             'xmax': np.int64(row['xmax']),
             'ymin': np.int64(row['ymin']),
             'ymax': np.int64(row['ymax']),
             'label': row['label'],
             'slice': slice_no,
             'split': after_sl_ch_name + "index_{}".format(index),
             'prob': row['prob']}, ignore_index=True)

    split_grouped_by = output_df.groupby(["split", "slice"]).prob.max()
