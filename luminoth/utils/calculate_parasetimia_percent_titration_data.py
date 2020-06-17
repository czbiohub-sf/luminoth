import pandas as pd
import sys
import os

CONFIDENCE_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


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
            slice_total_df = slice_total_df[slice_total_df['prob' >= j]]
            slice_filtered_df = slice_total_df[
                slice_total_df['label'].isin(parasite_classes)]
            parasitemia_percentages["sl{}_{}".format(i, j)] = (
                len(slice_filtered_df) / len(slice_total_df)) * 100

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

    for index, row in df.iterrows():
        image_path = row["image_id"]