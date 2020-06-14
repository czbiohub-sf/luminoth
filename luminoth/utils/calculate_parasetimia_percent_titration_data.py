import pandas as pd
import sys
import os


if __name__ == "__main__":
    input_file = sys.argv[1]
    print(input_file)
    df = pd.read_csv(input_file)
    total_classes = len(df)
    parasite_classes = ["ring", "schizont", "troph"]
    filtered_df = df[df['label'].isin(parasite_classes)]
    for index, row in df.iterrows():
        image_path = row["image_id"]
        break
    print(image_path)
    titration_point = int(
        os.path.basename(image_path).split("Titration_point")[-1].split("_")[0])
    parasitemia_percentage = (len(filtered_df) / len(df)) * 100
    print(titration_point, parasitemia_percentage)
