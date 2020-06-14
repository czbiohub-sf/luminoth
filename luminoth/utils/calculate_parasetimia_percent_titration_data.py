import pandas as pd
import sys


if __name__ == "__main__":
    input_file = sys.argv[1]
    print(input_file)
    df = pd.read_csv(input_file)
    total_classes = len(df)
    parasite_classes = ["ring", "schizont", "troph"]
    filtered_df = df[df['label'].isin(parasite_classes)]
    parasitemia_percentage = [len(filtered_df) / len(df)] + 100
    print(parasitemia_percentage)
