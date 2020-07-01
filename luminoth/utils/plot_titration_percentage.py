import matplotlib.pyplot as plt
import ast
import sys
import numpy as np


x = [18] + [8.5 * 0.5**i for i in range(0, 10)]
CONFIDENCE_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


# modify point 8 by combining the data
# iterate through the text lines skip if starts with python3
# process the rest

# draw y = x line, plot slice 1, slice 2, slice 3, slice 4, slice 5
# draw y = x line, plot slice confidence threshold
# 0.5, 0.55, 0.6, 0.65, 0.75 line

if __name__ == "__main__":
    f = open(sys.argv[1])
    lines = f.readlines()
    lines = [line for line in lines if not line.startswith("python")]
    titration_points = [int(line.split(" ")[1]) for line in lines]
    dictionaries = [
        ast.literal_eval("{" + line.split("{")[1].split("}")[0] + "}")
        for line in lines]

    slice_1 = [0] * 11
    slice_2 = [0] * 11
    slice_3 = [0] * 11
    slice_4 = [0] * 11
    slice_5 = [0] * 11

    simplified_dictionary = {}
    for titration_point, dictionary in zip(titration_points, dictionaries):
        simplified_dictionary[titration_point] = dictionary

    for titration_point, dictionary in simplified_dictionary.items():
        for key, value in dictionary.items():
            if "sl1_0.5" == key:
                slice_1[titration_point] = value
            if "sl2_0.5" == key:
                slice_2[titration_point] = value
            if "sl3_0.5" == key:
                slice_3[titration_point] = value
            if "sl4_0.5" == key:
                slice_4[titration_point] = value
            if "sl5_0.5" == key:
                slice_5[titration_point] = value
    labels = ['Y = X', 'Slice 1', 'Slice 2', 'Slice 3', 'Slice 4', 'Slice 5']
    colors = ['black', 'red', 'green', 'blue', 'cyan', 'yellow']
    plt.figure()
    plt.loglog(x, x, color=colors[0], label=labels[0])
    plt.loglog(x, slice_1, marker='o', color=colors[1], label=labels[1])
    plt.loglog(x, slice_2, marker='o', color=colors[2], label=labels[2])
    plt.loglog(x, slice_3, marker='o', color=colors[3], label=labels[3])
    plt.loglog(x, slice_4, marker='o', color=colors[4], label=labels[4])
    plt.loglog(x, slice_5, marker='o', color=colors[5], label=labels[5])

    plt.legend(loc='lower right')
    plt.show()
    slice_confidences = {
        confidence: [] for confidence in CONFIDENCE_THRESHOLDS}
    labels = []
    for confidence in CONFIDENCE_THRESHOLDS:
        for titration_point, dictionary in sorted(
                simplified_dictionary.items()):
            for key, value in dictionary.items():
                if confidence in key:
                    slice_confidences[confidence].append(value)
        labels.append("Confidence threshold {} %".format(int(
            confidence * 100)))
    n = len(labels)
    plt.figure()
    colors = plt.cm.jet(np.linspace(0, 1, n))
    for i in range(n):
        plt.loglog(
            x, slice_confidences[CONFIDENCE_THRESHOLDS[i]],
            marker='o', label=labels[i])
    plt.legend(loc='upper left')
    plt.show()
