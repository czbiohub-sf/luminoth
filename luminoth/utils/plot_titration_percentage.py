import matplotlib.pyplot as plt
import ast
from collections import defaultdict


def dsum(dicts):
    ret = defaultdict(float)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)


x = [18] + [8.5 * 0.5**i for i in range(0, 10)]

# modify point 8 by combining the data
# iterate through the text lines skip if starts with python3
# process the rest

# draw y = x line, plot slice 1, slice 2, slice 3, slice 4, slice 5
# draw y = x line, plot slice confidence threshold
# 0.5, 0.55, 0.6, 0.65, 0.75 line
f = open("output_titration_per_slice.txt")
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
eights_dictionaries = []
for titration_point, dictionary in zip(titration_points, dictionaries):
    if titration_point == 8:
        # accumulate the values per slice percentages
        eights_dictionaries.append(dictionary)
    else:
        simplified_dictionary[titration_point] = dictionary

simplified_dictionary[8] = dsum(eights_dictionaries)
simplified_dictionary[8]['sl1_0.5'] = \
    (simplified_dictionary[8]['sl_num1_0.5'] / simplified_dictionary[8]['sl_den1_0.5']) / 100
simplified_dictionary[8]['sl2_0.5'] = \
    (simplified_dictionary[8]['sl_num2_0.5'] / simplified_dictionary[8]['sl_den2_0.5']) * 100
simplified_dictionary[8]['sl3_0.5'] = \
    (simplified_dictionary[8]['sl_num3_0.5'] / simplified_dictionary[8]['sl_den3_0.5']) * 100
simplified_dictionary[8]['sl4_0.5'] = \
    (simplified_dictionary[8]['sl_num4_0.5'] / simplified_dictionary[8]['sl_den4_0.5']) * 100
simplified_dictionary[8]['sl5_0.5'] = \
    (simplified_dictionary[8]['sl_num5_0.5'] / simplified_dictionary[8]['sl_den5_0.5']) * 100

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
colors = ['black', 'blue', 'blue', 'blue', 'blue', 'blue']

plt.loglog(x, x, color=colors[0])
plt.loglog(x, slice_1, marker='o', color=colors[1])
plt.loglog(x, slice_2, marker='o', color=colors[2])
plt.loglog(x, slice_3, marker='o', color=colors[3])
plt.loglog(x, slice_4, marker='o', color=colors[4])
plt.loglog(x, slice_5, marker='o', color=colors[5])


plt.legend(
    ['Y = X', 'Slice 1', 'Slice 2', 'Slice 3', 'Slice 4', 'Slice 5'],
    loc='lower right')

plt.show()


plt.gca().set_color_cycle(['black', 'blue', 'blue', 'blue', 'blue', 'blue'])
plt.loglog(x, slice_3_confidence_point5)
plt.loglog(x, slice_3_confidence_point55)
plt.loglog(x, slice_3_confidence_point6)
plt.loglog(x, slice_3_confidence_point65)
plt.loglog(x, slice_3_confidence_point7)
plt.loglog(x, slice_3_confidence_point75)
plt.loglog(x, slice_3_confidence_point8)
plt.loglog(x, slice_3_confidence_point85)
plt.loglog(x, slice_3_confidence_point9)
plt.loglog(x, slice_3_confidence_point95)





plt.legend(
    ['Y = X', 'Slice 1', 'Slice 2', 'Slice 3', 'Slice 4', 'Slice 5'],
    loc='lower right')

plt.show()