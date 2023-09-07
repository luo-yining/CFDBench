# Needed Variables: w is width of barrier, h is height of barrier
# 1. Geometry:
#     point7 = 0,h,0
#     point8 = 0.5,h,0
#     point9 = 0.5+w,h,0
#     point10 = 0.5+w,0.1,0
#     point11 = 0.5+w,0,0
#     point12 = 1.5,h,0
# 2. Meshing:
#     nodes_x1 = w/0.01+1
#     nodes_x2 = (1-w)/0.01+1
#     nodes_y1 = (h-0.1)/0.01+1
#     nodes_y2 = (1-h)/0.01+1
# 3. Saving: project_name


import numpy as np

# calculate the height and width of the barrier
h_0 = np.linspace(0.11, 0.15, 5)
w_0 = np.linspace(0.01, 0.1, 10)

# create a list of all the possible combinations of h and w
hw = []
for i in range(len(h_0)):
    for j in range(len(w_0)):
        hw.append([h_0[i], w_0[j]])
# print(hw)
# print(len(hw))
# with open("hw.txt", "w", encoding="utf8") as f_w:
#     for i in range(len(hw)):
#         f_w.write(str(hw[i][0]) + " " + str(hw[i][1]) + "\n")
# exit()

# create replaced variables for the rpl file
string_old = [
    "{point4}",
    "{point5}",
    "{point6}",
    "{point8}",
    "{point11}",
    "{point12}",
    "{nodes_x1}",
    "{nodes_x2}",
    "{nodes_y1}",
    "{nodes_y2}",
    "{project_name}",
]


def get_variable(i):
    h = hw[i][0]
    w = hw[i][1]
    point8 = "0," + str(h) + ",0"
    point6 = "0.5," + str(h) + ",0"
    point5 = str(0.5+w) + "," + str(h) + ",0"
    point11 = str(0.5+w) + ",0.1,0"
    point4 = str(0.5+w) + ",0,0"
    point12 = "1.5," + str(h) + ",0"
    nodes_x1 = str(w/0.01+1)
    nodes_x2 = str((1-w)/0.01+1)
    nodes_y1 = str((h-0.1)/0.01+1)
    nodes_y2 = str((1-h)/0.01+1)
    project_name = "project" + str(i)
    string_new = [point4, point5, point6, point8, point11, point12, nodes_x1, nodes_x2, nodes_y1, nodes_y2, project_name]
    return string_new


# create rpl.file for meshing by replacing the variables in sample.rpl
src_file = "project_test.rpl"

for i in range(len(hw)):
    string_replace = get_variable(i)
    dst_file = string_replace[-1] + ".rpl"
    with open(dst_file, "a", encoding="utf8") as f_w:
        lines = open(src_file, "r", encoding="utf8").readlines()
        lines = [line.strip() for line in lines]
        output_lines = []
        for line in lines:
            # line = line.replace(string_old[0], str(mesh))
            line = line.replace(string_old[0], string_replace[0])
            line = line.replace(string_old[1], string_replace[1])
            line = line.replace(string_old[2], string_replace[2])
            line = line.replace(string_old[3], string_replace[3])
            line = line.replace(string_old[4], string_replace[4])
            line = line.replace(string_old[5], string_replace[5])
            line = line.replace(string_old[6], string_replace[6])
            line = line.replace(string_old[7], string_replace[7])
            line = line.replace(string_old[8], string_replace[8])
            line = line.replace(string_old[9], string_replace[9])
            line = line.replace(string_old[10], string_replace[10])
            output_lines.append(line)
        print(f"Writting case{i}, {len(output_lines)} lines")
        f_w.write("\n".join(output_lines) + "\n")
