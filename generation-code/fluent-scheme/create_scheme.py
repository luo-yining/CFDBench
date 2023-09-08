# save once each 0.25 flow time
# time step = 0.25
import numpy as np


src_file = "sample_step_geo.txt"
dst_file = "mesh208.scm"

string_old = [
    "{mesh}",
    "{density}",
    "{viscosity}",
    "{velocity}",
    "{old_var}",
    "{path_file}",  # e.g. D:\works\PINN-database\cavityflow\case1\case1.txt
    "{new_var}",
    "{num_time_steps}",
    "{save_path}",  # e.g. D:\works\PINN-database\cavityflow\case1
    "{case_name}",  # e.g. case1.cas.h5
    "{data_name}",  # e.g. case1.dat.h5
]

# density = np.hstack((np.linspace(0.1, 1, 91), np.linspace(1.05, 6.45, 109)))
# print(density)
# viscosity = np.linspace(1e-5, 2.09e-4, 200)
# print(viscosity)
# velocity = np.linspace(1, 20.9, 200)
# print(velocity)


# 2.For laminar flow
# velocity: 0.1-5 ,du = 0.1, #case: 0-49
# density: 10-1000, drho = 110, while viscosity: 0.01-1, dmu = 0.11 #case: 50-149
# -------------------------------------------------------
# num_time_steps : int = 1000
# standard_density = 100.0
# standard_viscosity = 0.1
# standard_velocity = 1.0
# u1 = np.linspace(0.1, 5, 50)
# u2 = np.ones(100)*standard_velocity
# velocity = np.hstack((u1, u2))
# rho1 = np.linspace(10, 1000, 10)
# mu1 = np.linspace(0.01, 1, 10)
# density = np.ones(50)*standard_density
# viscosity = np.ones(50)*standard_viscosity
# for i in range(0, 10):
#     rho2 = np.ones(10)*rho1[i]
#     density = np.hstack((density, rho2))
#     viscosity = np.hstack((viscosity, mu1))
# with open("laminar_set.txt", "w", encoding="utf8") as f:
#     for i in range(0, 150):
#         f.write(str(i) + " "+ str(velocity[i]) + " " + str(density[i]) + " " + str(viscosity[i]) + "\n")
# # print(velocity)
# print(density)
# print(viscosity)


# 1 For cavity flow
# velocity: 1-50 ,du = 1, #case: 0-49
# density: 0.1, 0.5, 1-10, drho = 1, while viscosity: 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2 #case: 50-1133
#-------------------------------------------------------
# num_time_steps : int = 1000
# standard_density = 1.0
# standard_viscosity = 1e-5
# standard_velocity = 10.0

# u1 = np.linspace(1, 50, 50)
# u2 = np.ones(84)*standard_velocity
# velocity = np.hstack((u1, u2))
# rho1 = np.linspace(1, 10, 10)
# rho1 = np.append(rho1, np.array([0.1, 0.5]))
# mu1 = np.array([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
# density = np.ones(50)*standard_density
# viscosity = np.ones(50)*standard_viscosity
# for i in range(0, 12):
#     rho2 = np.ones(7)*rho1[i]
#     density = np.hstack((density, rho2))
#     viscosity = np.hstack((viscosity, mu1))
# with open("cavity_set.txt", "w", encoding="utf8") as f:
#     for i in range(0, len(velocity)):
#         f.write(str(i) + " "+ str(velocity[i]) + " " + str(density[i]) + " " + str(viscosity[i]) + "\n")

# print(len(velocity), len(density), len(viscosity))
# print(velocity)
# print(density)
# print(viscosity)

# exit()

# 3 for karman vortex
# read velocity, density, viscosity from karman_set.txt
# -------------------------------------------------------
# velocity = []
# density = []
# viscosity = []
# with open("karman_set.txt", "r", encoding="utf8") as f:
#     lines = f.readlines()
#     lines = [line.strip() for line in lines]
#     for line in lines:
#         line = line.split()
#         velocity.append(float(line[1]))
#         density.append(float(line[3]))
#         viscosity.append(float(line[4]))
# num_time_steps : int = 2000
# standard_density = 10.0
# standard_viscosity = 1e-3
# standard_velocity = 1.0
# case_num = len(velocity)
# print(len(velocity), len(density), len(viscosity))
# print(velocity)
# print(density)
# print(viscosity)


# # 4.For step flow
# velocity: 0.05-1 du = 0.05; 1-2 ,du = 0.02, #case: 0-69
# density: 10-1000, drho = 110, while viscosity: 0.01-1, dmu = 0.11 #case: 70-169
# -------------------------------------------------------
num_time_steps : int = 100
standard_density = 100.0
standard_viscosity = 0.1
standard_velocity = 1.0
u1 = np.linspace(0.05, 1.0, 20)
u2 = np.linspace(1.02, 2.0, 50)
u3 = np.ones(100)*standard_velocity
velocity = np.hstack((u1, u2, u3))
rho1 = np.linspace(10, 1000, 10)
mu1 = np.linspace(0.01, 1, 10)
density = np.ones(70)*standard_density
viscosity = np.ones(70)*standard_viscosity
for i in range(0, 10):
    rho2 = np.ones(10)*rho1[i]
    density = np.hstack((density, rho2))
    viscosity = np.hstack((viscosity, mu1))
# with open("step_set.txt", "w", encoding="utf8") as f:
#     for i in range(0, 170):
#         f.write(str(i) + " "+ str(velocity[i]) + " " + str(density[i]) + " " + str(viscosity[i]) + "\n")
# print(velocity)
# print(density)
# print(viscosity)
# exit()


with open(dst_file, "a", encoding="utf8") as f_w:
    for i in range(208, 220):
        if i == 0:
            old_var = "export-0"
        else:
            old_var = "export-" + str(i - 1)
        new_var = "export-" + str(i)
        # old_var = "export-0"
        # new_var = "export-0"
        mesh = "E:\\CFDBench\\stepflow\\case" + str(i) + "\\project" + str(i-170) + ".msh"
        path_file = (
            "E:\\CFDBench\\stepflow\\case" + str(i) + "\\data" + str(i) + ".txt")
        save_path = "E:\\CFDBench\\stepflow\\case" + str(i)
        case_name = "case" + str(i) + ".cas.h5"
        data_name = "case" + str(i) + ".dat.h5"
        lines = open(src_file, "r", encoding="utf8").readlines()
        lines = [line.strip() for line in lines]
        output_lines = []
        for line in lines:
            line = line.replace(string_old[0], str(mesh))
            # line = line.replace(string_old[1], str(density[i]))
            # line = line.replace(string_old[2], str(viscosity[i]))
            # line = line.replace(string_old[3], str(velocity[i]))
            line = line.replace(string_old[4], str(old_var))
            line = line.replace(string_old[5], str(path_file))
            line = line.replace(string_old[6], str(new_var))
            # line = line.replace(string_old[7], str(num_time_steps))
            line = line.replace(string_old[8], str(save_path))
            line = line.replace(string_old[9], str(case_name))
            line = line.replace(string_old[10], str(data_name))
            output_lines.append(line)
        print(f"Writting case{i}, {len(output_lines)} lines")
        f_w.write("\n".join(output_lines) + "\n")
