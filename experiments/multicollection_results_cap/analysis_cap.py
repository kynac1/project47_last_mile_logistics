import json
import os
import matplotlib.pyplot as plt

# %matplotlib qt

data = {}
cd = os.path.dirname(os.path.abspath(__file__))
# fl = os.listdir(cd)
# os.path.join(
#         os.path.dirname(os.path.abspath(__file__)), "..", "data"
#     )

# fig, ax = plt.subplots()

for fname in os.listdir(cd):
    name, ext = os.path.splitext(fname)
    if ext == ".json":
        fields = name.split("_")
        if fields[0] == "constant":
            fields[1] = int(fields[1])
            fields[2] = int(fields[2])
            fields[3] = int(fields[3])
            fields[4] = int(fields[4])
            fields[7] = int(fields[7])
            fields[8] = int(fields[8])
            fields[9] = int(fields[9])
            fields[10] = int(fields[10])
            with open(os.path.join(cd, fname), "r") as f:
                data[tuple(fields[1::])] = json.load(f)
print(list(data.keys()))


packages_over_time = {}
for k, v in data.items():
    packages_over_time[k] = []
    for i in range(len(v) - 1):
        packages_over_time[k].append(
            v[i + 1]["number_of_packages"]
            - (
                v[i]["number_of_packages"]
                - len(v[i]["delivered_packages"]["days_taken"])
            )
        )

plt.cla()
for k, v in packages_over_time.items():
    if k[4] == "wait":
        plt.plot(v, label=str(k))
        plt.savefig("packages_over_time_cap.png")
# plt.legend()
# plt.show()

packages_over_time = {}
for k, v in data.items():
    packages_over_time[k] = []
    for i in range(len(v) - 1):
        packages_over_time[k].append(
            v[i + 1]["number_of_packages"]
            - (
                v[i]["number_of_packages"]
                - len(v[i]["delivered_packages"]["days_taken"])
            )
        )

plt.cla()
for k, v in packages_over_time.items():
    if k[4] == "wait":
        plt.plot(v, label=str(k))
        plt.savefig("packages_over_time_cap.png")