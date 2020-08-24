import matplotlib.pyplot as plt
import numpy as np
act = 150
data = open("outReactionSim.txt", "r")
if act == 75:
    datal = data.readlines()[97:120]
elif act == 100:
    datal = data.readlines()[73:96]
elif act == 150:
    datal = data.readlines()[49:72]
lsydata = datal[1:-1:2]
temps = datal[0:-1:2]

for i, x in enumerate(temps):
    temps[i] = float(x.split(" ")[1][0:4])

temps = np.array(temps)*3000

for i, x in enumerate(lsydata):
    lsydata[i] = [x.split("[")[1].split("]")[0]]
    temporary = lsydata[i][0].split(" ")
    for j in temporary:
        if j != "":
            lsydata[i].append(j)
    del lsydata[i][0]
    for l, z in enumerate(lsydata[i]):
        if z[-1] == ".":
            z += "0"
        lsydata[i][l] = float(z)
lsxdata = np.array(range(0, 1501, 75))
lsxdata = lsxdata / 1500 * 1000
lsydata = np.array(lsydata)
data.close()

print(lsydata)

# fig, ax = plt.subplots()
# ax.set_ylim(0, 100)
# ax.set_xlim(0, 1000)
# ax.set_ylabel('Percentage Reacted (%)')
# ax.set_xlabel('Time (ms)')
# ax.set_title('Activation Energy = {}kJ'.format(act))
# for i in range(len(lsydata)):
#     ax.plot(lsxdata, lsydata[i], label = "Temp = {:.0f}K".format(temps[i]))
#     ax.legend()

# fig.set_size_inches(12, 8)
# plt.show()