"""
    list_bestn.py
"""

rmsd_list = []

with open("rmsd.csv") as fh:
    for k, line in enumerate(fh):
        # print(line)
        code, rmsd = line[:-1].split(",")
        v = float(rmsd)
        # print([k], code, v)
        rmsd_list.append((code, v))
        if k > 3:
            # break
            pass

rmsd_list = sorted(rmsd_list, key=lambda x: x[1])

for pair in rmsd_list[0:5]:
    print(*pair)
