# while not len(input('Next:')):
#     print(111)
#
# a = input('Input your name:')
# print(a)


# import numpy as np
#
# a = np.random.uniform(0, 1, [1, 8096, 6])
#
# a[a[..., 1] > 0.2][:, 1] = 10
#
# print(1)


import csv
import os

path = '111.csv'
if os.path.exists(path):
    os.remove(path)

with open(path, 'w', newline="") as f:
    writer = csv.writer(f)
    head = ['x', 'y', 'score']
    writer.writerow(head)
    for i in range(20):
        writer.writerow([i + 1, i + 2, i + 3])
    print('Done.')
