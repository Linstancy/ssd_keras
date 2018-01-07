import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

path = 'D:/workSpace/PyCharm/0104/ssd_keras/datasets/LaneMarkings/1'
for filename in os.listdir(path)[:1]:
    sub_path = os.path.join(path, filename)
    print(sub_path)

    im = cv2.imread(sub_path)
    # cv2.imshow('', im)

    GrayImage = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, ts = cv2.threshold(GrayImage, 150, 255, cv2.THRESH_BINARY)

    height, width = ts.shape

    coords = []
    start = False
    start_coord = 0
    for column in range(width):
        column_data = ts[:, column] / 255.

        windows = 200
        repeat_rate = .8
        for row in range(height - windows):
            column_window_data = column_data[row: row + windows]
            if start:
                if sum(column_window_data) / float(windows) < repeat_rate:
                    start = False
                    coords.append((start_coord, column))
                    break
            else:
                if sum(column_window_data) / float(windows) > repeat_rate:
                    start = True
                    start_coord = column
                    break

    for coord in coords:
        start, end = coord
        im = cv2.rectangle(im, (start, 0), (end, height), (0, 0, 255), 1)

    cv2.imshow('', im)

while True:
    if cv2.waitKey(50) == ord('q'):
        break
print('Done.')
