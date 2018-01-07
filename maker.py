import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

path = 'D:/workSpace/PyCharm/0104/ssd_keras/datasets/LaneMarkings/1'
for filename in os.listdir(path)[100:105]:
    sub_path = os.path.join(path, filename)
    print(sub_path)

    im = cv2.imread(sub_path)
    # cv2.imshow('', im)

    GrayImage = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh4 = cv2.threshold(GrayImage, 150, 255, cv2.THRESH_BINARY)
    plt.subplot(1, 2, 1), plt.imshow(im)
    plt.title('NORMAL')
    plt.subplot(1, 2, 2), plt.imshow(thresh4, 'gray')
    plt.title('THRESH_TOZERO')
    plt.xticks([]), plt.yticks([])
    plt.show()

# while True:
#     if cv2.waitKey(50) == ord('q'):
#         break
# print('Done.')
