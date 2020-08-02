import cv2
import numpy as np
import matplotlib.pyplot as plt
# from imreg_test import *
#from imreg_dft import imreg
#import imreg
from dft_test import _logpolar, similarity, imshow, _get_log_base

template = cv2.imread("./1.jpg",0)
source = cv2.imread("./1.jpg",0)
# template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
# source = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)

col, row = template.shape
center = (col // 2, row // 2)

t_x = -50.
t_y = -50.


M = cv2.getRotationMatrix2D(center, -160, 0.8)
source = cv2.warpAffine(source, M, (col, row))

N = np.float32([[1,0,t_x],[0,1,t_y]])
source = cv2.warpAffine(source, N, (col, row))
source = cv2.resize(source, (col, row), interpolation=cv2.INTER_CUBIC)
#logbase = _get_log_base(template.shape, template.shape)
##print(logbase)

# im2, scale, angle, (t0, t1) = similarity(template, source)
# print(scale, angle)
# imshow(template, source, im2)

# im2 = _logpolar(template, template.shape, 1.00878256)
result = similarity(source, template)
# im2 = template[0:300,0:1000]
# cv2.imshow("1", im2)
# cv2.waitKey(0)
# imshow(im2, im2, im2)

print(result['angle'], result['scale'], result)
# imshow(source, template, result['timg'])
# plt.show()




