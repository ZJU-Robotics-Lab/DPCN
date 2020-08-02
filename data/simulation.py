import cv2
import numpy as np
import random
try:
    import scipy.ndimage.interpolation as ndii
except ImportError:
    import ndimage.interpolation as ndii
import matplotlib.pyplot as plt


def generate_random_data(height, width, count):
    x, y, gt, trans = zip(*[generate_img_and_rot_img(height, width) for i in range(0, count)])

    X = np.asarray(x) * 255
    X = X.repeat(1, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)
    Y = np.asarray(y) * 100
    Y = Y.repeat(1, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)

    return X, Y, gt, trans

def generate_img_and_rot_img(height, width):
    shape = (height, width)

    triangle_location = get_random_location(*shape)
    triangle_location1 = get_random_location(*shape)
    triangle_location2 = get_random_location(*shape)
    circle_location1 = get_random_location(*shape, zoom=0.7)
    circle_location2 = get_random_location(*shape, zoom=0.5)
    circle_location3 = get_random_location(*shape, zoom=0.9)
    mesh_location = get_random_location(*shape)
    square_location = get_random_location(*shape, zoom=0.8)
    plus_location = get_random_location(*shape, zoom=1.2)
    plus_location1 = get_random_location(*shape, zoom=1.2)
    plus_location2 = get_random_location(*shape, zoom=1.2)
    plus_location3 = get_random_location(*shape, zoom=1.2)
    plus_location4 = get_random_location(*shape, zoom=1.2)

    # Create input image
    arr = np.zeros(shape, dtype=bool)
    arr = add_triangle(arr, *triangle_location)
    arr = add_triangle(arr, *triangle_location1)
    arr = add_triangle(arr, *triangle_location2)
    arr = add_circle(arr, *circle_location1)
    arr = add_circle(arr, *circle_location2, fill=True)
    arr = add_circle(arr, *circle_location3)
    arr = add_mesh_square(arr, *mesh_location)
    arr = add_filled_square(arr, *square_location)
    arr = add_plus(arr, *plus_location)
    arr = add_plus(arr, *plus_location1)
    arr = add_plus(arr, *plus_location2)
    arr = add_plus(arr, *plus_location3)
    arr = np.reshape(arr, (1, height, width)).astype(np.float32)

    angle = np.random.rand() * 180
    t_x = (np.random.rand()-0.5) * 50.
    t_y = (np.random.rand()-0.5) * 50.
    trans = np.array((t_y, t_x))

    if angle < -180.0:
        angle = angle + 360.0
    elif angle > 180.0:
        angle = angle - 360.0

    # arr = arr[0,]
    # rot = ndii.rotate(arr, angle)

    (_, h, w) = arr.shape
    (cX, cY) = (w//2, h//2)
    rot = arr[0,]

    N = np.float32([[1,0,t_x],[0,1,t_y]])
    rot = cv2.warpAffine(rot, N, (w, h))

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rot = cv2.warpAffine(rot, M, (w, h))
    rot = cv2.resize(rot, (h, w), interpolation=cv2.INTER_CUBIC)
# for heterogeneous image, comment out if you want them to be homogeneous
    rot = cv2.GaussianBlur(rot, (9,9), 13)
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    rot = cv2.filter2D(rot, -1, kernel)

    rot = rot[np.newaxis, :]
# for dynamic obstacles, comment out if you dont want any dynamic obstacles
    # arr[0,] = add_plus(arr[0], *plus_location4)
    # arr = np.reshape(arr, (1, height, width)).astype(np.float32)

    return arr, rot, angle, trans

def add_square(arr, x, y, size):
    s = int(size / 2)
    arr[x-s,y-s:y+s] = True
    arr[x+s,y-s:y+s] = True
    arr[x-s:x+s,y-s] = True
    arr[x-s:x+s,y+s] = True

    return arr

def add_filled_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

    return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, yy > y - s, yy < y + s]))

def logical_and(arrays):
    new_array = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        new_array = np.logical_and(new_array, a)

    return new_array

def add_mesh_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

    return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, xx % 2 == 1, yy > y - s, yy < y + s, yy % 2 == 1]))

def add_triangle(arr, x, y, size):
    s = int(size / 2)

    triangle = np.tril(np.ones((size, size), dtype=bool))

    arr[x-s:x-s+triangle.shape[0],y-s:y-s+triangle.shape[1]] = triangle

    return arr

def add_circle(arr, x, y, size, fill=False):
    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]
    circle = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    new_arr = np.logical_or(arr, np.logical_and(circle < size, circle >= size * 0.7 if not fill else True))

    return new_arr

def add_plus(arr, x, y, size):
    s = int(size / 2)
    arr[x-1:x+1,y-s:y+s] = True
    arr[x-s:x+s,y-1:y+1] = True

    return arr

def get_random_location(width, height, zoom=1.0):
    x = int(width * random.uniform(0.22, 0.78))
    y = int(height * random.uniform(0.22, 0.78))

    size = int(min(width, height) * random.uniform(0.06, 0.12) * zoom)

    return (x, y, size)