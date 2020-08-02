# imreg.py

# Copyright (c) 2011-2020, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""FFT based image registration.
Imreg is a Python library that implements an FFT-based technique for
translation, rotation and scale-invariant image registration [1].
:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_
:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine
:License: BSD 3-Clause
:Version: 2020.1.1
Requirements
------------
* `CPython >= 3.6 <https://www.python.org>`_
* `Numpy 1.14 <https://www.numpy.org>`_
* `Scipy 1.3 <https://www.scipy.org>`_
* `Matplotlib 3.1 <https://www.matplotlib.org>`_  (optional for plotting)
Notes
-----
Imreg is no longer being actively developed.
This implementation is mainly for educational purposes.
An improved version is being developed at https://github.com/matejak/imreg_dft.
References
----------
1. An FFT-based technique for translation, rotation and scale-invariant
   image registration. BS Reddy, BN Chatterji.
   IEEE Transactions on Image Processing, 5, 1266-1271, 1996
2. An IDL/ENVI implementation of the FFT-based algorithm for automatic
   image registration. H Xiea, N Hicksa, GR Kellera, H Huangb, V Kreinovich.
   Computers & Geosciences, 29, 1045-1055, 2003.
3. Image Registration Using Adaptive Polar Transform. R Matungka, YF Zheng,
   RL Ewing. IEEE Transactions on Image Processing, 18(10), 2009.
Examples
--------
>>> im0 = imread('t400')
>>> im1 = imread('Tr19s1.3')
>>> im2, scale, angle, (t0, t1) = similarity(im0, im1)
>>> imshow(im0, im1, im2)
>>> im0 = imread('t350380ori')
>>> im1 = imread('t350380shf')
>>> t0, t1 = translation(im0, im1)
>>> t0, t1
(20, 50)
"""

__version__ = '2020.1.1'

__all__ = (
    'translation', 'similarity', 'similarity_matrix', 'logpolar', 'highpass',
    'imread', 'imshow', 'ndii'
)

import math
import cv2
import numpy
from numpy.fft import fft2, ifft2, fftshift

try:
    import scipy.ndimage.interpolation as ndii
except ImportError:
    import ndimage.interpolation as ndii


def translation(im0, im1):
    """Return translation vector to register images."""
    shape = im0.shape
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = numpy.unravel_index(numpy.argmax(ir), shape)
    if t0 > shape[0] // 2:
        t0 = t0 - shape[0]
    if t1 > shape[1] // 2:
        t1 = t0 - shape[1]
    return [t0, t1]

def similarity(im0, im1):
    """Return similarity transformed image im1 and transformation parameters.
    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.
    A similarity transformation is an affine transformation with isotropic
    scale and without shear.
    Limitations:
    Image shapes must be equal and square.
    All image areas must have same scale, rotation, and shift.
    Scale change must be less than 1.8.
    No subpixel precision.
    """
    if im0.shape != im1.shape:
        raise ValueError('images must have same shapes')
    if len(im0.shape) != 2:
        raise ValueError('images must be 2 dimensional')

    f0 = fftshift(abs(fft2(im0)))
    f1 = fftshift(abs(fft2(im1)))

    h = highpass(f0.shape)
    f0 *= h
    f1 *= h
    del h

    f0, log_base = logpolar(f0)
    f1, log_base = logpolar(f1)

    f0 = fft2(f0)
    f1 = fft2(f1)
    eps=1e-10
    r0 = abs(f0) * abs(f1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (r0 + eps)))
    ir = fftshift(ir)

    i0, i1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)
    # i0 -= f0.shape[0] // 2
    # i1 -= f0.shape[1] // 2
    print(i0, i1)
    angle = -180.0 * i0 / ir.shape[0]
    scale = log_base ** i1
    print(angle, scale)
    if scale > 1.8:
        ir = abs(ifft2((f1 * f0.conjugate()) / (r0 + eps)))
        ir = fftshift(ir)
        print("***********************")
        i0, i1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)
        i0 = i0-ir.shape[0] // 2
        i1 = i1-ir.shape[0] // 2
        # imshow(ir*10000,ir*10000,ir*10000)
        print(i0, i1)

        angle = 180.0 * i0 / ir.shape[0]
        scale = 1.0 / (log_base ** i1)
        if scale > 1.8:
            raise ValueError('images are not compatible. Scale change > 1.8')

    # if angle < -90.0:
    #     angle += 180.0
    # elif angle > 90.0:
    #     angle -= 180.0

    print(angle, scale)

    im2 = ndii.zoom(im1, 1.0/scale)
    im2 = ndii.rotate(im2, -angle)
    if im2.shape < im0.shape:
        t = numpy.zeros_like(im0)
        t[:im2.shape[0], :im2.shape[1]] = im2
        im2 = t
    elif im2.shape > im0.shape:
        im2 = im2[:im0.shape[0], :im0.shape[1]]

    f0 = fft2(im0)
    f1 = fft2(im2)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)
    
    f2_rot = numpy.rot90(f1,2)
    f2_rot = f2_rot[:im0.shape[0], :im0.shape[1]]
    ir_rot = abs(ifft2((f0 * f2_rot.conjugate()) / (abs(f0) * abs(f2_rot))))
    t0_rot, t1_rot = numpy.unravel_index(numpy.argmax(ir_rot), ir_rot.shape)
    
    print("compare",ir[t0,t1],ir_rot[t0_rot,t1_rot])
    if(ir[t0,t1] < ir_rot[t0_rot,t1_rot]):
        angle = angle + 180
        im2 = numpy.rot90(im2, -180)

    if t0 > f0.shape[0] // 2:
        t0 = t0-f0.shape[0]
    if t1 > f0.shape[1] // 2:
        t1 = t1-f0.shape[1]

    im2 = ndii.shift(im2, [t0, t1])

    # correct parameters for ndimage's internal processing
    if angle > 0.0:
        d = int(int(im1.shape[1] / scale) * math.sin(math.radians(angle)))
        t0, t1 = t1, d+t0
    elif angle < 0.0:
        d = int(int(im1.shape[0] / scale) * math.sin(math.radians(angle)))
        t0, t1 = d+t1, d+t0
    scale = (im1.shape[1] - 1) / (int(im1.shape[1] / scale) - 1)

    if angle < -180.0:
        angle = angle+360.0
    elif angle > 180.0:
        angle = angle-360.0

    return im2, scale, angle, [-t0, -t1]


def similarity_matrix(scale, angle, vector):
    """Return homogeneous transformation matrix from similarity parameters.
    Transformation parameters are: isotropic scale factor, rotation angle
    (in degrees), and translation vector (of size 2).
    The order of transformations is: scale, rotate, translate.
    """
    S = numpy.diag([scale, scale, 1.0])
    R = numpy.identity(3)
    angle = math.radians(angle)
    R[0, 0] = math.cos(angle)
    R[1, 1] = math.cos(angle)
    R[0, 1] = -math.sin(angle)
    R[1, 0] = math.sin(angle)
    T = numpy.identity(3)
    T[:2, 2] = vector
    return numpy.dot(T, numpy.dot(R, S))


def logpolar(image, angles=None, radii=None):
    """Return log-polar transformed image and log base."""
    shape = image.shape
    center = shape[0] / 2, shape[1] / 2
    if angles is None:
        angles = shape[0]
    if radii is None:
        radii = shape[1]
    theta = numpy.empty((angles, radii), dtype='float64')
    theta.T[:] = numpy.linspace(0, numpy.pi, angles, endpoint=False) * -1.0
    # d = radii
    d = numpy.hypot(shape[0] - center[0], shape[1] - center[1])
    log_base = 10.0 ** (math.log10(d) / (radii))
    radius = numpy.empty_like(theta)
    radius[:] = numpy.power(log_base,
                            numpy.arange(radii, dtype='float64')) - 1.0
    x = radius * numpy.sin(theta) + center[0]
    y = radius * numpy.cos(theta) + center[1]
    output = numpy.empty_like(x)
    ndii.map_coordinates(image, [x, y], output=output)
    return output, log_base


def highpass(shape):
    """Return highpass filter to be multiplied with fourier transform."""
    x = numpy.outer(
        numpy.cos(numpy.linspace(-math.pi/2.0, math.pi/2.0, shape[0])),
        numpy.cos(numpy.linspace(-math.pi/2.0, math.pi/2.0, shape[1])))
    return (1.0 - x) * (2.0 - x)


def imread(fname, norm=True):
    """Return image data from img&hdr uint8 files."""
    with open(fname + '.hdr', 'r') as fh:
        hdr = fh.readlines()
    img = numpy.fromfile(fname + '.img', numpy.uint8, -1)
    img.shape = int(hdr[4].split()[-1]), int(hdr[3].split()[-1])
    if norm:
        img = img.astype('float64')
        img = img/255.0
    return img


def imshow(im0, im1, im2, im3=None, cmap=None, **kwargs):
    """Plot images using matplotlib."""
    from matplotlib import pyplot

    if im3 is None:
        im3 = abs(im2 - im0)
    pyplot.subplot(221)
    pyplot.imshow(im0, cmap, **kwargs)
    pyplot.subplot(222)
    pyplot.imshow(im1, cmap, **kwargs)
    pyplot.subplot(223)
    pyplot.imshow(im3, cmap, **kwargs)
    pyplot.subplot(224)
    pyplot.imshow(im2, cmap, **kwargs)
    pyplot.show()


if __name__ == '__main__':
    import os
    import doctest

    try:
        os.chdir('data')
    except Exception:
        pass
    doctest.testmod()