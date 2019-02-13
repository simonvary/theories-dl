"""
..
    2018, Simon Vary
Module providing basic 2D transforms
Routine listings
----------------
transform_wavelet2(method)
    2D wavelet transform.

transform_wavelet2(method)
    2D wavelet transform.
"""

import numpy as np
import pywt

class transform_wavelet2(object):
    def __init__(self, wavelet_name, **kwargs):
        self._wavelet_name = wavelet_name
        self._mode = kwargs.get('mode', 'symmetric')
        self._level = kwargs.get('level', None)
        self._coeff_slices = None
        self._coeff_shapes = None

    def forward(self, x):
        _x = x.copy()
        coeffs = pywt.wavedecn(x,
                            self._wavelet_name,
                            mode = self._mode,
                            level = self._level)
        # First forward pass initializes the transform
        # by introducing dimensions and wavelet shapes.
        if (self._coeff_slices==None) or (self._coeff_shapes == None):
            coeff_vec, self._coeff_slices, self._coeff_shapes = pywt.ravel_coeffs(coeffs)
            self.m = coeff_vec.size
            self.n = x.size
        else:
            coeff_vec, _, _ = pywt.ravel_coeffs(coeffs)
        return coeff_vec

    def backward(self, y):
        if (self._coeff_slices==None) or (self._coeff_shapes == None):
            print('Warning! Cannot do backward() before initialized forward()!')
        else:
            coeffs = pywt.unravel_coeffs(y, self._coeff_slices, self._coeff_shapes)
            x = pywt.waverecn(coeffs, self._wavelet_name, mode = self._mode)
        return x

class transform_randsubsample2(object):
    def __init__(self, image_shape, delta):
        self._image_shape = image_shape
        self._delta = delta
        self.n = np.prod(image_shape)
        self.m = int(delta*np.prod(image_shape))
        self._mask = np.sort(np.random.choice(self.n, self.m))

    def forward(self, x):
        _x = x.copy()
        return _x.reshape(self.n,1)[self._mask]

    def backward(self, y):
        x = np.zeros((self.n,1))
        x[self._mask] = y
        return x.reshape(self._image_shape[0], self._image_shape[1])
