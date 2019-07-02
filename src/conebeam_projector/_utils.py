# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import numpy as np
import pycuda.autoinit  # noqa
import pycuda.driver as cuda
from pycuda import gpuarray


def divup(a, b):
    if a % b:
        return a / b + 1
    else:
        return a / b


def ndarray_to_float_tex(tex_ref, ndarray, address_mode=cuda.address_mode.BORDER, filter_mode=cuda.filter_mode.LINEAR):
    if isinstance(ndarray, np.ndarray):
        cu_array = cuda.np_to_array(ndarray, 'C')
    elif isinstance(ndarray, gpuarray.GPUArray):
        cu_array = cuda.gpuarray_to_array(ndarray, 'C')
    else:
        raise TypeError(
            'ndarray must be numpy.ndarray or pycuda.gpuarray.GPUArray')

    cuda.TextureReference.set_array(tex_ref, cu_array)

    cuda.TextureReference.set_address_mode(
        tex_ref, 0, address_mode)
    if ndarray.ndim >= 2:
        cuda.TextureReference.set_address_mode(
            tex_ref, 1, address_mode)
    if ndarray.ndim >= 3:
        cuda.TextureReference.set_address_mode(
            tex_ref, 2,  address_mode)
    cuda.TextureReference.set_filter_mode(
        tex_ref, filter_mode)
    tex_ref.set_flags(tex_ref.get_flags(
    ) & ~cuda.TRSF_NORMALIZED_COORDINATES & ~cuda.TRSF_READ_AS_INTEGER)
