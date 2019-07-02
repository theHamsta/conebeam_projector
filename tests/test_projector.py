# -*- coding: utf-8 -*-
"""

"""
import os

import mock
import numpy as np
import pyconrad.autoinit
import pyconrad.config
import pycuda.gpuarray as gpuarray

import conebeam_projector
from edu.stanford.rsl.conrad.phantom import NumericalSheppLogan3D

if "CI" in os.environ:
    pyconrad.imshow = mock.MagicMock()


def test_forward_projection():
    phantom = np.array(NumericalSheppLogan3D(
        *pyconrad.config.get_reco_size()).getNumericalSheppLoganPhantom(), np.float32)
    pyconrad.imshow(phantom, "phantom")
    projector = conebeam_projector.CudaProjector()

    sino = gpuarray.zeros(pyconrad.config.get_sino_shape(), np.float32)

    for use_max_intensity in [True, False]:
        projector.forward_project_cuda_raybased(phantom, sino, use_maximum_intensity_projection=use_max_intensity)
        pyconrad.imshow(sino, "Sinogram %s" % use_max_intensity)


def test_for_and_backward_projection():
    phantom = np.array(NumericalSheppLogan3D(
        *pyconrad.config.get_reco_size()).getNumericalSheppLoganPhantom(), np.float32)
    pyconrad.imshow(phantom, "phatom")
    projector = conebeam_projector.CudaProjector()

    sino = gpuarray.zeros(pyconrad.config.get_sino_shape(), np.float32)

    for use_max_intensity in [False]:
        projector.forward_project_cuda_raybased(phantom, sino, use_maximum_intensity_projection=use_max_intensity)
        pyconrad.imshow(sino, "Sinogram %s" % use_max_intensity)
        backprojection = projector.backProjectPixelDrivenCuda(sino)
        pyconrad.imshow(backprojection, "backprojection")


def test_backward_projection():

    sino = np.random.rand(*pyconrad.config.get_sino_shape())
    sino = gpuarray.to_gpu(np.ascontiguousarray(sino, np.float32))
    backprojection = gpuarray.zeros(pyconrad.config.get_reco_shape(), np.float32)

    projector = conebeam_projector.CudaProjector()

    for use_max_intensity in [False]:
        projector.backProjectPixelDrivenCuda(sino, backprojection)
        pyconrad.imshow(backprojection, "backprojection")

    backprojection.fill(0)
    for i in range(sino.shape[0]):
        projector.backProjectPixelDrivenCudaIdx(sino[i], backprojection, i)
        pyconrad.imshow(backprojection, "backprojection with index")


def main():
    test_forward_projection()
    test_backward_projection()
    test_for_and_backward_projection()


if __name__ == '__main__':
    main()
