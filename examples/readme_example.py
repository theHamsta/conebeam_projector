#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""


import numpy as np
import pyconrad.autoinit
import pyconrad.config
import pycuda.gpuarray as gpuarray

import conebeam_projector
from edu.stanford.rsl.conrad.phantom import NumericalSheppLogan3D

phantom = np.array(NumericalSheppLogan3D(
    *pyconrad.config.get_reco_size()).getNumericalSheppLoganPhantom(), np.float32)
pyconrad.imshow(phantom, "phantom")
projector = conebeam_projector.CudaProjector()

sino = gpuarray.zeros(pyconrad.config.get_sino_shape(), np.float32)

projector.forward_project_cuda_raybased(phantom, sino, use_maximum_intensity_projection=False)
pyconrad.imshow(sino, "Sinogram")
backprojection = projector.backProjectPixelDrivenCuda(sino)
pyconrad.imshow(backprojection, "backprojection")
