import collections
import os

import numpy as np
import pyconrad
import pyconrad.config
import pycuda.autoinit  # NOQA
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from PIL import Image
from pycuda.compiler import SourceModule

from conebeam_projector._utils import divup, ndarray_to_float_tex

_SMALL_VALUE = 1e-12

_ = pyconrad.ClassGetter(
    'edu.stanford.rsl.conrad.numerics'
)  # type: pyconrad.AutoCompleteConrad


class CudaProjector:
    def __init__(self):
        self.is_configured = False
        self._manually_set_normalizer = None
        self._init_kernels()
        self._num_projections = pyconrad.config.get_geometry().getProjectionStackSize()
        self.start_idx = 0
        self.forward_projection_stepsize = 1.
        self.stop_idx = self._num_projections

    def _init_config(self):
        geo = pyconrad.config.get_geometry()

        self._voxelSize = (geo.getVoxelSpacingX(),
                           geo.getVoxelSpacingY(),
                           geo.getVoxelSpacingZ())
        self._volumeSize = (geo.getReconDimensionX(),
                            geo.getReconDimensionY(),
                            geo.getReconDimensionZ())
        self._origin = (geo.getOriginX(),
                        geo.getOriginY(),
                        geo.getOriginZ())

        self._volshape = pyconrad.config.get_reco_shape()

        self._volumeEdgeMaxPoint = []
        for i in range(3):
            self._volumeEdgeMaxPoint.append(self._volumeSize[i] -
                                            1. - _SMALL_VALUE)
            # self._volumeEdgeMaxPoint.append( self._volumeSize[i] -
            #     0.5 - SMALL_VALUE)

        self._volumeEdgeMinPoint = []
        for i in range(3):
            self._volumeEdgeMinPoint.append(-1. + _SMALL_VALUE)

        self._projectionMatrices = pyconrad.config.get_projection_matrices()
        self._num_projections = geo.getProjectionStackSize()
        self._width = geo.getDetectorWidth()
        self._height = geo.getDetectorHeight()
        self._srcPoint = np.ndarray([3 * self._num_projections], np.float32)
        self._invARmatrix = np.ndarray(
            [3 * 3 * self._num_projections], np.float32)
        for i in range(self._num_projections):
            self._computeCanonicalProjectionMatrix(i)
        self._inv_AR_matrix_gpu = gpuarray.to_gpu(self._invARmatrix)
        self._proj_mats_gpu = gpuarray.to_gpu(self._projectionMatrices)
        if self._manually_set_normalizer is None:
            self._normalizer = geo.getSourceToAxisDistance(
            ) * geo.getSourceToDetectorDistance() * np.pi / self._num_projections
        else:
            self._normalizer = self._manually_set_normalizer
        self.is_configured = True

# TODO: property for _manually_set_normalizer

    def _init_kernels(self):
        with open(os.path.join(os.path.dirname(__file__), 'BackProjectionKernel.cu')) as f:
            read_data = f.read()
            f.closed

        self._module_backprojection = SourceModule(read_data)
        self._tex_backprojection = self._module_backprojection.get_texref(
            'tex_sino')
        self._kernel_backprojection = self._module_backprojection.get_function(
            "backProjectionKernel")
        self._kernel_backprojection_withvesselmask = self._module_backprojection.get_function(
            "backProjectionKernelWithConstrainingVolume")
        self._kernel_backprojection_multiplicative = self._module_backprojection.get_function(
            "backprojectMultiplicative")

        with open(os.path.join(os.path.dirname(__file__), 'ForwardProjectionKernel.cu')) as f:
            read_data = f.read()
        f.closed

        self._module_forwardprojection = SourceModule(read_data)
        self._kernel_forwardprojection = self._module_forwardprojection.get_function(
            "forwardProjectionKernel")
        self._tex_forwardprojection = self._module_forwardprojection.get_texref(
            'gTex3D')

    def _getOriginTransform(self):
        currOrigin = _.SimpleVector.from_list(self._origin)

        centeredOffset = _.SimpleVector.from_list(self._volumeSize)
        voxelSpacing = _.SimpleVector.from_list(self._voxelSize)
        centeredOffset.subtract(1.)
        centeredOffset.multiplyElementWiseBy([voxelSpacing])
        centeredOffset.divideBy(-2.)

        return _.SimpleOperators.subtract(currOrigin, centeredOffset)

    def forward_project_cuda_raybased(self, vol, sino_gpu, use_maximum_intensity_projection=False):
        self._check_config()
        assert sino_gpu.ndim == 3

        if isinstance(vol, gpuarray.GPUArray) or (isinstance(vol, np.ndarray) and vol.ndim == 3):
            ndarray_to_float_tex(
                self._tex_forwardprojection, vol)

        block = (32, 8, 1)
        grid = (int(divup(sino_gpu.shape[2], block[0])),
                int(divup(sino_gpu.shape[1], block[1])), 1)

        for t in range(sino_gpu.shape[0]):
            if isinstance(vol, list) or (isinstance(vol, np.ndarray) and vol.ndim == 4):
                ndarray_to_float_tex(
                    self._tex_forwardprojection, vol[t])
                assert vol[t].dtype == np.float32
            elif isinstance(vol, collections.Callable):
                ndarray_to_float_tex(
                    self._tex_forwardprojection, vol(t))
                assert vol(t).dtype == np.float32
            elif isinstance(vol, gpuarray.GPUArray) or (isinstance(vol, np.ndarray) and vol.ndim == 3):
                assert vol.dtype == np.float32
            else:
                raise TypeError('!')

            assert sino_gpu.dtype == np.float32

            self._kernel_forwardprojection(
                sino_gpu[t],
                np.int32(sino_gpu.shape[2]),
                np.int32(sino_gpu.shape[1]),
                np.float32(self.forward_projection_stepsize),
                np.float32(self._voxelSize[0]),
                np.float32(self._voxelSize[1]),
                np.float32(self._voxelSize[2]),
                np.float32(self._volumeEdgeMinPoint[0]),
                np.float32(self._volumeEdgeMinPoint[1]),
                np.float32(self._volumeEdgeMinPoint[2]),
                np.float32(self._volumeEdgeMaxPoint[0]),
                np.float32(self._volumeEdgeMaxPoint[1]),
                np.float32(self._volumeEdgeMaxPoint[2]),
                np.float32(self._srcPoint[3 * t + 0]),
                np.float32(self._srcPoint[3 * t + 1]),
                np.float32(self._srcPoint[3 * t + 2]),
                self._inv_AR_matrix_gpu,
                np.int32(t),
                np.int32(use_maximum_intensity_projection),
                grid=grid, block=block
            )

    def forward_project_cuda_idx(self, vol, sino_gpu, idx, use_maximum_intensity_projection=False):
        self._check_config()

        assert idx >= 0 and idx < self._num_projections, "Invalid projection index"
        assert sino_gpu.ndim == 2
        assert sino_gpu.size == sino_gpu.shape[1] * sino_gpu.shape[0]
        assert sino_gpu.dtype == np.float32

        block = (32, 8, 1)
        grid = (int(divup(sino_gpu.shape[1], block[0])),
                int(divup(sino_gpu.shape[0], block[1])), 1)

        if isinstance(vol, list) or (isinstance(vol, np.ndarray) and vol.ndim == 4):
            ndarray_to_float_tex(self._tex_forwardprojection, vol[idx])
        elif isinstance(vol, collections.Callable):
            ndarray_to_float_tex(self._tex_forwardprojection, vol(idx))
        elif isinstance(vol, gpuarray.GPUArray) or (isinstance(vol, np.ndarray) and vol.ndim == 3):
            ndarray_to_float_tex(self._tex_forwardprojection, vol)
        else:
            raise TypeError('!')

        self._kernel_forwardprojection(
            sino_gpu,
            np.int32(sino_gpu.shape[1]),
            np.int32(sino_gpu.shape[0]),
            np.float32(self.forward_projection_stepsize),  # step size
            np.float32(self._voxelSize[0]),
            np.float32(self._voxelSize[1]),
            np.float32(self._voxelSize[2]),
            np.float32(self._volumeEdgeMinPoint[0]),
            np.float32(self._volumeEdgeMinPoint[1]),
            np.float32(self._volumeEdgeMinPoint[2]),
            np.float32(self._volumeEdgeMaxPoint[0]),
            np.float32(self._volumeEdgeMaxPoint[1]),
            np.float32(self._volumeEdgeMaxPoint[2]),
            np.float32(self._srcPoint[3 * idx + 0]),
            np.float32(self._srcPoint[3 * idx + 1]),
            np.float32(self._srcPoint[3 * idx + 2]),
            self._inv_AR_matrix_gpu,
            np.int32(idx),
            np.int32(use_maximum_intensity_projection),
            np.float32(0.1),
            grid=grid, block=block
        )

    def _computeCanonicalProjectionMatrix(self, projIdx):

        geo = pyconrad.config.get_geometry()

        proj = geo.getProjectionMatrices()[projIdx]
        self._invVoxelScale = _.SimpleMatrix(3, 3)
        self._invVoxelScale.setElementValue(0, 0, 1.0 / self._voxelSize[0])
        self._invVoxelScale.setElementValue(1, 1, 1.0 / self._voxelSize[1])
        self._invVoxelScale.setElementValue(2, 2, 1.0 / self._voxelSize[2])

        invARmatrixMat = proj.getRTKinv()

        invAR = _.SimpleOperators.multiplyMatrixProd(
            self._invVoxelScale, invARmatrixMat)

        counter = 3 * 3 * projIdx
        for r in range(3):
            for c in range(3):
                self._invARmatrix[counter] = invAR.getElement(r, c)
                counter += 1

        originShift = self._getOriginTransform()

        srcPtW = proj.computeCameraCenter().negated()
        self._srcPoint[3 * projIdx + 0] = -(-0.5 * (self._volumeSize[0] - 1.0) + originShift.getElement(
            0) * self._invVoxelScale.getElement(0, 0) + self._invVoxelScale.getElement(0, 0) * srcPtW.getElement(0))
        self._srcPoint[3 * projIdx + 1] = -(-0.5 * (self._volumeSize[1] - 1.0) + originShift.getElement(
            1) * self._invVoxelScale.getElement(1, 1) + self._invVoxelScale.getElement(1, 1) * srcPtW.getElement(1))
        self._srcPoint[3 * projIdx + 2] = -(-0.5 * (self._volumeSize[2] - 1.0) + originShift.getElement(
            2) * self._invVoxelScale.getElement(2, 2) + self._invVoxelScale.getElement(2, 2) * srcPtW.getElement(2))

    def backProjectPixelDrivenCudaIdx(self, sino_gpu: gpuarray.GPUArray, vol_gpu: gpuarray.GPUArray, projIdx, constraining_vol=None, multiplicative=False):

        self._check_config()

        assert projIdx >= 0 and projIdx < self._num_projections, "Invalid projection index"
        assert sino_gpu.ndim == 2
        assert vol_gpu.shape == self._volshape
        assert sino_gpu.dtype == np.float32
        assert vol_gpu.dtype == np.float32
        if constraining_vol:
            assert isinstance(constraining_vol, gpuarray.GPUArray)

        ndarray_to_float_tex(self._tex_backprojection, sino_gpu)

        block = (32, 8, 1)
        grid = (int(divup(vol_gpu.shape[2], block[0])),
                int(divup(vol_gpu.shape[1], block[1])), 1)

        if constraining_vol:
            if multiplicative:
                self._kernel_backprojection_multiplicative(vol_gpu.gpudata,
                                                           self._proj_mats_gpu,
                                                           constraining_vol,
                                                           np.int32(projIdx),
                                                           np.int32(
                                                               self._volshape[2]),
                                                           np.int32(
                                                               self._volshape[1]),
                                                           np.int32(
                                                               self._volshape[0]),
                                                           np.float32(
                                                               -self._origin[0]),
                                                           np.float32(
                                                               -self._origin[1]),
                                                           np.float32(
                                                               -self._origin[2]),
                                                           np.float32(
                                                               self._voxelSize[0]),
                                                           np.float32(
                                                               self._voxelSize[1]),
                                                           np.float32(
                                                               self._voxelSize[2]),
                                                           np.float32(
                                                               self._normalizer),
                                                           grid=grid, block=block
                                                           )
            else:
                self._kernel_backprojection_withvesselmask(vol_gpu.gpudata,
                                                           self._proj_mats_gpu,
                                                           constraining_vol,
                                                           np.int32(projIdx),
                                                           np.int32(
                                                               self._volshape[2]),
                                                           np.int32(
                                                               self._volshape[1]),
                                                           np.int32(
                                                               self._volshape[0]),
                                                           np.float32(
                                                               -self._origin[0]),
                                                           np.float32(
                                                               -self._origin[1]),
                                                           np.float32(
                                                               -self._origin[2]),
                                                           np.float32(
                                                               self._voxelSize[0]),
                                                           np.float32(
                                                               self._voxelSize[1]),
                                                           np.float32(
                                                               self._voxelSize[2]),
                                                           np.float32(
                                                               self._normalizer),
                                                           grid=grid, block=block
                                                           )

        else:
            self._kernel_backprojection(vol_gpu.gpudata,
                                        self._proj_mats_gpu,
                                        np.int32(projIdx),
                                        np.int32(self._volshape[2]),
                                        np.int32(self._volshape[1]),
                                        np.int32(self._volshape[0]),
                                        np.float32(-self._origin[0]),
                                        np.float32(-self._origin[1]),
                                        np.float32(-self._origin[2]),
                                        np.float32(self._voxelSize[0]),
                                        np.float32(self._voxelSize[1]),
                                        np.float32(self._voxelSize[2]),
                                        np.float32(self._normalizer),
                                        grid=grid, block=block
                                        )

    def backProjectPixelDrivenCuda(self, sino: np.ndarray, vol=None, multiplicative=False, static_vol_gpu=None):
        assert sino.dtype == np.float32

        self._check_config()

        if not vol:
            vol = np.zeros(self._volshape, np.float32)

        if isinstance(vol, gpuarray.GPUArray):
            vol_gpu = vol
        else:
            vol_gpu = cuda.mem_alloc(vol.nbytes)

        if isinstance(vol, np.ndarray) and vol.ndim == 3:
            cuda.memcpy_htod(vol_gpu, vol)

        for t in range(self.start_idx, self.stop_idx):
            if isinstance(vol, list) or vol.ndim == 4:
                cuda.memcpy_htod(vol_gpu, vol[t])

            ndarray_to_float_tex(self._tex_backprojection, sino[t])

            block = (32, 8, 1)
            grid = (int(divup(self._volshape[2], block[0])),
                    int(divup(self._volshape[1], block[1])), 1)

            if multiplicative:
                self._kernel_backprojection_multiplicative(vol_gpu,
                                                           self._proj_mats_gpu,
                                                           static_vol_gpu,
                                                           np.int32(t),
                                                           np.int32(
                                                               self._volshape[2]),
                                                           np.int32(
                                                               self._volshape[1]),
                                                           np.int32(
                                                               self._volshape[0]),
                                                           np.float32(
                                                               -self._origin[0]),
                                                           np.float32(
                                                               -self._origin[1]),
                                                           np.float32(
                                                               -self._origin[2]),
                                                           np.float32(
                                                               self._voxelSize[0]),
                                                           np.float32(
                                                               self._voxelSize[1]),
                                                           np.float32(
                                                               self._voxelSize[2]),
                                                           np.float32(
                                                               self._normalizer),
                                                           grid=grid, block=block
                                                           )
            else:
                self._kernel_backprojection(vol_gpu,
                                            self._proj_mats_gpu,
                                            np.int32(t),
                                            np.int32(self._volshape[2]),
                                            np.int32(self._volshape[1]),
                                            np.int32(self._volshape[0]),
                                            np.float32(-self._origin[0]),
                                            np.float32(-self._origin[1]),
                                            np.float32(-self._origin[2]),
                                            np.float32(self._voxelSize[0]),
                                            np.float32(self._voxelSize[1]),
                                            np.float32(self._voxelSize[2]),
                                            np.float32(self._normalizer),
                                            grid=grid, block=block
                                            )

        if not isinstance(vol, gpuarray.GPUArray):
            cuda.memcpy_dtoh(vol, vol_gpu)

        return vol

    def _check_config(self):

        if not self.is_configured:
            self._init_config()
            return

        if self._origin != pyconrad.config.get_reco_origin():
            self._init_config()
            return
        if self._voxelSize != pyconrad.config.get_reco_spacing():
            self._init_config()
            return


def write_projections(vols: list, filename, proj_idx, do_compression=False):

    projector = CudaProjector()  # type: CudaProjector

    sino_gpu = gpuarray.GPUArray(
        pyconrad.config.get_sino_shape()[1:], np.float32)
    imlist = []

    for vol, idx in zip(vols, range(len(vols))):
        projector.forward_project_cuda_idx(vol, sino_gpu, proj_idx)
        imlist.append(Image.fromarray(sino_gpu.get()))

    if do_compression:
        imlist[0].save(filename, save_all=True, compression='tiff_deflate',
                       append_images=imlist[1:])
    else:
        imlist[0].save(filename, save_all=True,
                       append_images=imlist[1:])
    print('Wrote file ' + filename)
