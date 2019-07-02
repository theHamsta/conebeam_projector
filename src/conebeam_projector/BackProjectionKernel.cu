// from ConeBeamBackProjector.cl of CONRAD
#include <pycuda-helpers.hpp>

// TODO make sino 3d tex
texture<fp_tex_float, cudaTextureType2D, cudaReadModeElementType> tex_sino;

// TODO make gProjMatrix __constant__

__device__ inline int getImageIdx(int x, int y, int z, int imgWidth, int imgHeight)
{
	return z * imgHeight * imgWidth + y * imgWidth + x;
}

__global__ void backProjectionKernel(
	float *imgGrid,
	float *gProjMatrix,
	int projIdx,
	int imgSizeX,
	int imgSizeY,
	int imgSizeZ,
	float originX,
	float originY,
	float originZ,
	float spacingX,
	float spacingY,
	float spacingZ,
	float normalizer
	//float spacingU,
	//float spacingV,

)
{

	int gidx = blockIdx.x;
	int gidy = blockIdx.y;
	int lidx = threadIdx.x;
	int lidy = threadIdx.y;

	int locSizex = blockDim.x;
	int locSizey = blockDim.y;

	int x = gidx * locSizex + lidx;
	int y = gidy * locSizey + lidy;

	// check if inside image boundaries
	if (x >= imgSizeX || y >= imgSizeY)
		return;

	float4 pos = {(float)x * spacingX - originX, (float)y * spacingY - originY, 0.0f, 1.0f};
	float precomputeR = gProjMatrix[projIdx * 12 + 3] + pos.y * gProjMatrix[projIdx * 12 + 1] + pos.x * gProjMatrix[projIdx * 12 + 0];
	float precomputeS = gProjMatrix[projIdx * 12 + 7] + pos.y * gProjMatrix[projIdx * 12 + 5] + pos.x * gProjMatrix[projIdx * 12 + 4];
	float precomputeT = gProjMatrix[projIdx * 12 + 11] + pos.y * gProjMatrix[projIdx * 12 + 9] + pos.x * gProjMatrix[projIdx * 12 + 8];

	int sizeXY = imgSizeX * imgSizeY;

	for (int z = 0; z < imgSizeZ; ++z)
	{

		pos.z = ((float)z * spacingZ) - originZ;
		float r = pos.z * gProjMatrix[projIdx * 12 + 2] + precomputeR;
		float s = pos.z * gProjMatrix[projIdx * 12 + 6] + precomputeS;
		float t = pos.z * gProjMatrix[projIdx * 12 + 10] + precomputeT;

		// compute projection coordinates
		float denom = 1.0f / t;
		float fu = r * denom;
		float fv = s * denom;
		float u = fu + 0.5f;						   // + 0.5f;
		float v = fv + 0.5f;						   // + 0.5f;
		float proj_val = tex2D<float>(tex_sino, u, v); // <--- correct for non-padded detector

		// compute volume index for x,y,z
		int idx = z * sizeXY + (y * imgSizeX + x);
		imgGrid[idx] += proj_val * denom * denom * normalizer;
		// imgGrid[idx] = tex2D<float>(tex_sino, ((float) x) , ((float) y) );
	}
}

__global__ void backProjectionKernelWithConstrainingVolume(
	float *imgGrid,
	float *gProjMatrix,
	float *vesselMask,
	int projIdx,
	int imgSizeX,
	int imgSizeY,
	int imgSizeZ,
	float originX,
	float originY,
	float originZ,
	float spacingX,
	float spacingY,
	float spacingZ,
	float normalizer
	//float spacingU,
	//float spacingV,

)
{

	int gidx = blockIdx.x;
	int gidy = blockIdx.y;
	int lidx = threadIdx.x;
	int lidy = threadIdx.y;

	int locSizex = blockDim.x;
	int locSizey = blockDim.y;

	int x = gidx * locSizex + lidx;
	int y = gidy * locSizey + lidy;

	// check if inside image boundaries
	if (x >= imgSizeX || y >= imgSizeY)
		return;

	float4 pos = {(float)x * spacingX - originX, (float)y * spacingY - originY, 0.0f, 1.0f};
	float precomputeR = gProjMatrix[projIdx * 12 + 3] + pos.y * gProjMatrix[projIdx * 12 + 1] + pos.x * gProjMatrix[projIdx * 12 + 0];
	float precomputeS = gProjMatrix[projIdx * 12 + 7] + pos.y * gProjMatrix[projIdx * 12 + 5] + pos.x * gProjMatrix[projIdx * 12 + 4];
	float precomputeT = gProjMatrix[projIdx * 12 + 11] + pos.y * gProjMatrix[projIdx * 12 + 9] + pos.x * gProjMatrix[projIdx * 12 + 8];

	int sizeXY = imgSizeX * imgSizeY;

	for (int z = 0; z < imgSizeZ; ++z)
	{

		pos.z = ((float)z * spacingZ) - originZ;
		float r = pos.z * gProjMatrix[projIdx * 12 + 2] + precomputeR;
		float s = pos.z * gProjMatrix[projIdx * 12 + 6] + precomputeS;
		float t = pos.z * gProjMatrix[projIdx * 12 + 10] + precomputeT;

		// compute projection coordinates
		float denom = 1.0f / t;
		float fu = r * denom;
		float fv = s * denom;
		float u = fu + 0.5f;						   // + 0.5f;
		float v = fv + 0.5f;						   // + 0.5f;
		float proj_val = tex2D<float>(tex_sino, u, v); // <--- correct for non-padded detector

		// compute volume index for x,y,z
		int idx = z * sizeXY + (y * imgSizeX + x);

		if (vesselMask[idx] > 1.e-3f)
		{
			if (normalizer == 0.f){
				imgGrid[idx] += proj_val;
			} else {
				imgGrid[idx] += proj_val * denom * denom * normalizer;
			}
		}
		// imgGrid[idx] = tex2D<float>(tex_sino, ((float) x) , ((float) y) );
	}
}

__global__ void backprojectMultiplicative(
	float *imgGrid,
	float *__restrict__ gProjMatrix,
	float *staticVol,
	int projIdx,
	int imgSizeX,
	int imgSizeY,
	int imgSizeZ,
	float originX,
	float originY,
	float originZ,
	float spacingX,
	float spacingY,
	float spacingZ,
	float normalizer
	//float spacingU,
	//float spacingV,

)
{

	int gidx = blockIdx.x;
	int gidy = blockIdx.y;
	int lidx = threadIdx.x;
	int lidy = threadIdx.y;

	int locSizex = blockDim.x;
	int locSizey = blockDim.y;

	int x = gidx * locSizex + lidx;
	int y = gidy * locSizey + lidy;

	// check if inside image boundaries
	if (x >= imgSizeX || y >= imgSizeY)
		return;

	float4 pos = {(float)x * spacingX - originX, (float)y * spacingY - originY, 0.0f, 1.0f};
	float precomputeR = gProjMatrix[projIdx * 12 + 3] + pos.y * gProjMatrix[projIdx * 12 + 1] + pos.x * gProjMatrix[projIdx * 12 + 0];
	float precomputeS = gProjMatrix[projIdx * 12 + 7] + pos.y * gProjMatrix[projIdx * 12 + 5] + pos.x * gProjMatrix[projIdx * 12 + 4];
	float precomputeT = gProjMatrix[projIdx * 12 + 11] + pos.y * gProjMatrix[projIdx * 12 + 9] + pos.x * gProjMatrix[projIdx * 12 + 8];

	for (int z = 0; z < imgSizeZ; ++z)
	{

		pos.z = ((float)z * spacingZ) - originZ;
		float r = pos.z * gProjMatrix[projIdx * 12 + 2] + precomputeR;
		float s = pos.z * gProjMatrix[projIdx * 12 + 6] + precomputeS;
		float t = pos.z * gProjMatrix[projIdx * 12 + 10] + precomputeT;

		// compute projection coordinates
		float denom = 1.0f / t;
		float fu = r * denom;
		float fv = s * denom;
		float u = fu + 0.5f;						   // + 0.5f;
		float v = fv + 0.5f;						   // + 0.5f;
		float proj_val = tex2D<float>(tex_sino, u, v); // <--- correct for non-padded detector

		// compute volume index for x,y,z
		int idx = getImageIdx(x, y, z, imgSizeX, imgSizeY);
		imgGrid[idx] += proj_val * denom * denom * normalizer * staticVol[idx];
		// imgGrid[idx] = tex2D<float>(tex_sino, ((float) x) , ((float) y) );
	}
}
