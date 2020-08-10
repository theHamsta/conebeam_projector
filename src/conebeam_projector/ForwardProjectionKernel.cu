// from ConeBeamBackProjector.cl of CONRAD
#include <pycuda-helpers.hpp>

// TODO make sino 3d tex
texture<fp_tex_float, cudaTextureType3D, cudaReadModeElementType> gTex3D;

typedef float TvoxelValue;
typedef float Tcoord_dev;
typedef float TdetValue;

/* --------------------------------------------------------------------------
 *
 *
 *    Ray-tracing algorithm implementation in CUDA kernel programming
 *
 *
 * -------------------------------------------------------------------------- */

__device__ float
project_ray(float sx, float sy, float sz, // X-ray source position
            float rx, float ry, float rz, // Ray direction
            float stepsize, // ALPHA_STEP_SIZE Step size in ray direction
            float gVolumeEdgeMinPoint0, float gVolumeEdgeMinPoint1,
            float gVolumeEdgeMinPoint2, float gVolumeEdgeMaxPoint0,
            float gVolumeEdgeMaxPoint1, float gVolumeEdgeMaxPoint2) {

  gVolumeEdgeMinPoint0 += 0.5;
  gVolumeEdgeMinPoint1 += 0.5;
  gVolumeEdgeMinPoint2 += 0.5;

  gVolumeEdgeMaxPoint0 += 0.5;
  gVolumeEdgeMaxPoint1 += 0.5;
  gVolumeEdgeMaxPoint2 += 0.5;

  // Step 1: compute alpha value at entry and exit point of the volume
  float minAlpha, maxAlpha;
  minAlpha = 0;
  maxAlpha = INFINITY;

  if (0.0f != rx) {
    float reci = 1.0f / rx;
    float alpha0 = (gVolumeEdgeMinPoint0 - sx) * reci;
    float alpha1 = (gVolumeEdgeMaxPoint0 - sx) * reci;
    minAlpha = fmin(alpha0, alpha1);
    maxAlpha = fmax(alpha0, alpha1);
  }

  if (0.0f != ry) {
    float reci = 1.0f / ry;
    float alpha0 = (gVolumeEdgeMinPoint1 - sy) * reci;
    float alpha1 = (gVolumeEdgeMaxPoint1 - sy) * reci;
    minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
    maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
  }

  if (0.0f != rz) {
    float reci = 1.0f / rz;
    float alpha0 = (gVolumeEdgeMinPoint2 - sz) * reci;
    float alpha1 = (gVolumeEdgeMaxPoint2 - sz) * reci;
    minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
    maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
  }

  // we start not at the exact entry point
  // => we can be sure to be inside the volume
  // minAlpha += stepsize * 0.5f;

  // Step 2: Cast ray if it intersects the volume
  float pixel = 0.0f;

  // Trapezoidal rule (interpolating function = piecewise linear func)
  float px, py, pz;

  // Entrance boundary
  // In CUDA, voxel centers are located at (xx.5, xx.5, xx.5),
  //  whereas, SwVolume has voxel centers at integers.
  // For the initial interpolated value, only a half stepsize is
  //  considered in the computation.
  if (minAlpha < maxAlpha) {
    px = sx + minAlpha * rx;
    py = sy + minAlpha * ry;
    pz = sz + minAlpha * rz;
   pixel += 0.5 * tex3D(gTex3D, px + 0.5, py + 0.5f, pz + 0.5);
    // read_imagef(gTex3D, sampler, (float4)(px + 0.5f,
    // py + 0.5f, pz - gVolumeEdgeMinPoint[2],0)).x;
    minAlpha += stepsize;
  }

  // Mid segments
  while (minAlpha < maxAlpha) {
    px = sx + minAlpha * rx;
    py = sy + minAlpha * ry;
    pz = sz + minAlpha * rz;
    // if (calcVesselIntegral)
    // {
    //     assert(false && "not implemented yet");
    // }
    // else
    // {
    // pixel += tex3D<float>(gTex3D, px , py , pz - gVolumeEdgeMinPoint2);
    pixel += tex3D(gTex3D, px + 0.5f, py + 0.5f, pz + 0.5f);
    // }
    minAlpha += stepsize;
  }

  // Scaling by stepsize;
  pixel *= stepsize;

  // Last segment of the line
  if (pixel > 0.0f) {
    pixel -= 0.5 * stepsize * tex3D(gTex3D, px + 0.5f, py + 0.5f, pz + 0.5f);
    minAlpha -= stepsize;
    float lastStepsize = maxAlpha - minAlpha;
    pixel +=
        0.5 * lastStepsize * tex3D(gTex3D, px + 0.5f, py + 0.5f, pz + 0.5f);

    px = sx + maxAlpha * rx;
    py = sy + maxAlpha * ry;
    pz = sz + maxAlpha * rz;
    // The last segment of the line integral takes care of the
    // varying length.
    pixel +=
        0.5 * lastStepsize * tex3D(gTex3D, px + 0.5f, py + 0.5f, pz + 0.5f);
  }

  // -------------------------------------------------------------------

  return pixel;
} // ProjectRay

__device__ float project_ray_maximum_intensity(
    float sx, float sy, float sz, // X-ray source position
    float rx, float ry, float rz, // Ray direction
    float stepsize,               // ALPHA_STEP_SIZE Step size in ray direction
    float gVolumeEdgeMinPoint0, float gVolumeEdgeMinPoint1,
    float gVolumeEdgeMinPoint2, float gVolumeEdgeMaxPoint0,
    float gVolumeEdgeMaxPoint1, float gVolumeEdgeMaxPoint2) {

  gVolumeEdgeMinPoint0 += 0.5;
  gVolumeEdgeMinPoint1 += 0.5;
  gVolumeEdgeMinPoint2 += 0.5;

  gVolumeEdgeMaxPoint0 += 0.5;
  gVolumeEdgeMaxPoint1 += 0.5;
  gVolumeEdgeMaxPoint2 += 0.5;


  // Step 1: compute alpha value at entry and exit point of the volume
  float minAlpha, maxAlpha;
  minAlpha = 0;
  maxAlpha = INFINITY;

  if (0.0f != rx) {
    float reci = 1.0f / rx;
    float alpha0 = (gVolumeEdgeMinPoint0 - sx - 0.5f) * reci;
    float alpha1 = (gVolumeEdgeMaxPoint0 - sx - 0.5f) * reci;
    minAlpha = fmin(alpha0, alpha1);
    maxAlpha = fmax(alpha0, alpha1);
  }

  if (0.0f != ry) {
    float reci = 1.0f / ry;
    float alpha0 = (gVolumeEdgeMinPoint1 - sy - 0.5f) * reci;
    float alpha1 = (gVolumeEdgeMaxPoint1 - sy - 0.5f) * reci;
    minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
    maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
  }

  if (0.0f != rz) {
    float reci = 1.0f / rz;
    float alpha0 = (gVolumeEdgeMinPoint2 - sz - 0.5f) * reci;
    float alpha1 = (gVolumeEdgeMaxPoint2 - sz - 0.5f) * reci;
    minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
    maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
  }

  // we start not at the exact entry point
  // => we can be sure to be inside the volume
  // minAlpha += stepsize * 0.5f;

  // Step 2: Cast ray if it intersects the volume
  float pixel = 0.0f;

  // Trapezoidal rule (interpolating function = piecewise linear func)
  float px, py, pz;

  // Entrance boundary
  // In CUDA, voxel centers are located at (xx.5, xx.5, xx.5),
  //  whereas, SwVolume has voxel centers at integers.
  // For the initial interpolated value, only a half stepsize is
  //  considered in the computation.
  if (minAlpha < maxAlpha) {
    px = sx + minAlpha * rx;
    py = sy + minAlpha * ry;
    pz = sz + minAlpha * rz;
    pixel = tex3D(gTex3D, px + 0.5, py + 0.5f,
                  pz + 0.5); // read_imagef(gTex3D, sampler, (float4)(px + 0.5f,
                             // py + 0.5f, pz - gVolumeEdgeMinPoint[2],0)).x;
    minAlpha += stepsize;
  }

  // Mid segments
  while (minAlpha < maxAlpha) {
    px = sx + minAlpha * rx;
    py = sy + minAlpha * ry;
    pz = sz + minAlpha * rz;
    // if (calcVesselIntegral)
    // {
    //     assert(false && "not implemented yet");
    // }
    // else
    // {
    // pixel += tex3D<float>(gTex3D, px , py , pz - gVolumeEdgeMinPoint2);
    pixel = fmax(tex3D(gTex3D, px + 0.5f, py + 0.5f, pz + 0.5f), pixel);
    // }
    minAlpha += stepsize;
  }

  // Scaling by stepsize;
  // pixel *= stepsize;

  // Last segment of the line
  if (pixel > 0.0f) {
    pixel -= tex3D(gTex3D, px + 0.5f, py + 0.5f, pz + 0.5f - 0.5f);
    minAlpha -= stepsize;
    float lastStepsize = maxAlpha - minAlpha;
    pixel +=
        0.5 * lastStepsize * tex3D(gTex3D, px, py, pz - gVolumeEdgeMinPoint2);

    px = sx + maxAlpha * rx;
    py = sy + maxAlpha * ry;
    pz = sz + maxAlpha * rz;
    // The last segment of the line integral takes care of the
    // varying length.
    pixel +=
        0.5 * lastStepsize *
        tex3D(gTex3D, px + 0.5f, py + 0.5f, pz - gVolumeEdgeMinPoint2 - 0.5f);
  }

  // -------------------------------------------------------------------

  return pixel;
} // ProjectRay

__global__ void
forwardProjectionKernel(TdetValue *pProjection, int projWidth, int projHeight,
                        float stepsize,
                        // System geometry (for FP) parameters
                        float voxelSize0, float voxelSize1, float voxelSize2,
                        float gVolumeEdgeMinPoint0, float gVolumeEdgeMinPoint1,
                        float gVolumeEdgeMinPoint2, float gVolumeEdgeMaxPoint0,
                        float gVolumeEdgeMaxPoint1, float gVolumeEdgeMaxPoint2,
                        float gSrcPoint0, float gSrcPoint1, float gSrcPoint2,
                        //   Tcoord_dev *gVoxelElementSize,
                        //   Tcoord_dev *gVolumeEdgeMinPoint,
                        //   Tcoord_dev *gVolumeEdgeMaxPoint,
                        //   Tcoord_dev *gSrcPoint,
                        Tcoord_dev *gInvARmatrix, int projectionNumber,
                        int useMaximumIntensityProjection, bool additive) {
  int gidx = blockIdx.x;
  int gidy = blockIdx.y;
  int lidx = threadIdx.x;
  int lidy = threadIdx.y;

  int locSizex = blockDim.x;
  int locSizey = blockDim.y;

  int udx = gidx * locSizex + lidx;
  int vdx = gidy * locSizey + lidy;

  if (udx >= projWidth || vdx >= projHeight) {
    return;
  }

  int idx = vdx * projWidth + udx;

  float u = udx;
  float v = vdx;

  // pProjection += projectionNumber * (projWidth * projHeight);
  // TODO
  // gSrcPoint += projectionNumber * 3;
  gInvARmatrix += projectionNumber * 9;

  // compute ray direction
  float rx = gInvARmatrix[2] + v * gInvARmatrix[1] + u * gInvARmatrix[0];
  float ry = gInvARmatrix[5] + v * gInvARmatrix[4] + u * gInvARmatrix[3];
  float rz = gInvARmatrix[8] + v * gInvARmatrix[7] + u * gInvARmatrix[6];

  // normalize ray direction
  float normFactor = 1.0f / (sqrt((rx * rx) + (ry * ry) + (rz * rz)));
  rx *= normFactor;
  ry *= normFactor;
  rz *= normFactor;

  // compute forward projection
  float pixel = [&]() {
    if (!useMaximumIntensityProjection) {
      float pixel = project_ray(
          gSrcPoint0, gSrcPoint1, gSrcPoint2, rx, ry, rz, stepsize,
          gVolumeEdgeMinPoint0, gVolumeEdgeMinPoint1, gVolumeEdgeMinPoint2,
          gVolumeEdgeMaxPoint0, gVolumeEdgeMaxPoint1, gVolumeEdgeMaxPoint2);
      // normalize pixel value to world coordinate system units
      return pixel * sqrtf((rx * voxelSize0) * (rx * voxelSize0) +
                           (ry * voxelSize1) * (ry * voxelSize1) +
                           (rz * voxelSize2) * (rz * voxelSize2));

    } else {
      return project_ray_maximum_intensity(
          gSrcPoint0, gSrcPoint1, gSrcPoint2, rx, ry, rz, stepsize,
          gVolumeEdgeMinPoint0, gVolumeEdgeMinPoint1, gVolumeEdgeMinPoint2,
          gVolumeEdgeMaxPoint0, gVolumeEdgeMaxPoint1, gVolumeEdgeMaxPoint2);
    }
  }();

  if (additive) {
    pProjection[idx] += pixel;
  } else {
    pProjection[idx] = pixel;
  }

  return;
};
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public
 * License (GPL).
 */
