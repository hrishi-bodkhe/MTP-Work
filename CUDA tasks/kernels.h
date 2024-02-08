#ifndef KERNELS_H
#define KERNELS_H
#include "preprocessing.h"

__global__ void divideTCbysix(float *tc);

__global__ void triangleCountVertexCentric(ll *csr_offsets, ll *csr_edges, ll totalvertices, ll* tc);

__global__ void print2(int n, ll *arr);

__global__ void print(float *n, ll *arr);

__global__ void findJustSmallest(ll *arr, ll size, ll target, ll *ans);

__global__ void replaceNodeWithDegree(ll *csr_offsets, ll *input_frontier, ll *deg_for_input_frontier, ll size);

__global__ void constructFrontierOffset(ll *csr_offsets, ll *input_frontier, ll *frontier_offset, ll size, ll *prefixSum);

__global__ void ssspBucketWorklistKernel(ll workers, ll *csr_offsets, ll *csr_edges, ll *csr_weights, ll *curr, ll *next1, ll *next2, ll *dist, float *idx1, float *idx2);

__global__ void ssspBucketWorklistKernel2(ll workers, ll *csr_offsets, ll *csr_edges, ll *csr_weights, ll *curr, ll *next1, ll *next2, ll *next3, ll *dist, float *idx1, float *idx2, float *idx3);

__global__ void ssspEdgeWorklist(ll *csr_offsets, ll *csr_edges, ll *csr_weights, ll *input_frontier, ll *frontier_offset, ll *output_frontier, ll *prefixSum, ll *dist, float *idx, ll size);

__global__ void ssspBalancedWorklistKernel(ll totalvertices, ll workers, ll *dindex, ll *dheadvertex, ll *dweights, ll *curr, ll *next1, ll *next2, ll *dist, float *idx1, float *idx2, ll limit);

__global__ void ssspWorklistKernel2(ll workers, ll *dindex, ll *dheadvertex, ll *dweights, ll *curr, ll *next1, ll *next2, ll *dist, float *idx1, float *idx2);

__global__ void mergeWorklist(ll *curr, ll *next1, ll *next2, float *idx1, float *idx2);

__global__ void ssspWorklistKernel(ll workers, ll *dindex, ll *dheadvertex, ll *dweights, ll *curr, ll *next, ll *dist, float *idx, ll limit);

__global__ void setIndexForWorklist(float *idx);

__global__ void setIndexForWorklist2(float *idx1, float *idx2);

__global__ void setIndexForWorklist(float *idx1, float *idx2, float *idx3);

__global__ void ssspEdgeCall(ll totalEdges, ll *dsrc, ll *ddest, ll *dweights, ll *dist, int *ddchanged);

__global__ void ssspVertexCall(ll vertices, ll *dindex, ll *dheadVertex, ll *dweights, ll *dist, int *ddchanged);

__global__ void ssspVertexInit(ll vertices, ll *dist);

__global__ void initSrc(ll src, ll *dist);

__global__ void init(ll src, ll *dist, ll *curr);

__global__ void printDist(ll vertices, ll *dist);

__global__ void printCSRKernel(ll vertices, ll *index);

__global__ void checkCorrectness(ll totalVertices, ll *vdist, ll *wdist, int *equalityFlag);

#endif