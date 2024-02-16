#include<cuda_runtime.h>
#include "kernels.h"

__global__ void divideTCArray(unsigned int *tc, unsigned int val, ll totalvertices){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= totalvertices) return;

    tc[id] = tc[id] / val;
}

__global__ void triangleCountSortedVertexCentricKernel(ll *csr_offsets, ll *csr_edges, unsigned int *tc, ll totalvertices){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= totalvertices) return;

    ll src_vertex = id;

    ll neighbor_start_for_src = csr_offsets[src_vertex];
    ll neighbor_end_for_src = csr_offsets[src_vertex + 1];

    for(int k = neighbor_start_for_src; k < neighbor_end_for_src; ++k){
        ll dest_vertex = csr_edges[k];
        ll neighbor_start_for_dest = csr_offsets[dest_vertex];
        ll neighbor_end_for_dest = csr_offsets[dest_vertex + 1];

        ll i = neighbor_start_for_src;
        ll j = neighbor_start_for_dest;

        unsigned int count = 0;

        while(i < neighbor_end_for_src && j < neighbor_end_for_dest){
            ll diff = csr_edges[i] - csr_edges[j];

            if(diff == 0){
                ++count;
                ++i;
                ++j;
            }
            else if(diff < 0) ++i;
            else ++j;
        }

        tc[src_vertex] += count;
    }

    tc[src_vertex] /= 2;
}

__global__ void triangleCountEdgeCentricKernel(ll *csr_offsets, ll *csr_edges, unsigned int *tc, ll totaledges, ll totalvertices){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= totaledges) return;

    ll dest_vertex = csr_edges[id];

    ll src_vertex = -1;
    // Applying Binary Search to find the source offset for the respective thread.
    ll start = 0;
    ll end = totalvertices - 1;

    while(start <= end){
        ll mid = start + (end - start) / 2;
        if(csr_offsets[mid] == id){
            src_vertex = mid;
            break;
        }
        else if(csr_offsets[mid] < id) {
            src_vertex = mid;
            start = mid + 1;
        }
        else {
            end = mid - 1;
        }
    }

    // Find common nodes between src and dest

    ll neighbor_start_for_src = csr_offsets[src_vertex];
    ll neighbor_end_for_src = csr_offsets[src_vertex + 1];
    ll neighbor_start_for_dest = csr_offsets[dest_vertex];
    ll neighbor_end_for_dest = csr_offsets[dest_vertex + 1];

    ll i = neighbor_start_for_src;
    ll j = neighbor_start_for_dest;

    unsigned int count = 0;

    while(i < neighbor_end_for_src && j < neighbor_end_for_dest){
        ll diff = csr_edges[i] - csr_edges[j];

        if(diff == 0){
            ++count;
            ++i;
            ++j;
        }
        else if(diff < 0) ++i;
        else ++j;
    }

    atomicAdd(&tc[src_vertex], count);
}

__global__ void divideTCbysix(float *tc){
    *tc = *tc / 6;
}

__global__ void triangleCountVertexCentric(ll *csr_offsets, ll *csr_edges, ll totalvertices, ll *tc){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= totalvertices) return;

    ll vertex_p = id;
    ll start_p = csr_offsets[vertex_p];
    ll end_p = csr_offsets[vertex_p + 1];

    for(ll s = start_p; s < end_p; ++s){
        ll vertex_t = csr_edges[s];

        for(ll i = s + 1; i < end_p; ++i){
            ll vertex_r = csr_edges[i];

            if(vertex_t != vertex_r){
                ll start_r = csr_offsets[vertex_r];
                ll end_r = csr_offsets[vertex_r + 1];

                for(int j = start_r; j < end_r; ++j){
                    if(csr_edges[j] == vertex_t){
                        tc[vertex_p] += 1;
                        break;
                    }
                }
            }
        }
    }

//    tc[vertex_p] /= 2;
}

__global__ void ssspBucketWorklistKernel(ll workers, ll *csr_offsets, ll *csr_edges, ll *csr_weights, ll *curr, ll *next1, ll *next2, ll *dist, float *idx1, float *idx2){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= workers) return;

    ll u = curr[id];

    ll start = csr_offsets[u];
    ll end = csr_offsets[u + 1];

    for(ll i = start; i < end; ++i){
        ll v = csr_edges[i];
        ll wt = csr_weights[i];

        if(dist[v] > dist[u] + wt){
            atomicMin(&dist[v], dist[u] + wt);
            ll deg = csr_offsets[v + 1] - csr_offsets[v];
            ll index;

            if(deg <= 16) {
                index = atomicAdd(idx1, 1);
                next1[index] = v;
            }
            else if(deg >= 16){
                index = atomicAdd(idx2, 1);
                next2[index] = v;
            }
        }
    }
}

__global__ void ssspBucketWorklistKernel2(ll workers, ll *csr_offsets, ll *csr_edges, ll *csr_weights, ll *curr, ll *next1, ll *next2, ll *next3, ll *dist, float *idx1, float *idx2, float *idx3){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= workers) return;

    ll u = curr[id];

    ll start = csr_offsets[u];
    ll end = csr_offsets[u + 1];

    for(ll i = start; i < end; ++i){
        ll v = csr_edges[i];
        ll wt = csr_weights[i];
        ll deg = csr_offsets[v + 1] - csr_offsets[v];

        if(dist[v] > dist[u] + wt){
            atomicMin(&dist[v], dist[u] + wt);

            ll index;

            if(deg <= 32) {
                index = atomicAdd(idx1, 1);
                next1[index] = v;
            }
            else if(deg > 32 && deg <= 256){
                index = atomicAdd(idx2, 1);
                next2[index] = v;
            }
            else{
                index = atomicAdd(idx3, 1);
                next3[index] = v;
            }
        }
    }
}

__global__ void replaceNodeWithDegree(ll *csr_offsets, ll *input_frontier, ll *deg_for_input_frontier, ll size){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= size) return;

    ll node = input_frontier[id];
    deg_for_input_frontier[id] = csr_offsets[node + 1] - csr_offsets[node];
}

__global__ void ssspEdgeWorklist(ll *csr_offsets, ll *csr_edges, ll *csr_weights, ll *input_frontier, ll *frontier_offset, ll *output_frontier, ll *prefixSum, ll *dist, float *idx, ll size){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= *prefixSum) return;

//    for(ll i = 0; i <= size; ++i){
//        printf("i : %lld, element: %lld\n", i, frontier_offset[i]);
//    }
    ll src_offset = -1;
    // Applying Binary Search to find the source offset for the respective thread.
    ll start = 0;
    ll end = size;
//    printf("start: %lld, end: %lld\n", start, end);

    while(start <= end){
        ll mid = start + (end - start) / 2;
        if(frontier_offset[mid] == id){
            src_offset = mid;
            break;
        }
        else if(frontier_offset[mid] < id) {
            src_offset = mid;
            start = mid + 1;
        }
        else {
            end = mid - 1;
        }
    }

//    printf("Thread id: %d, Source offset: %lld\n",id, src_offset);

    ll edge_src = input_frontier[src_offset];                                   // Getting source node of the particular edge.

    ll edge_offset = csr_offsets[edge_src] + id - frontier_offset[src_offset];    // Getting offset of destination node of the particular edge.

    if(dist[csr_edges[edge_offset]] > dist[edge_src] + csr_weights[edge_offset]){
        atomicMin(&dist[csr_edges[edge_offset]], dist[edge_src] + csr_weights[edge_offset]);
        ll index = atomicAdd(idx , 1);
        output_frontier[index] = csr_edges[edge_offset];
    }
}

__global__ void constructFrontierOffset(ll *csr_offsets, ll *input_frontier, ll *frontier_offset, ll size, ll *prefixSum){
    /** Computes the prefix sum for each element of input frontier and stores it in frontier offset. **/
//    printf("size: %lld\n", size);
////    frontier_offset[0] = 0;

//    for(ll i = 0; i < size; ++i){
//        printf("i : %lld, element: %lld\n", i, input_frontier[i]);
//    }

//    printf("\n");
////
//    for(ll i = 1; i <= size; ++i){
//        ll node = input_frontier[i - 1];                        // Getting the respective vertex;
//        ll deg = csr_offsets[node + 1] - csr_offsets[node];     // Calculating degree of the vertex from csr offsets
//
//        frontier_offset[i] = deg + frontier_offset[i - 1];  // Storing the sum
//    }

    *prefixSum = frontier_offset[size];
//    printf("%lld\n",*prefixSum);

//    printf("\n Printing frontier offset:\n");
//    for(ll i = 0; i <= size; ++i){
//        printf("i: %lld, element: %lld\n", i, frontier_offset[i]);
//    }
//    printf("Finished frontier offset construction.\n");
}

__global__ void findJustSmallest(ll *arr, ll size, ll target, ll *ans){
    /*** Returns the index of the target element, if present, else returns the index of the just smaller element. ***/
    // Uses Binary Search
//    printf("Here\n");
    ll start = 0;
    ll end = size - 1;
    ll mid = start + (end - start) / 2;

    while(start < end){
//        printf("Here");
        if(arr[mid] == target){
            *ans = mid;
            return;
        }
        else if(arr[mid] > target) end = mid - 1;
        else start = mid + 1;
    }

    *ans = end;
}

__global__ void ssspBalancedWorklistKernel(ll totalvertices, ll workers, ll *dindex, ll *dheadvertex, ll *dweights, ll *curr, ll *next1, ll *next2, ll *dist, float *idx1, float *idx2, ll limit){
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= workers) return;

    ll u = curr[id];
    ll start = dindex[u];
    ll end = dindex[u + 1];

    for(ll i = start; i < end; ++i){
        ll v = dheadvertex[i];
        ll wt = dweights[i];

        if(dist[v] > dist[u] + wt){
            atomicMin(&dist[v], dist[u] + wt);

            if(id < limit) {
                ll index1 = atomicAdd(idx1, 1);
                next1[index1] = v;
//                assert(index1 < totalvertices && "Index out of range");
if(index1 >= totalvertices){
    printf("Index out of range");
    return;
}
//                printf("%ld ", next1[index1]);
            }
            else {
                ll index2 = atomicAdd(idx2, 1);
//                assert(index2 < totalvertices && "Index out of range");
if(index2 >= totalvertices){
    printf("Index out of range");
    return;
}
                next2[index2] = v;
//                printf("%ld ", next2[index2]);
            }
        }
    }
}

__global__ void checkCorrectness(ll totalVertices, unsigned int *vdist, unsigned int *wdist, int *equalityFlag){
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= totalVertices) return;

    if(vdist[id] != wdist[id]) {
        *equalityFlag = 0;
    }
}

__global__ void checkCorrectness(ll totalVertices, ll *vdist, ll *wdist, int *equalityFlag){
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= totalVertices) return;

    if(vdist[id] != wdist[id]) {
        *equalityFlag = 0;
    }
}

__global__ void print2(int n, ll *arr){
    for(int i = 0; i < n; ++i) printf("%ld ", arr[i]);
    printf("\n");
}

__global__ void print(float *n, ll *arr){
    for(int i = 0; i < int(*n); ++i) printf("%ld ", arr[i]);
    printf("\n");
}

__global__ void ssspWorklistKernel2(ll workers, ll *dindex, ll *dheadvertex, ll *dweights, ll *curr, ll *next1, ll *next2, ll *dist, float *idx1, float *idx2){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= workers) return;

    ll u = curr[id];
    ll start = dindex[u];
    ll end = dindex[u + 1];

    for(ll i = start; i < end; ++i){
        ll v = dheadvertex[i];
        ll wt = dweights[i];

        if(dist[v] > dist[u] + wt){
            atomicMin(&dist[v], dist[u] + wt);

            if(id % 2 == 0) {
                ll index1 = atomicAdd(idx1, 1);
                next1[index1] = v;
//                printf("%ld ", next1[index1]);
            }
            else {
                ll index2 = atomicAdd(idx2, 1);
                next2[index2] = v;
//                printf("%ld ", next2[index2]);
            }
        }
    }
}

__global__ void mergeWorklist(ll *curr, ll *next1, ll *next2, float *idx1, float *idx2){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    int n1 = *idx1;
    int n2 = *idx2;

//    printf("n1: %d\n", n1);
//    printf("n2: %d\n", n2);

    if(id >= (n1 + n2)) return;
//    printf("id: %d ", id);

    if(id < n1) curr[id] = next1[id];
    else curr[id] = next2[id - n1];
//    printf("fdfdf%ld ", curr[id]);
}

__global__ void ssspWorklistKernel(ll workers, ll *dindex, ll *dheadvertex, ll *dweights, ll *curr, ll *next, ll *dist, float *idx, ll limit){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= workers) return;

    ll u = curr[id];
//    printf("u: %ld, workers: %ld\n", u, workers);
    //check whether u is within range or not
//    if(u >= totalVertices) printf("Out of range u\n");
    ll start = dindex[u];
    ll end = dindex[u + 1];
//    printf("u: %ld ", u);



    for(ll i = start; i < end; ++i){
        ll v = dheadvertex[i];
        ll wt = dweights[i];

        if(dist[v] > dist[u] + wt){
            atomicMin(&dist[v], dist[u] + wt);
            ll index = atomicAdd(idx, 1);
            //check whether index is within range or not
//            if(index >= limit) {
//                printf("%ld. Index is out of range.\n", index);
//                return;
//            }
//            printf("index: %ld ", index);
            next[index] = v;
        }
    }
}

__global__ void setIndexForWorklist(float *idx){
//    printf("Entered\n");
    *idx = 0;
//    printf("Leaving\n");
    return;
}

__global__ void setIndexForWorklist2(float *idx1, float *idx2){
//    printf("Entered\n");
    *idx1 = 0;
    *idx2 = 0;
//    printf("Leaving\n");
    return;
}

__global__ void setIndexForWorklist(float *idx1, float *idx2, float *idx3){
    *idx1 = 0;
    *idx2 = 0;
    *idx3 = 0;
    return;
}

__global__ void ssspEdgeCall(ll totalEdges, ll *dsrc, ll *ddest, ll *dweights, ll *dist, int *ddchanged){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= totalEdges) return;

    ll u = dsrc[id];
    ll v = ddest[id];
    ll wt = dweights[id];

    if(dist[v] > dist[u] + wt){
        atomicMin(&dist[v], dist[u] + wt);
        *ddchanged = 1;
    }
}

__global__ void ssspVertexCall(ll vertices, ll *dindex, ll *dheadVertex, ll *dweights, ll *dist, int *ddchanged){
    unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;

    if(u >= vertices) return;

    ll startIdx = dindex[u];
    ll endIdx = dindex[u + 1];

//    printf("Before u: %lld, dist[u]: %lld", u, dist[u]);

    for(ll i = startIdx; i < endIdx; ++i){
        ll v = dheadVertex[i];
        ll wt = dweights[i];
//        printf("%lld\n", wt);

        if(dist[v] > (dist[u] + wt)){
            atomicMin(&dist[v], dist[u] + wt);
            *ddchanged = 1;
        }
    }

//    printf("After v: %lld, dist[v]: %lld", u, dist[u]);
}

__global__ void ssspVertexInit(ll vertices, ll *dist){
    unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;

    if(u >= vertices) return;

    dist[u] = INT_MAX;
}

__global__ void initSrc(ll src, ll *dist){
    dist[src] = 0;
}

__global__ void init(ll src, ll *dist, ll *curr){
    dist[src] = 0;
    curr[0] = src;
}

__global__ void printDist(ll vertices, ll *dist){
    for(ll u = 0; u < 40; ++u) printf("%lld ", dist[u]);
    printf("\n");
}

__global__ void printCSRKernel(ll vertices, ll *index){
    for(ll u = 0; u < vertices + 1; ++u){
        printf("%lld ", index[u]);
    }
}

__global__ void printTC(ll totalvertices, unsigned int *tc){
    for(long u = 0; u < 40; ++u) printf("%u ", tc[u]);
    printf("\n");
}