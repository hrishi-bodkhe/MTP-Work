#include<cuda_runtime.h>
#include "kernels.h"

__global__ void ssspBalancedWorklistKernel(ll workers, ll *dindex, ll *dheadvertex, ll *dweights, ll *curr, ll *next1, ll *next2, ll *dist, float *idx1, float *idx2, ll limit){
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

__global__ void checkCorrectness(ll totalVertices, ll *vdist, ll *wdist, int *equalityFlag){
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= totalVertices) return;

    if(vdist[id] != wdist[id]) equalityFlag = 0;
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
            if(index >= limit) {
                printf("%ld. Index is out of range.\n", index);
                return;
            }
//            printf("index: %ld ", index);
            next[index] = v;
        }
    }
}

__global__ void swapWorklist(ll *curr, ll *next, ll idx){
//    printf("here");
    ll *temp = curr;
    curr = next;
    next = temp;

    printf("idx: %ld\n", idx);

    for(ll i = 0; i < idx; ++i) printf("%ld ", curr[i]);
    printf("\n");
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
    for(ll u = 0; u < 10; ++u) printf("%lld ", dist[u]);
    printf("\n");
}

__global__ void printCSRKernel(ll vertices, ll *index){
    for(ll u = 0; u < vertices + 1; ++u){
        printf("%lld ", index[u]);
    }
}