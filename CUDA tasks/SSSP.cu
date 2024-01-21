#include<cuda.h>
#include "preprocessing.h"

//__device__ float *idx;

void ssspWorklist(ll totalVertices, ll *dindex, ll *dheadvertex, ll *dweights);

void ssspVertexCentric(ll totalVertices, ll *dindex, ll *dheadvertex, ll *dweights);

void ssspEdgeCentric(ll totalVertices, ll totalEdges, ll *src, ll *dest, ll *weights);

void buildCSR(ll vertices, ll edges, vector<Edge>& edgelist, ll *index, ll *headvertex, ll *weights, unordered_map<ll, ll>& degrees);

void buildCOO(ll edges, vector<Edge>& edgelist, ll *src, ll *dest, ll *weights);

__global__ void ssspWorklistKernel(ll workers, ll *dindex, ll *dheadvertex, ll *dweights, ll *curr, ll *next, ll *dist, float *idx){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= workers) return;
//    printf("id: %d. Here 1\n", id);

    ll u = curr[id];
    ll start = dindex[u];
//    printf("start: %ld \n", start);
    ll end = dindex[u + 1];
//    printf("end: %ld \n", end);
//    printf("Here 2\n");

    for(ll i = start; i < end; ++i){
        ll v = dheadvertex[i];
        ll wt = dweights[i];
//        printf("Here 3\n");
        if(dist[v] > dist[u] + wt){
            atomicMin(&dist[v], dist[u] + wt);
            ll index = atomicAdd(idx, 1);

            next[index] = v;
//            printf("v: %ld, next[%ld]: %ld\n", v, index, next[index]);
//            printf("Here 4\n");
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

int main(){
    int sortedOption;
    cout << "Do you want the edge list to be in sorted order? Enter 1 for Yes or 0 for No. ";
    cin >> sortedOption;

    string filename = "../../Graphs/";
    string ext = ".mtx";

    cout << "Choose input file: "<< endl;
    cout << "1. chesapeake" << endl;
    cout << "2. co-papers-citeseer" << endl;
    cout << "3. co-papers-dblp" << endl;
    cout << "4. delaunay_n17" << endl;
    cout << "5. delaunay_n24" << endl;
    cout << "6. hugetrace-00000" << endl;
    cout << "7. inf-europe_osm" << endl;
    cout << "8. inf-luxembourg_osm" << endl;
    cout << "9. inf-road_usa" << endl;
    cout << "10. kron_g500-logn16" << endl;
    cout << "11. kron_g500-logn21" << endl;
    cout << "12. rgg_n_2_16_s0" << endl;
    cout << "13. rgg_n_2_24_s0" << endl;
    cout << "14. channel-500x100x100-b050" << endl;
    cout << "15. hugetrace-00020" << endl;

    int fileNo;
    cin >> fileNo;

    int directed = 0;
    int weighted = 0;

    switch(fileNo){
        case 1:
            filename += "chesapeake";
            break;
        case 2:
            filename += "co-papers-citeseer";
            break;
        case 3:
            filename += "co-papers-dblp";
            break;
        case 4:
            filename += "delaunay_n17";
            break;
        case 5:
            filename += "delaunay_n24";
            break;
        case 6:
            filename += "hugetrace-00000";
            break;
        case 7:
            filename += "inf-europe_osm";
            break;
        case 8:
            filename += "inf-luxembourg_osm";
            break;
        case 9:
            filename += "inf-road_usa";
            break;
        case 10:
            filename += "kron_g500-logn16";
            break;
        case 11:
            filename += "kron_g500-logn21";
            break;
        case 12:
            filename += "rgg_n_2_16_s0";
            break;
        case 13:
            filename += "rgg_n_2_24_s0";
            break;
        case 14:
            filename += "channel-500x100x100-b050";
            break;
        case 15:
            filename += "hugetrace-00020";
            break;
        default:
            cout << "Invalid Choice." << endl;
            return 0;
    }

    if(fileNo == 10 || fileNo == 11) weighted = 1;

    ifstream file(filename + ext);

    if (!file.is_open())
    {
        cerr << "Failed to open the file." << endl;
        return 0;
    }

    cout << "What do you want to compute?" << endl;
    cout << "1. Vertex-Based SSSP" << endl;
    cout << "2. Edge-Based SSSP" << endl;
    cout << "3. Worklist-Based SSSP" << endl;

    int algoChoice;
    cin >> algoChoice;

    cout << endl;
    cout << "Graph: " << filename << endl;

    ll totalVertices;
    ll temp1, totalEdges; // for skipping the first line vertices, edges
    vector<Edge> edgeList;
    string line;

    ll batchSize = INT_MAX;
    bool skipLineOne = true;

    unordered_map<ll, ll> degrees;

    ll maxDegree = INT_MIN;
    ll avgDegree = 0;

    while (getline(file, line))
    {
        // Skip comments
        if (line[0] == '%')
            continue;

        // Skip the first line after comments
        if (skipLineOne)
        {
            istringstream iss(line);
            iss >> totalVertices >> temp1 >> totalEdges;
            skipLineOne = false;

            totalEdges = directed ? totalEdges : 2 * totalEdges;
            continue;
        }

        ll src, dest, wt;

        istringstream iss(line);
        if (weighted)
            iss >> src >> dest >> wt;
        else
            iss >> src >> dest;

        Edge e;
        e.src = src - 1;
        e.dest = dest - 1;
        e.wt = weighted ? wt : 1;

        edgeList.push_back(e);

        degrees[e.src] += 1;
        maxDegree = max(degrees[e.src], maxDegree);
        avgDegree += degrees[e.src];

        if (!directed)
        {
            e.src = dest - 1;
            e.dest = src - 1;
            e.wt = weighted ? wt : 1;

            edgeList.push_back(e);

            degrees[e.src] += 1;
            maxDegree = max(degrees[e.src], maxDegree);
            avgDegree += degrees[e.src];
        }
    }

    file.close();

    avgDegree /= totalVertices;

    cout << "Vertices: " << totalVertices << endl;
    cout << "Edges: " << totalEdges << endl;
    cout << "Maximum Degree: " << maxDegree << endl;
    cout << "Average Degree: " << avgDegree << endl;

    if(sortedOption) sort(edgeList.begin(), edgeList.end(), comp_Edges_and_dest);

    ll *hindex;
    ll *hsrc;
    ll *hheadvertex;
    ll *hweights;

    if(algoChoice == 1 || algoChoice == 3) hindex = (ll *)malloc((totalVertices + 1) * sizeof(ll));
    else if(algoChoice == 2) hsrc = (ll *)malloc((totalEdges) * sizeof (ll));
    hheadvertex = (ll *)malloc(totalEdges * sizeof(ll));
    hweights = (ll *)malloc(totalEdges * sizeof(ll));

    size_t initialFreeMemory, totalMemory;
    cudaMemGetInfo(&initialFreeMemory, &totalMemory);
    cout << "Initial Free Memory: " << initialFreeMemory / (1024 * 1024 * 1024) << " GB" << endl;

    if(algoChoice == 1 || algoChoice == 3) buildCSR(totalVertices, totalEdges, edgeList, hindex, hheadvertex, hweights, degrees);
    else if(algoChoice == 2) buildCOO(totalEdges, edgeList, hsrc, hheadvertex, hweights);

    ll *dindex;
    ll *dsrc;
    ll *dheadVertex;
    ll *dweights;

    if(algoChoice == 1 || algoChoice == 3) cudaMalloc(&dindex, (ll)(totalVertices + 1) * sizeof(ll));
    else if(algoChoice == 2) cudaMalloc(&dsrc, (ll)(totalEdges) * sizeof(ll));
    cudaMalloc(&dheadVertex, (ll)(totalEdges) * sizeof(ll));
    cudaMalloc(&dweights, (ll)(totalEdges) * sizeof(ll));

    if(algoChoice == 1 || algoChoice == 3) cudaMemcpy(dindex, hindex, (ll)(totalVertices + 1) * sizeof(ll), cudaMemcpyHostToDevice);
    else if(algoChoice == 2) cudaMemcpy(dsrc, hsrc, (ll)(totalEdges) * sizeof(ll), cudaMemcpyHostToDevice);
    cudaMemcpy(dheadVertex, hheadvertex, (ll)(totalEdges) * sizeof(ll), cudaMemcpyHostToDevice);
    cudaMemcpy(dweights, hweights, (ll)(totalEdges) * sizeof(ll), cudaMemcpyHostToDevice);

    cout << endl;
    cout << "Graph Built" << endl;
    cout << endl;

    if(algoChoice == 1) ssspVertexCentric(totalVertices, dindex, dheadVertex, dweights);
    else if(algoChoice == 2) ssspEdgeCentric(totalVertices, totalEdges, dsrc, dheadVertex, dweights);
    else if(algoChoice == 3) ssspWorklist(totalVertices, dindex, dheadVertex, dweights);
    else{
        cout << "Invalid choice!" << endl;
        return 0;
    }

    size_t finalFreeMemory;
    cudaMemGetInfo(&finalFreeMemory, &totalMemory);
    size_t consumedMemory = initialFreeMemory - finalFreeMemory;
    cout << "Final Free Memory: " << finalFreeMemory / (1024 * 1024 * 1024) << " GB" << endl;
    cout << "Consumed Memory: " << consumedMemory / (1024 * 1024) << " MB" << endl;

    return 0;
}

void ssspWorklist(ll totalVertices, ll *dindex, ll *dheadvertex, ll *dweights){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0.0;
    float time;

    ll *dist;
    cudaMalloc(&dist, (ll)(totalVertices) *sizeof(ll));

    ll srcVertex = 0;

    unsigned int nodeblocks = ceil((double)totalVertices / (double)BLOCKSIZE);

    time = 0.0;
    cudaEventRecord(start);
    ssspVertexInit<<<nodeblocks, BLOCKSIZE>>>(totalVertices, dist);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    totalTime += time;

    cout << "Initialized distance array" << endl;
    cout << endl;

    float *workers = (float*)malloc(sizeof(float));
//    cout << "done";
    *workers = 1;

    ll *curr;
    cudaMalloc(&curr, totalVertices * sizeof(ll));

    ll *next;
    cudaMalloc(&next, totalVertices * sizeof(ll));

    cout << "Initialized current worklist" << endl;
    cout << endl;

    float *idx;
    cudaMalloc(&idx, sizeof(float));

    cout << "Defined index for next worklist" << endl;
    cout << endl;

    time = 0.0;
    cudaEventRecord(start);
    init<<<1,1>>>(srcVertex, dist, curr);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    totalTime += time;

    cout << "Initialized source distance and current worklist" << endl;
    cout << endl;

    ll itr = 1;

    while(true){
        unsigned blocks = ceil((double)(*workers) / BLOCKSIZE);
        time = 0.0;
        cudaEventRecord(start);
        setIndexForWorklist<<<1, 1>>>(idx);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        totalTime += time;
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            return;
        }

//        cout << "Here";

//        cout << "Before Kernel: " << endl;
//        printDist<<<1,1>>>(*workers, curr);

        if(itr % 2 != 0) {
            time = 0.0;
            cudaEventRecord(start);
            ssspWorklistKernel<<<blocks, BLOCKSIZE>>>(*workers, dindex, dheadvertex, dweights, curr, next, dist, idx);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            cudaEventElapsedTime(&time, start, stop);
            totalTime += time;
            cudaDeviceSynchronize();
        }
        else{
            time = 0.0;
            cudaEventRecord(start);
            ssspWorklistKernel<<<blocks, BLOCKSIZE>>>(*workers, dindex, dheadvertex, dweights, next, curr, dist, idx);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            cudaEventElapsedTime(&time, start, stop);
            totalTime += time;
            cudaDeviceSynchronize();
        }

        ++itr;

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error1: %s\n", cudaGetErrorString(err));
            return;
        }

        time = 0.0;
        cudaEventRecord(start);
        cudaMemcpy(workers, idx, sizeof(int), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        totalTime += time;

        if(*workers == 0) break;

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            return;
        }
    }

    cout << "Total Iterations: " << itr << endl;

    cout << "First 10 values of dist vector: ";
    printDist<<<1,1>>>(totalVertices, dist);
    cudaDeviceSynchronize();

    cout << "Total Time: " << totalTime << endl;
}

void ssspEdgeCentric(ll totalVertices, ll totalEdges, ll *src, ll *dest, ll *weights){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0.0;
    float time;

    ll *dist;
    cudaMalloc(&dist, (ll)(totalVertices) *sizeof(ll));

    ll srcVertex = 0;

    unsigned int nodeblocks = ceil((double)totalVertices / (double)BLOCKSIZE);

    time = 0.0;
    cudaEventRecord(start);
    ssspVertexInit<<<nodeblocks, BLOCKSIZE>>>(totalVertices, dist);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    totalTime += time;

    cout << "Initialized distance array" << endl;
    cout << endl;

    time = 0.0;
    cudaEventRecord(start);
    initSrc<<<1,1>>>(srcVertex, dist);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    totalTime += time;

    cout << "Initialized source distance" << endl;
    cout << endl;

    int *hchanged;
    hchanged = (int *)malloc(sizeof(int));

    int *dchanged;
    cudaMalloc(&dchanged, sizeof(int));

    unsigned blocks = ceil((double)totalEdges / BLOCKSIZE);

    int itr = 1;

    while(true){
        *hchanged = 0;
        cudaMemcpy(dchanged, hchanged, sizeof(int), cudaMemcpyHostToDevice);

//        cout << "Launching Kernel: " << endl;

        time = 0.0;
        cudaEventRecord(start);
        ssspEdgeCall<<<blocks, BLOCKSIZE>>>(totalEdges, src, dest, weights, dist, dchanged);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        totalTime += time;

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(hchanged, dchanged, sizeof(int), cudaMemcpyDeviceToHost);

//        cout << "Done Iteration: " << itr << endl;

        ++itr;

        if(*hchanged == 0) break;
    }

    cout << "Total Iterations: " << itr << endl;

    cout << "First 10 values of dist vector: ";
    printDist<<<1,1>>>(totalVertices, dist);
    cudaDeviceSynchronize();

    cout << "Total Time: " << totalTime << endl;
}

void ssspVertexCentric(ll totalVertices, ll *dindex, ll *dheadvertex, ll *dweights){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0.0;
    float time;

    ll *dist;
    cudaMalloc(&dist, (ll)(totalVertices) *sizeof(ll));

    ll src = 0;

    unsigned int nodeblocks = ceil((double)totalVertices / (double)BLOCKSIZE);

    time = 0.0;
    cudaEventRecord(start);
    ssspVertexInit<<<nodeblocks, BLOCKSIZE>>>(totalVertices, dist);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    totalTime += time;

    cout << "Initialized distance array" << endl;
    cout << endl;

    time = 0.0;
    cudaEventRecord(start);
    initSrc<<<1,1>>>(src, dist);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    totalTime += time;

    cout << "Initialized source distance" << endl;
    cout << endl;

    int *hchanged;
    hchanged = (int *)malloc(sizeof(int));

    int *dchanged;
    cudaMalloc(&dchanged, sizeof(int));

    int itr = 1;

    while(true){
        *hchanged = 0;
        cudaMemcpy(dchanged, hchanged, sizeof(int), cudaMemcpyHostToDevice);

//        cout << "Launching Kernel: " << endl;

        time = 0.0;
        cudaEventRecord(start);
        ssspVertexCall<<<nodeblocks, BLOCKSIZE>>>(totalVertices, dindex, dheadvertex, dweights, dist, dchanged);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        totalTime += time;

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(hchanged, dchanged, sizeof(int), cudaMemcpyDeviceToHost);

//        cout << "Done Iteration: " << itr << endl;

        ++itr;

        if(*hchanged == 0) break;
    }

    cout << "Total Iterations: " << itr << endl;

    cout << "First 10 values of dist vector: ";
    printDist<<<1,1>>>(totalVertices, dist);
    cudaDeviceSynchronize();

    cout << "Total Time: " << totalTime << endl;
}

void buildCOO(ll edges, vector<Edge>& edgelist, ll *src, ll *dest, ll *weights){
    for(ll i = 0; i < edges; ++i){
        Edge& e = edgelist[i];
        ll u = e.src;
        ll v = e.dest;
        ll wt = e.wt;

        src[i] = u;
        dest[i] = v;
        weights[i] = wt;
    }
}

void buildCSR(ll vertices, ll edges, vector<Edge>& edgelist, ll *index, ll *headvertex, ll *weights, unordered_map<ll, ll>& degrees){
    index[0] = 0;

    for(ll i = 0; i < edges; ++i){
        Edge& e = edgelist[i];
        ll u = e.src;
        ll v = e.dest;
        ll wt = e.wt;

        index[u + 1] = degrees[u];
        headvertex[i] = v;
        weights[i] = wt;
    }

    for(ll u = 1; u < vertices + 1; ++u) index[u] += index[u - 1];
}