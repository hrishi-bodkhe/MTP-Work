#include "preprocessing.h"
#include "kernels.h"
#include<cuda.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// Function to create a new node
Node *createNode(int v, int weighted, int wt)
{
    Node *node = (Node *)malloc(sizeof(Node));
    node->vertex = v;
    node->next = NULL;

    if (weighted == 1)
        node->wt = wt;
    else
        wt = 1;

    return node;
}

bool comp_Edges_and_dest(Edge &a, Edge &b)
{
    return a.src == b.src ? a.dest < b.dest : a.src < b.src;
}

bool comp_Edges(Edge &a, Edge &b)
{
    return a.src < b.src;
}

int takeChoices(int& directed, int& weighted, int& algoChoice, string& filename, int& sortedOption, string& filenameforCorrection){
    filename += "../../Graphs/";

    string ext = ".mtx";
    int fileNo;

    cout << "Do you want the edge list to be in sorted order? Enter 1 for Yes or 0 for No: ";
    cin >> sortedOption;

    cout << endl;
    cout << "Choose input file: "<< endl;
    cout << " 1. chesapeake               2. rgg_n_2_16_s0             3. kron_g500-logn16         4. inf-luxembourg_osm       5. delaunay_n17" << endl;
    cout << " 6. co-papers-citeseer       7. co-papers-dblp            8. kron_g500-logn21         9. hugetrace-00000         10. channel-500x100x100-b050" << endl;
    cout << "11. delaunay_n23            12. hugetrace-00020          13. delaunay_n24            14. rgg_n_2_24_s0           15. inf-road_usa"<< endl;
    cout << "16. nlpkkt240               17. inf-europe_osm" << endl;

    cout << endl;
    cout << "Enter Your Choice: ";
    cin >> fileNo;
    cout << endl;

    switch(fileNo){
        case 1:
            filename += "chesapeake";
            filenameforCorrection += "chesapeake";
            break;
        case 2:
            filename += "rgg_n_2_16_s0";
            filenameforCorrection += "rgg_n_2_16_s0";
            break;
        case 3:
            filename += "kron_g500-logn16";
            filenameforCorrection += "kron_g500-logn16";
            break;
        case 4:
            filename += "inf-luxembourg_osm";
            filenameforCorrection += "inf-luxembourg_osm";
            break;
        case 5:
            filename += "delaunay_n17";
            filenameforCorrection += "delaunay_n17";
            break;
        case 6:
            filename += "co-papers-citeseer";
            filenameforCorrection += "co-papers-citeseer";
            break;
        case 7:
            filename += "co-papers-dblp";
            filenameforCorrection += "co-papers-dblp";
            break;
        case 8:
            filename += "kron_g500-logn21";
            filenameforCorrection += "kron_g500-logn21";
            break;
        case 9:
            filename += "hugetrace-00000";
            filenameforCorrection += "hugetrace-00000";
            break;
        case 10:
            filename += "channel-500x100x100-b050";
            filenameforCorrection += "channel-500x100x100-b050";
            break;
        case 11:
            filename += "delaunay_n23";
            filenameforCorrection += "delaunay_n23";
            break;
        case 12:
            filename += "hugetrace-00020";
            filenameforCorrection += "hugetrace-00020";
            break;
        case 13:
            filename += "delaunay_n24";
            filenameforCorrection += "delaunay_n24";
            break;
        case 14:
            filename += "rgg_n_2_24_s0";
            filenameforCorrection += "rgg_n_2_24_s0";
            break;
        case 15:
            filename += "inf-road_usa";
            filenameforCorrection += "inf-road_usa";
            break;
        case 16:
            filename += "nlpkkt240";
            filenameforCorrection += "nlpkkt240";
            break;
        case 17:
            filename += "inf-europe_osm";
            filenameforCorrection += "inf-europe_osm";
            break;
        default:
            cout << "Invalid Choice." << endl;
            return 0;
    }

    if(fileNo == 3 || fileNo == 8 || fileNo == 16) weighted = 1;

    filename += ext;
    ifstream file(filename);

    if (!file.is_open())
    {
        cerr << "Failed to open the file." << endl;
        return 0;
    }

    cout << "What do you want to compute?" << endl;
    cout << "1. Vertex-Based SSSP" << endl;
    cout << "2. Edge-Based SSSP" << endl;
    cout << "3. Worklist-Based SSSP" << endl;
    cout << "4. Even Odd Thread Distributed Worklist Based SSSP" << endl;
    cout << "5. Balanced Worklist Based SSSP" << endl;
    cout << "6. Edge Centric Worklist Based SSSP" << endl;
    cout << "7. Bucket Based Worklist SSSP" << endl;
    cout << "8. Bucket Based Extended Worklist SSSP" << endl;   // Not Working
    cout << "9. Triangle Counting Vertex Centric" << endl;
    cout << "10. Triangle Counting Edge Centric" << endl;
    cout << "11. Triangle Counting Sorted Vertex Centric" << endl;
    cout << endl;

    cout << "Enter Your Choice: ";

    cin >> algoChoice;

    if(algoChoice >= 9 && algoChoice <= 11) filenameforCorrection = "../Gunrockresults/TC/" + filenameforCorrection + ".txt";
    else filenameforCorrection = "../Gunrockresults/SSSP/" + filenameforCorrection + ".txt";

    cout << endl;
    cout << "Graph: " << filename << endl;

    file.close();
//    filename = "input.txt";

    return 1;
}

void constructCSR(ll &vertices, ll *index, ll *headvertex, ll *weights, int directed, int weighted, vector<Edge> &edgeList, map<ll, ll> vertexCount, ll* vertexToIndexMap)
{
    ll edges = edgeList.size();

    // constructing indices for index array
    index[0] = 0;
    int i = 1;
    for(auto& p: vertexCount){
        index[i] = p.second;
        ++i;
    }

    i = 0;
    for(auto& p: vertexCount){
        vertexToIndexMap[i++] = p.first;
    }

    for (ll j = 1; j < vertices + 1; ++j)
        index[j] += index[j - 1];

    // constructing the headvertex and weights array
    for (ll j = 0; j < edges; ++j)
    {
        Edge e = edgeList[j];
        headvertex[j] = e.dest;
        weights[j] = e.wt;
    }
}

void printCSR(ll &vertices, ll *index, ll *headvertex, ll *weights, ll &edges, ll *vertexToIndexMap)
{
    cout << "----------------------------------STARTED PRINTING CSR---------------------------------" << endl;
//    cout << "Vertex Mapping: ";
//    for(int i = 0; i < vertices; ++i)
//        cout << vertexToIndexMap[i] << ' ';
//    cout << endl;

    cout << "Index: ";
    for(int i = 0; i < vertices + 1; ++i)
        cout << index[i] << ' ';
    cout << endl;

    cout << "Head Vertex: ";
    for (int i = 0; i < edges; ++i)
        cout << headvertex[i] << ' ';
    cout << endl;
//
//    cout << "Weights: ";
//    for (int i = 0; i < edges; ++i)
//        cout << weights[i] << ' ';
//    cout << endl;
}

void printEdgeList(vector<Edge> &edgeList)
{
    cout << "-------------------------------STARTED PRINTING EDGELIST------------------------------" << endl;
    for (Edge e : edgeList)
        cout << e.src << ' ' << e.dest << ' ' << e.wt << endl;
}

ll nearestPowerOf2(ll value) {
    if (value <= 0) {
        return 1;
    }

    ll exponent = round(log2(value));
    return pow(2, exponent);
}

void printTimings(vector<double>& timings){
    for(double i: timings) cout << i << endl;
}

void constructSrcCSR(ll &vertices, ll *index, ll *sources, ll *headvertex, ll *weights, int directed, int weighted, vector<Edge> &edgeList, map<ll, ll> vertexCount, ll* vertexToIndexMap)
{
    ll edges = edgeList.size();

    // constructing indices for index array
    index[0] = 0;
    int i = 1;
    for(auto& p: vertexCount){
        index[i] = p.second;
        ++i;
    }

    i = 0;
    for(auto& p: vertexCount){
        vertexToIndexMap[i++] = p.first;
    }

    for (ll j = 1; j < vertices + 1; ++j)
        index[j] += index[j - 1];

    // constructing the headvertex and weights array
    for (ll j = 0; j < edges; ++j)
    {
        Edge e = edgeList[j];
        sources[j] = e.src;
        headvertex[j] = e.dest;
        weights[j] = e.wt;
    }
}

void ssspBalancedWorklist(ll totalVertices, ll totalEdges, ll *dindex, ll *dheadvertex, ll *dweights, ll src, string &filenameforCorrection){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0.0;
    float time;

    ll *dist;
    cudaMalloc(&dist, (ll)(totalVertices) *sizeof(ll));

    cout << "Chosen source vertex is: " << src << endl;

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
    float *temp1 = (float*)malloc(sizeof(float));       //Host index for 1st worklist
    float *temp2 = (float*)malloc(sizeof(float));       //Host index for 2nd worklist

    *workers = 1;

    ll *curr;
    cudaMalloc(&curr, (4 * totalVertices) * sizeof(ll));

    ll *next1;
    cudaMalloc(&next1, (2 * totalVertices) * sizeof(ll));

    ll *next2;
    cudaMalloc(&next2, (2 * totalVertices) * sizeof(ll));

    cout << "Initialized current worklist" << endl;
    cout << endl;

    float *idx1, *idx2;                 //Device indices for the worklists
    cudaMalloc(&idx1, sizeof(float));
    cudaMalloc(&idx2, sizeof(float));

    cout << "Defined indices for next worklists" << endl;
    cout << endl;

    time = 0.0;
    cudaEventRecord(start);
    init<<<1,1>>>(src, dist, curr);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    totalTime += time;

    cout << "Initialized source distance and current worklist" << endl;
    cout << endl;

    ll itr = 0;

    unsigned blocks = ceil((double)(*workers) / BLOCKSIZE);

    while(true){
        time = 0.0;
        cudaEventRecord(start);
        setIndexForWorklist2<<<1, 1>>>(idx1, idx2);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        totalTime += time;
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error2: %s\n", cudaGetErrorString(err));
            return;
        }

        ll limit = *workers / 2;

        time = 0.0;
        cudaEventRecord(start);
        ssspBalancedWorklistKernel<<<blocks, BLOCKSIZE>>>(2 * totalVertices, *workers, dindex, dheadvertex, dweights, curr, next1, next2, dist, idx1, idx2, limit);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        totalTime += time;
        cudaDeviceSynchronize();

        ++itr;

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error1: %s\n", cudaGetErrorString(err));
            return;
        }

        time = 0.0;
        cudaEventRecord(start);
        cudaMemcpy(temp1, idx1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp2, idx2, sizeof(int), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        totalTime += time;

        *workers = *temp1 + *temp2;

        if(*workers == 0) break;

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error3: %s\n", cudaGetErrorString(err));
            return;
        }

        blocks = ceil((double) (*workers) / BLOCKSIZE);

        time = 0.0;
        cudaEventRecord(start);
        mergeWorklist<<<blocks, BLOCKSIZE>>>(curr, next1, next2, idx1, idx2);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        totalTime += time;

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error4: %s\n", cudaGetErrorString(err));
            return;
        }

//        print2<<<1,1>>>(*workers, curr);
//        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error5: %s\n", cudaGetErrorString(err));
            return;
        }
    }

    cout << "Total Iterations: " << itr << endl;

    cout << "First 10 values of dist vector: ";
    printDist<<<1,1>>>(totalVertices, dist);
    cudaDeviceSynchronize();

    cout << "Total Time: " << totalTime << endl;

    cout << endl;

    cout << "Checking correctness with vertex-centric approach..." << endl;

    ssspVertexCentricCorrectness(totalVertices, dindex, dheadvertex, dweights, src, dist);

    cout << "Checking correctness with SlabGraph results..." << endl;
    checkSSSPCorrectnessWithSlabGraph(dist, filenameforCorrection);
}

void ssspWorklist2(ll totalVertices, ll totalEdges, ll *dindex, ll *dheadvertex, ll *dweights, ll srcVertex, string &filenameforCorrection){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0.0;
    float time;

    ll *dist;
    cudaMalloc(&dist, (ll)(totalVertices) *sizeof(ll));

    cout << "Chosen source vertex is: " << srcVertex << endl;

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
    float *temp1 = (float*)malloc(sizeof(float));       //Host index for 1st worklist
    float *temp2 = (float*)malloc(sizeof(float));       //Host index for 2nd worklist

    *workers = 1;

    ll *curr;
    cudaMalloc(&curr, 4 * totalVertices * sizeof(ll));

    ll *next1;
    cudaMalloc(&next1, 2 * totalVertices * sizeof(ll));

    ll *next2;
    cudaMalloc(&next2, 2 * totalVertices * sizeof(ll));

    cout << "Initialized current worklist" << endl;
    cout << endl;

    float *idx1, *idx2;
    cudaMalloc(&idx1, sizeof(float));
    cudaMalloc(&idx2, sizeof(float));

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

    ll itr = 0;

    unsigned blocks = ceil((double)(*workers) / BLOCKSIZE);

    while(true){
        time = 0.0;
        cudaEventRecord(start);
        setIndexForWorklist2<<<1, 1>>>(idx1, idx2);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        totalTime += time;
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error2: %s\n", cudaGetErrorString(err));
            return;
        }

        time = 0.0;
        cudaEventRecord(start);
        ssspWorklistKernel2<<<blocks, BLOCKSIZE>>>(*workers, dindex, dheadvertex, dweights, curr, next1, next2, dist, idx1, idx2);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        totalTime += time;
        cudaDeviceSynchronize();

        ++itr;

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error1: %s\n", cudaGetErrorString(err));
            return;
        }

        time = 0.0;
        cudaEventRecord(start);
        cudaMemcpy(temp1, idx1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp2, idx2, sizeof(int), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        totalTime += time;

        *workers = *temp1 + *temp2;

        if(*workers == 0) break;

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error3: %s\n", cudaGetErrorString(err));
            return;
        }

        blocks = ceil((double) (*workers) / BLOCKSIZE);

        time = 0.0;
        cudaEventRecord(start);
        mergeWorklist<<<blocks, BLOCKSIZE>>>(curr, next1, next2, idx1, idx2);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        totalTime += time;

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error4: %s\n", cudaGetErrorString(err));
            return;
        }

//        print2<<<1,1>>>(*workers, curr);
//        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error5: %s\n", cudaGetErrorString(err));
            return;
        }
    }

    cout << "Total Iterations: " << itr << endl;

    cout << "First 10 values of dist vector: ";
    printDist<<<1,1>>>(totalVertices, dist);
    cudaDeviceSynchronize();

    cout << "Total Time: " << totalTime << endl;

    cout << endl;

    cout << "Checking correctness with vertex-centric approach..." << endl;

    ssspVertexCentricCorrectness(totalVertices, dindex, dheadvertex, dweights, srcVertex, dist);

    cout << "Checking correctness with SlabGraph results..." << endl;
    checkSSSPCorrectnessWithSlabGraph(dist, filenameforCorrection);
}

void ssspWorklist(ll totalVertices, ll totalEdges, ll *dindex, ll *dheadvertex, ll *dweights, ll srcVertex, string &filenameforCorrection){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0.0;
    float time;

    ll *dist;
    cudaMalloc(&dist, (ll)(totalVertices) *sizeof(ll));

    cout << "Chosen source vertex is: " << srcVertex << endl;

    unsigned int nodeblocks = ceil((double)totalVertices / (double)BLOCKSIZE);

    ssspVertexInit<<<nodeblocks, BLOCKSIZE>>>(totalVertices, dist);
    cudaDeviceSynchronize();
    cout << "Initialized distance array" << endl;
    cout << endl;

    float *workers = (float*)malloc(sizeof(float));
//    cout << "done";
    *workers = 1;

    ll *curr;
    cudaMalloc(&curr, (3 * totalVertices) * sizeof(ll));

    ll *next;
    cudaMalloc(&next, (3 * totalVertices) * sizeof(ll));

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

    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess) {
//        printf("CUDA Error0: %s\n", cudaGetErrorString(err));
//        return;
//    }

    cout << "Initialized source distance and current worklist" << endl;
    cout << endl;

    ll itr = 1;
    unsigned worklist_blocks;

    time = 0.0;
    cudaEventRecord(start);
    while(true){
        worklist_blocks = ceil((double)(*workers) / BLOCKSIZE);

        setIndexForWorklist<<<1, 1>>>(idx);
        cudaDeviceSynchronize();

//        err = cudaGetLastError();
//        if (err != cudaSuccess) {
//            printf("CUDA Error1: %s\n", cudaGetErrorString(err));
//            return;
//        }

        if(itr % 2 != 0) {
            ssspWorklistKernel<<<worklist_blocks, BLOCKSIZE>>>(*workers, dindex, dheadvertex, dweights, curr, next, dist, idx, 3 * totalVertices);
            cudaDeviceSynchronize();

//            err = cudaGetLastError();
//            if (err != cudaSuccess) {
//                printf("CUDA Error odd: %s\n", cudaGetErrorString(err));
//                return;
//            }

//            print<<<1,1>>>(idx, next);
//            cudaDeviceSynchronize();
        }
        else{
            ssspWorklistKernel<<<worklist_blocks, BLOCKSIZE>>>(*workers, dindex, dheadvertex, dweights, next, curr, dist, idx, 3 * totalVertices);
            cudaDeviceSynchronize();

//            err = cudaGetLastError();
//            if (err != cudaSuccess) {
//                printf("CUDA Error even: %s\n", cudaGetErrorString(err));
//                return;
//            }

//            print<<<1,1>>>(idx, curr);
//            cudaDeviceSynchronize();
        }

        ++itr;

//        err = cudaGetLastError();
//        if (err != cudaSuccess) {
//            printf("CUDA Error2: %s\n", cudaGetErrorString(err));
//            return;
//        }

        cudaMemcpy(workers, idx, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        if(*workers == 0) break;

//        err = cudaGetLastError();
//        if (err != cudaSuccess) {
//            printf("CUDA Erro3: %s\n", cudaGetErrorString(err));
//            return;
//        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    totalTime += time;
    cudaDeviceSynchronize();

    cout << "Total Iterations: " << itr << endl;

    cout << "First 40 values of dist vector: ";
    printDist<<<1,1>>>(totalVertices, dist);
    cudaDeviceSynchronize();

    cout << "Total Time: " << totalTime << endl;

    cout << endl;

    cout << "Checking correctness with vertex-centric approach..." << endl;

    ssspVertexCentricCorrectness(totalVertices, dindex, dheadvertex, dweights, srcVertex, dist);

    cout << "Checking correctness with SlabGraph results..." << endl;
    checkSSSPCorrectnessWithSlabGraph(dist, filenameforCorrection);
}

void ssspEdgeCentric(ll totalVertices, ll totalEdges, ll *src, ll *dest, ll *weights, ll srcVertex, string &filenameforCorrection){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0.0;
    float time;

    ll *dist;
    cudaMalloc(&dist, (ll)(totalVertices) *sizeof(ll));

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

    cout << "First 10 values of Device vector: ";
    printDist<<<1,1>>>(totalVertices, dist);
    cudaDeviceSynchronize();

    cout << "Total Time: " << totalTime << endl;

    cout << "Checking correctness with SlabGraph results..." << endl;
    checkSSSPCorrectnessWithSlabGraph(dist, filenameforCorrection);
}

void printssspCpu(ll totalVertices, ll *dist){
    cout << "First 10 values of Host vector: ";
    for(ll i = 0; i < 10; ++i) cout << dist[i] << ' ';
    cout << endl;
}

void ssspSerial(ll totalVertices, ll *index, ll *headvertex, ll *weights, ll *dist, ll src){
    for(ll i = 0; i < totalVertices; ++i) dist[i] = INT_MAX;

    dist[src] = 0;

    int changed;
    while(true){
        changed = 0;
        for(ll u = 0; u < totalVertices; ++u){
            ll start = index[u];
            ll end = index[u + 1];

            for(ll i = start; i < end; ++i){
                ll v = headvertex[i];
                ll wt = weights[i];

                if(dist[v] > dist[u] + wt){
                    dist[v] = dist[u] + wt;
                    changed = 1;
                }
            }
        }

        if(changed == 0) break;
    }
}

void ssspVertexCentricCorrectness(ll totalVertices, ll *dindex, ll *dheadvertex, ll *dweights, ll srcVertex, ll *wdist){
    ll *vdist;      //dist vector for vertex centric approach. wdist is dist vector for worklist based approach
    cudaMalloc(&vdist, (ll)(totalVertices) * sizeof(ll));

    unsigned int nodeblocks = ceil((double)totalVertices / (double)BLOCKSIZE);

    ssspVertexInit<<<nodeblocks, BLOCKSIZE>>>(totalVertices, vdist);
    cudaDeviceSynchronize();

    initSrc<<<1,1>>>(srcVertex, vdist);
    cudaDeviceSynchronize();

    int *hchanged;
    hchanged = (int *)malloc(sizeof(int));

    int *dchanged;
    cudaMalloc(&dchanged, sizeof(int));

    while(true){
        *hchanged = 0;
        cudaMemcpy(dchanged, hchanged, sizeof(int), cudaMemcpyHostToDevice);

        ssspVertexCall<<<nodeblocks, BLOCKSIZE>>>(totalVertices, dindex, dheadvertex, dweights, vdist, dchanged);
        cudaDeviceSynchronize();

        cudaMemcpy(hchanged, dchanged, sizeof(int), cudaMemcpyDeviceToHost);

        if(*hchanged == 0) break;
    }

    cout << "First 40 values of dist vector: ";
    printDist<<<1,1>>>(totalVertices, vdist);
    cudaDeviceSynchronize();

    int *hequalityFlag;
    int *dequalityFlag;

    hequalityFlag = (int *)malloc(sizeof(int));
    cudaMalloc(&dequalityFlag, sizeof(int));

    *hequalityFlag = 1;
    cudaMemcpy(dequalityFlag, hequalityFlag, sizeof(int), cudaMemcpyHostToDevice);

    checkCorrectness<<<nodeblocks, BLOCKSIZE>>>(totalVertices, vdist, wdist, dequalityFlag);
    cudaDeviceSynchronize();

    cudaMemcpy(hequalityFlag, dequalityFlag, sizeof(int), cudaMemcpyDeviceToHost);
    if(*hequalityFlag == 1) cout << "Correctness Verified!" << endl;
    else cout << "Incorrect Result!" << endl;
}

void ssspVertexCentric(ll totalVertices, ll *dindex, ll *dheadvertex, ll *dweights, ll src, string &filenameforCorrection){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0.0;
    float time;

    ll *dist;
    cudaMalloc(&dist, (ll)(totalVertices) *sizeof(ll));

    cout << "Chosen source vertex is: " << src << endl;

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

    cout << "Checking correctness with SlabGraph results..." << endl;
    checkSSSPCorrectnessWithSlabGraph(dist, filenameforCorrection);
}

void ssspEdgeWorklistCentric(ll totalvertices, ll totalEdges, ll *csr_offsets, ll *csr_edges, ll *csr_weights, ll srcVertex, string &filenameforCorrection){
    // Timing Calculations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0.0;
    float time;

    // Allocating space on device for distance vector to store the distances
    ll *dist;
    cudaMalloc(&dist, (ll)(totalvertices) * sizeof(ll));
    cout << "Space allocated for distance vector on device." << endl;
    cout << endl;

    // Initialising distance vector
    unsigned int nodeblocks = ceil((double)totalvertices / (double)BLOCKSIZE);

//    time = 0.0;
//    cudaEventRecord(start);
    ssspVertexInit<<<nodeblocks, BLOCKSIZE>>>(totalvertices, dist);
    cudaDeviceSynchronize();
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&time, start, stop);
//    totalTime += time;

    cout << "Initialized distance array. Chosen source vertex is: " << srcVertex << endl;
    cout << endl;

    // Allocating space on device for input frontier. Size = number of vertices
    ll *input_frontier;
    cudaMalloc(&input_frontier, (ll)(2 * totalvertices) * sizeof(ll));
    cout << "Space allocated for input frontiers." << endl;
    cout << endl;

    ll *deg_for_input_frontier;
    cudaMalloc(&deg_for_input_frontier, (ll)(2 * totalvertices) * sizeof(ll));

    ll *frontier_offset;
    cudaMalloc(&frontier_offset, (ll)(2 * totalvertices + 1) * sizeof(ll));
    cout << "Space allocated for frontier offset." << endl;
    cout << endl;

    // Allocating space on device for output frontier. Size = number of vertices + 1
    ll *output_frontier;
    cudaMalloc(&output_frontier, (ll)(2 * totalvertices) * sizeof(ll));
    cout << "Space allocated for output frontier." << endl;
    cout << endl;

    // Defining global index that can operate on input frontier.
    float *idx;
    cudaMalloc(&idx, sizeof(float));
    cout << "Defined index for input frontier." << endl;
    cout << endl;

    // Initialising distance of source vertex and adding source vertex to input frontier.
    time = 0.0;
    cudaEventRecord(start);
    init<<<1,1>>>(srcVertex, dist, input_frontier);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    totalTime += time;

    cudaError_t err = cudaGetLastError();                   // Catching errors, if any.
    if (err != cudaSuccess) {
        printf("CUDA Error while initialising distance for source vertex: %s\n", cudaGetErrorString(err));
        return;
    }
    cout << "Initialized source distance and added source vertex to input frontier." << endl;
    cout << endl;

    // Defining number of workers. Initializing it with 1.
    float *workers = (float*)malloc(sizeof(float));
    *workers = 1;

    ll iterations = 1;  // For calculating total iterations

    // Meta data for computing frontier offset.
    ll *host_prefix_sum;
    host_prefix_sum = (ll *)malloc(sizeof(ll));
    ll *device_prefix_sum;
    cudaMalloc(&device_prefix_sum, sizeof(ll));

    // Declaring device ptrs for device arrays.
    thrust::device_ptr<ll> thrust_input_ptr;
    thrust::device_ptr<ll> thrust_output_ptr;

    ll frontier_size;
    unsigned sssp_kernel_blocks;
    unsigned degree_blocks;
    float prefixSumTime = 0.0;
    clock_t time_req;

    // Normal SSSP loop
    time = 0.0;
    cudaEventRecord(start);
    while(true){
        // Setting index of the frontier.
        setIndexForWorklist<<<1, 1>>>(idx);
        cudaDeviceSynchronize();

//        err = cudaGetLastError();
//        if (err != cudaSuccess) {
//            printf("CUDA Error while setting index for frontier: %s\n", cudaGetErrorString(err));
//            return;
//        }

        if(iterations % 2 != 0){
            /** Constructing Frontier offset **/
//            cout << "iteration: " << iterations << endl;

            // Allocating frontier_size to number of current workers
            frontier_size = *workers;
//            cout << "Size: " << frontier_size << endl;

            // Replacing nodes present in input_frontier with their respective degrees
            degree_blocks = ceil((double) (frontier_size) / BLOCKSIZE);
            time_req = clock();
            replaceNodeWithDegree<<<degree_blocks, BLOCKSIZE>>>(csr_offsets, input_frontier, deg_for_input_frontier, frontier_size);
            cudaDeviceSynchronize();

//            err = cudaGetLastError();
//            if(err != cudaSuccess){
//                printf("CUDA Error while replacing nodes with their degrees in frontier in odd iteration: %s\n", cudaGetErrorString(err));
//                return;
//            }

            // Assigning device pointers to device arrays
            thrust_input_ptr = thrust::device_pointer_cast(deg_for_input_frontier);
            thrust_output_ptr = thrust::device_pointer_cast(frontier_offset);

            thrust::exclusive_scan(thrust::device, thrust_input_ptr, thrust_input_ptr + frontier_size + 1, thrust_output_ptr);
            cudaDeviceSynchronize();
            time_req = clock() - time_req;

//            constructFrontierOffset<<<1,1>>>(csr_offsets, input_frontier, frontier_offset, frontier_size, device_prefix_sum);

//            err = cudaGetLastError();
//            if(err != cudaSuccess){
//                printf("CUDA Error while constructing frontier offset in odd iteration: %s\n", cudaGetErrorString(err));
//                return;
//            }

            /** Copying device prefix sum to host **/
//            cudaMemcpy(host_prefix_sum, device_prefix_sum, sizeof(ll), cudaMemcpyDeviceToHost);
//            cudaDeviceSynchronize();
            device_prefix_sum = frontier_offset + frontier_size;
            cudaMemcpy(host_prefix_sum, device_prefix_sum, sizeof(ll), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

//            cout << "Prefix Sum: " << *host_prefix_sum << endl;

            // Threads to be launched will be equal to prefix sum
            sssp_kernel_blocks = ceil((double) (*host_prefix_sum) / BLOCKSIZE);

            /** Launching the SSSP edge centric kernel **/
            ssspEdgeWorklist<<<sssp_kernel_blocks, BLOCKSIZE>>>(csr_offsets, csr_edges, csr_weights, input_frontier, frontier_offset, output_frontier, device_prefix_sum, dist, idx, frontier_size);
            cudaDeviceSynchronize();

//            err = cudaGetLastError();
//            if(err != cudaSuccess){
//                printf("CUDA Error while computing distance error in sssp kernel in odd iteration: %s\n", cudaGetErrorString(err));
//                return;
//            }
        }
        else{
            /** Constructing Frontier offset **/

            // Allocating frontier_size to number of current workers
            frontier_size = *workers;

            // Replacing nodes present in input_frontier with their respective degrees
            degree_blocks = ceil((double) (frontier_size) / BLOCKSIZE);
            time_req = clock();
            replaceNodeWithDegree<<<degree_blocks, BLOCKSIZE>>>(csr_offsets, output_frontier, deg_for_input_frontier, frontier_size);
            cudaDeviceSynchronize();

//            err = cudaGetLastError();
//            if(err != cudaSuccess){
//                printf("CUDA Error while replacing nodes with their degrees in frontier in even iteration: %s\n", cudaGetErrorString(err));
//                return;
//            }

            // Assigning device pointers to device arrays
            thrust_input_ptr = thrust::device_pointer_cast(deg_for_input_frontier);
            thrust_output_ptr = thrust::device_pointer_cast(frontier_offset);

//          constructFrontierOffset<<<1,1>>>(csr_offsets, output_frontier, frontier_offset, *workers, device_prefix_sum);
            thrust::exclusive_scan(thrust::device, thrust_input_ptr, thrust_input_ptr + frontier_size + 1, thrust_output_ptr);
            cudaDeviceSynchronize();
            time_req = clock() - time_req;

//            err = cudaGetLastError();
//            if(err != cudaSuccess){
//                printf("CUDA Error while constructing frontier offset in even iteration: %s\n", cudaGetErrorString(err));
//                return;
//            }

            /** Copying device prefix sum to host **/
//            cudaMemcpy(host_prefix_sum, device_prefix_sum, sizeof(ll), cudaMemcpyDeviceToHost);
//            cudaDeviceSynchronize();
            device_prefix_sum = frontier_offset + frontier_size;
            cudaMemcpy(host_prefix_sum, device_prefix_sum, sizeof(ll), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
//            cout << "Prefix Sum: " << *host_prefix_sum << endl;

            // Threads to be launched will be equal to prefix sum
            sssp_kernel_blocks = ceil((double) (*host_prefix_sum) / BLOCKSIZE);

            /** Launching the SSSP edge centric kernel **/
            ssspEdgeWorklist<<<sssp_kernel_blocks, BLOCKSIZE>>>(csr_offsets, csr_edges, csr_weights, output_frontier, frontier_offset, input_frontier, device_prefix_sum, dist, idx, frontier_size);
            cudaDeviceSynchronize();

//            err = cudaGetLastError();
//            if(err != cudaSuccess){
//                printf("CUDA Error while computing distance error in sssp kernel in even iteration: %s\n", cudaGetErrorString(err));
//                return;
//            }
        }

        ++iterations;
        prefixSumTime += ((float)time_req / CLOCKS_PER_SEC * 1000);

        // Copying the number of next workers to host.
        cudaMemcpy(workers, idx, sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        if(*workers == 0) break;

//        err = cudaGetLastError();
//        if(err != cudaSuccess){
//            printf("CUDA Error while copying device index to host workers: %s\n", cudaGetErrorString(err));
//            return;
//        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    totalTime += time;
    cudaDeviceSynchronize();


    cout << "Total Iterations: " << iterations << endl;
    cout << "Source Vertex: " << srcVertex << endl;
    cout << "First 40 values of dist vector: ";
    printDist<<<1,1>>>(totalvertices, dist);
    cudaDeviceSynchronize();
    cout << "Total Time: " << totalTime << endl;
    cout << "Time for Prefix Sum Calculation: " << prefixSumTime << endl;
    cout << endl;

    cout << "Checking correctness with vertex-centric approach..." << endl;
    ssspVertexCentricCorrectness(totalvertices, csr_offsets, csr_edges, csr_weights, srcVertex, dist);

    cout << "Checking correctness with SlabGraph results..." << endl;
    checkSSSPCorrectnessWithSlabGraph(dist, filenameforCorrection);
}

void ssspBucketWorklist(ll totalvertices, ll totaledges, ll *csr_offsets, ll *csr_edges, ll *csr_weights, ll srcVertex, string &filenameforCorrection){
    // Timing Calculations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0.0;
    float time;

    // Allocating space on device for distance vector to store the distances
    ll *dist;
    cudaMalloc(&dist, (ll)(totalvertices) * sizeof(ll));
    cout << "Space allocated for distance vector on device." << endl;
    cout << endl;

    // Initialising distance vector
    unsigned int nodeblocks = ceil((double)totalvertices / (double)BLOCKSIZE);
    ssspVertexInit<<<nodeblocks, BLOCKSIZE>>>(totalvertices, dist);
    cudaDeviceSynchronize();

    cout << "Initialized distance array. Chosen source vertex is: " << srcVertex << endl;
    cout << endl;

    // Allocating space on device for worklists. Size = number of vertices
    ll *curr1, *curr2, *next1, *next2;
    cudaMalloc(&curr1, (ll)(2 * totalvertices) * sizeof(ll));
    cudaMalloc(&curr2, (ll)(2 * totalvertices) * sizeof(ll));
    cudaMalloc(&next1, (ll)(2 * totalvertices) * sizeof(ll));
    cudaMalloc(&next2, (ll)(2 * totalvertices) * sizeof(ll));
    cout << "Space allocated for worklists." << endl;
    cout << endl;

    // Defining global index that can operate on worklists.
    float *idx1, *idx2;
    cudaMalloc(&idx1, sizeof(float));
    cudaMalloc(&idx2, sizeof(float));
    cout << "Defined index for input frontier." << endl;
    cout << endl;

    // Initialising distance of source vertex and adding source vertex to input frontier.
    time = 0.0;
    cudaEventRecord(start);
    init<<<1,1>>>(srcVertex, dist, curr1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    totalTime += time;

    cudaError_t err = cudaGetLastError();                   // Catching errors, if any.
    if (err != cudaSuccess) {
        printf("CUDA Error while initialising distance for source vertex: %s\n", cudaGetErrorString(err));
        return;
    }
    cout << "Initialized source distance and added source vertex to input frontier." << endl;
    cout << endl;

    // Defining number of workers. Initializing it with 1.
    float *workers1 = (float*)malloc(sizeof(float));
    float *workers2 = (float*)malloc(sizeof(float));
    *workers1 = 1;
    *workers2 = 0;

    ll iterations = 1;  // For calculating total iterations

    unsigned worklist1_blocks, worklist2_blocks;

    // Normal SSSP loop
    time = 0.0;
    cudaEventRecord(start);
    while(true){
        // Setting index of the frontier.
        setIndexForWorklist2<<<1, 1>>>(idx1, idx2);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error while setting indices for worklists: %s\n", cudaGetErrorString(err));
            return;
        }

        // Calculating the total blocks required for both kernels
        worklist1_blocks = ceil((double) (*workers1) / BLOCKSIZE);
        worklist2_blocks = ceil((double) (*workers2) / BLOCKSIZE);

        if(iterations % 2 != 0){
            /** Checking whether blocks are positive else no need to launch the kernel. **/
            // ODD iteration: Reading from curr_j worklist and writing to next1 or next2 worklist
            if(worklist1_blocks > 0) ssspBucketWorklistKernel<<<worklist1_blocks, BLOCKSIZE>>>(*workers1, csr_offsets, csr_edges, csr_weights, curr1, next1, next2, dist, idx1, idx2);
            if(worklist2_blocks > 0) ssspBucketWorklistKernel<<<worklist2_blocks, BLOCKSIZE>>>(*workers2, csr_offsets, csr_edges, csr_weights, curr2, next1, next2, dist, idx1, idx2);
        }
        else{
            // EVEN iteration: Reading from next_j worklist and writing to curr1 or curr2 worklist
            if(worklist1_blocks > 0) ssspBucketWorklistKernel<<<worklist1_blocks, BLOCKSIZE>>>(*workers1, csr_offsets, csr_edges, csr_weights, next1, curr1, curr2, dist, idx1, idx2);
            if(worklist2_blocks > 0) ssspBucketWorklistKernel<<<worklist2_blocks, BLOCKSIZE>>>(*workers2, csr_offsets, csr_edges, csr_weights, next2, curr1, curr2, dist, idx1, idx2);
        }
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error for sssp Kernel: %s\n", cudaGetErrorString(err));
            return;
        }

        // 1 Iteration completed!
        ++iterations;

        // Copying the number of next workers to host.
        cudaMemcpy(workers1, idx1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(workers2, idx2, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error while copying device indices to host workers: %s\n", cudaGetErrorString(err));
            return;
        }

        if(*workers1 == 0 && *workers2 == 0) break;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    totalTime += time;
    cudaDeviceSynchronize();

    cout << "Total Iterations: " << iterations << endl;
    cout << "Source Vertex: " << srcVertex << endl;
    cout << "First 40 values of dist vector: ";
    printDist<<<1,1>>>(totalvertices, dist);
    cudaDeviceSynchronize();
    cout << "Total Time: " << totalTime << endl;
    cout << endl;

    cout << "Checking correctness with vertex-centric approach..." << endl;
    ssspVertexCentricCorrectness(totalvertices, csr_offsets, csr_edges, csr_weights, srcVertex, dist);

    cout << "Checking correctness with SlabGraph results..." << endl;
    checkSSSPCorrectnessWithSlabGraph(dist, filenameforCorrection);
}

void ssspBucketWorklist2(ll totalvertices, ll totaledges, ll *csr_offsets, ll *csr_edges, ll *csr_weights, ll srcVertex, string &filenameforCorrection){
    // Timing Calculations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0.0;
    float time;

    // Allocating space on device for distance vector to store the distances
    ll *dist;
    cudaMalloc(&dist, (ll)(totalvertices) * sizeof(ll));
    cout << "Space allocated for distance vector on device." << endl;
    cout << endl;

    // Initialising distance vector
    unsigned int nodeblocks = ceil((double)totalvertices / (double)BLOCKSIZE);
    ssspVertexInit<<<nodeblocks, BLOCKSIZE>>>(totalvertices, dist);
    cudaDeviceSynchronize();

    cout << "Initialized distance array. Chosen source vertex is: " << srcVertex << endl;
    cout << endl;

    // Allocating space on device for worklists. Size = number of vertices
    ll *curr1, *curr2, *curr3, *next1, *next2, *next3;
    cudaMalloc(&curr1, (ll)(2 * totalvertices) * sizeof(ll));
    cudaMalloc(&curr2, (ll)(2 * totalvertices) * sizeof(ll));
    cudaMalloc(&curr3, (ll)(2 * totalvertices) * sizeof(ll));
    cudaMalloc(&next1, (ll)(2 * totalvertices) * sizeof(ll));
    cudaMalloc(&next2, (ll)(2 * totalvertices) * sizeof(ll));
    cudaMalloc(&next3, (ll)(2 * totalvertices) * sizeof(ll));
    cout << "Space allocated for worklists." << endl;
    cout << endl;

    // Defining global index that can operate on worklists.
    float *idx1, *idx2, *idx3;
    cudaMalloc(&idx1, sizeof(float));
    cudaMalloc(&idx2, sizeof(float));
    cudaMalloc(&idx3, sizeof(float));
    cout << "Defined index for input frontier." << endl;
    cout << endl;

    // Initialising distance of source vertex and adding source vertex to input frontier.
    time = 0.0;
    cudaEventRecord(start);
    init<<<1,1>>>(srcVertex, dist, curr1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    totalTime += time;

    cudaError_t err = cudaGetLastError();                   // Catching errors, if any.
    if (err != cudaSuccess) {
        printf("CUDA Error while initialising distance for source vertex: %s\n", cudaGetErrorString(err));
        return;
    }
    cout << "Initialized source distance and added source vertex to input frontier." << endl;
    cout << endl;

    // Defining number of workers. Initializing it with 1.
    float *workers1 = (float*)malloc(sizeof(float));
    float *workers2 = (float*)malloc(sizeof(float));
    float *workers3 = (float*)malloc(sizeof(float));
    *workers1 = 1;
    *workers2 = 0;
    *workers3 = 0;

    ll iterations = 1;  // For calculating total iterations

    unsigned worklist1_blocks, worklist2_blocks, worklist3_blocks;

    // Normal SSSP loop
    time = 0.0;
    cudaEventRecord(start);
    while(true){
        // Setting index of the frontier.
        setIndexForWorklist<<<1, 1>>>(idx1, idx2, idx3);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error while setting indices for worklists: %s\n", cudaGetErrorString(err));
            return;
        }

        // Calculating the total blocks required for both kernels
        worklist1_blocks = ceil((double) (*workers1) / BLOCKSIZE);
        worklist2_blocks = ceil((double) (*workers2) / BLOCKSIZE);
        worklist3_blocks = ceil((double) (*workers3) / BLOCKSIZE);

        if(iterations % 2 != 0){
            /** Checking whether blocks are positive else no need to launch the kernel. **/
            // ODD iteration: Reading from curr_j worklist and writing to next1 or next2 worklist
            if(worklist1_blocks > 0) ssspBucketWorklistKernel2<<<worklist1_blocks, BLOCKSIZE>>>(*workers1, csr_offsets, csr_edges, csr_weights, curr1, next1, next2, next3, dist, idx1, idx2, idx3);
            cudaDeviceSynchronize();
            if(worklist2_blocks > 0) ssspBucketWorklistKernel2<<<worklist2_blocks, BLOCKSIZE>>>(*workers2, csr_offsets, csr_edges, csr_weights, curr2, next1, next2, next3, dist, idx1, idx2, idx3);
            cudaDeviceSynchronize();
            if(worklist3_blocks > 0) ssspBucketWorklistKernel2<<<worklist3_blocks, BLOCKSIZE>>>(*workers3, csr_offsets, csr_edges, csr_weights, curr3, next1, next2, next3, dist, idx1, idx2, idx3);
            cudaDeviceSynchronize();

        }
        else{
            // EVEN iteration: Reading from next_j worklist and writing to curr1 or curr2 worklist
            if(worklist1_blocks > 0) ssspBucketWorklistKernel2<<<worklist1_blocks, BLOCKSIZE>>>(*workers1, csr_offsets, csr_edges, csr_weights, next1, curr1, curr2, curr3, dist, idx1, idx2, idx3);
            cudaDeviceSynchronize();
            if(worklist2_blocks > 0) ssspBucketWorklistKernel2<<<worklist2_blocks, BLOCKSIZE>>>(*workers2, csr_offsets, csr_edges, csr_weights, next2, curr1, curr2, curr3, dist, idx1, idx2, idx3);
            cudaDeviceSynchronize();
            if(worklist3_blocks > 0) ssspBucketWorklistKernel2<<<worklist3_blocks, BLOCKSIZE>>>(*workers3, csr_offsets, csr_edges, csr_weights, next3, curr1, curr2, curr3, dist, idx1, idx2, idx3);
            cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error for sssp Kernel: %s\n", cudaGetErrorString(err));
            return;
        }

        // 1 Iteration completed!
        ++iterations;

        // Copying the number of next workers to host.
        cudaMemcpy(workers1, idx1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(workers2, idx2, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(workers3, idx2, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error while copying device indices to host workers: %s\n", cudaGetErrorString(err));
            return;
        }

        if(*workers1 == 0 && *workers2 == 0 && *workers3 == 0) break;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    totalTime += time;
    cudaDeviceSynchronize();

    cout << "Total Iterations: " << iterations << endl;
    cout << "Source Vertex: " << srcVertex << endl;
    cout << "First 40 values of dist vector: ";
    printDist<<<1,1>>>(totalvertices, dist);
    cudaDeviceSynchronize();
    cout << "Total Time: " << totalTime << endl;
    cout << endl;

    cout << "Checking correctness with vertex-centric approach..." << endl;
    ssspVertexCentricCorrectness(totalvertices, csr_offsets, csr_edges, csr_weights, srcVertex, dist);

    cout << "Checking correctness with SlabGraph results..." << endl;
    checkSSSPCorrectnessWithSlabGraph(dist, filenameforCorrection);
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

void checkTCCorrectnessWithSlabGraph(unsigned int *wdist, string &filename){
    // Source Vertex should be 0
    if(filename == "../Gunrockresults/SSSP/nlpkkt240.txt") {
        cout << "Results are not available." << endl;
        return;
    }

    unsigned int *vdist;
    cudaMalloc(&vdist, (40) * sizeof (unsigned int));

    ifstream file(filename); // replace with your file name
    string line;
    vector<ll> numbers;

    if (file.is_open()) {
        if (getline(file, line)) {
            istringstream iss(line);
            int num;
            while (iss >> num) {
                numbers.push_back(num);
            }
        }
        file.close();
    }

    unsigned int vectorsize = numbers.size();

    unsigned int *temp;
    temp = (unsigned int *) malloc(vectorsize * sizeof(unsigned int));

    for(int i = 0; i < vectorsize; ++i){
        temp[i] = numbers[i];
    }

    cout << endl;

    cudaMemcpy(vdist, temp, (vectorsize) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    int *hequalityFlag;
    int *dequalityFlag;

    hequalityFlag = (int *)malloc(sizeof(int));
    cudaMalloc(&dequalityFlag, sizeof(int));

    *hequalityFlag = 1;
    cudaMemcpy(dequalityFlag, hequalityFlag, sizeof(int), cudaMemcpyHostToDevice);

    unsigned int nodeblocks = ceil((double)vectorsize / (double)BLOCKSIZE);

    checkCorrectness<<<nodeblocks, BLOCKSIZE>>>(vectorsize, vdist, wdist, dequalityFlag);
    cudaDeviceSynchronize();

    cout << "First 40 values of TC: ";
    printTC<<<1,1>>>(vectorsize, vdist);
    cudaDeviceSynchronize();

    cout << endl;

    cudaMemcpy(hequalityFlag, dequalityFlag, sizeof(int), cudaMemcpyDeviceToHost);
    if(*hequalityFlag == 1) cout << "Correctness Verified with SlabGraph!" << endl;
    else cout << "Incorrect Result!" << endl;
}

void checkSSSPCorrectnessWithSlabGraph(ll *wdist, string &filename){
    // Source Vertex should be 0
    if(filename == "../Gunrockresults/SSSP/nlpkkt240.txt") {
        cout << "Results are not available." << endl;
        return;
    }

    ll *vdist;
    cudaMalloc(&vdist, (40) * sizeof (ll));

    ifstream file(filename); // replace with your file name
    string line;
    vector<ll> numbers;

    if (file.is_open()) {
        if (getline(file, line)) {
            istringstream iss(line);
            int num;
            while (iss >> num) {
                numbers.push_back(num);
            }
        }
        file.close();
    }

    ll vectorsize = numbers.size();

    ll *temp;
    temp = (ll *) malloc(vectorsize * sizeof(ll));

    for(int i = 0; i < vectorsize; ++i){
        temp[i] = numbers[i];
    }

    cudaMemcpy(vdist, temp, (vectorsize) * sizeof(ll), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    int *hequalityFlag;
    int *dequalityFlag;

    hequalityFlag = (int *)malloc(sizeof(int));
    cudaMalloc(&dequalityFlag, sizeof(int));

    *hequalityFlag = 1;
    cudaMemcpy(dequalityFlag, hequalityFlag, sizeof(int), cudaMemcpyHostToDevice);

    unsigned int nodeblocks = ceil((double)vectorsize / (double)BLOCKSIZE);

    checkCorrectness<<<nodeblocks, BLOCKSIZE>>>(vectorsize, vdist, wdist, dequalityFlag);
    cudaDeviceSynchronize();

    cout << "First 40 values of TC: ";
    printDist<<<1,1>>>(vectorsize, vdist);
    cudaDeviceSynchronize();

    cout << endl;

    cudaMemcpy(hequalityFlag, dequalityFlag, sizeof(int), cudaMemcpyDeviceToHost);
    if(*hequalityFlag == 1) cout << "Correctness Verified with SlabGraph!" << endl;
    else cout << "Incorrect Result!" << endl;
}

void triangleCount(ll totalvertices, ll totaledges, ll *csr_offsets, ll *csr_edges, string &filenameforCorrection){
    // Timing Calculations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0.0;
    float time;

    // Host side Triangle Count
    float *host_tc = (float *)malloc(sizeof(float *));

    // Device side Triangle Count
    float *device_tc;
    cudaMalloc(&device_tc, sizeof(float));

    ll *device_tc_array;
    cudaMalloc(&device_tc_array, (totalvertices) * sizeof(ll));

    unsigned blocks = ceil((double) totalvertices / BLOCKSIZE);

    cout << endl;
    cout << "Launching TC Kernel" << endl;

    // Kernel for TC
    time = 0.0;
    cudaEventRecord(start);
    triangleCountVertexCentric<<<blocks, BLOCKSIZE>>>(csr_offsets, csr_edges, totalvertices, device_tc_array);
    cudaDeviceSynchronize();

//    divideTCbysix<<<1,1>>>(device_tc);
//    cudaDeviceSynchronize();

//    cudaMemcpy(host_tc, device_tc, sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    totalTime += time;
    cudaDeviceSynchronize();

    cout << "Finished Kernel" << endl;

//    cout << "Triangle Count: " << *host_tc << endl;
    cout << "Total Time: " << totalTime << endl;
    cout << "First 40 values of TC: ";
    printDist<<<1,1>>>(totalvertices, device_tc_array);
    cudaDeviceSynchronize();
    cout << endl;

    cout << "Checking Correctness with Gunrock..." << endl;
    checkSSSPCorrectnessWithSlabGraph(device_tc_array, filenameforCorrection);
}

void triangleCountEdgeCentric(ll totalvertices, ll totaledges, ll *csr_offsets, ll *csr_edges,  string &filenameforCorrection){
    // Timing Calculations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0.0;
    float time;

    unsigned int *device_tc_array;
    cudaMalloc(&device_tc_array, (totalvertices) * sizeof(unsigned int));

    unsigned blocks = ceil((double) totaledges / BLOCKSIZE);

    cout << endl;
    cout << "Launching Edge Centric TC Kernel" << endl;

    // Kernel for TC
    time = 0.0;
    cudaEventRecord(start);
    triangleCountEdgeCentricKernel<<<blocks, BLOCKSIZE>>>(csr_offsets, csr_edges, device_tc_array, totaledges, totalvertices);
    cudaDeviceSynchronize();

    blocks = ceil((double) totalvertices / BLOCKSIZE);
    divideTCArray<<<blocks, BLOCKSIZE>>>(device_tc_array, 2, totalvertices);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    totalTime += time;
    cudaDeviceSynchronize();

    cout << "Finished Kernel" << endl;

    cout << "Total Time: " << totalTime << endl;
    cout << "First 40 values of TC: ";
    printTC<<<1,1>>>(totalvertices, device_tc_array);
    cudaDeviceSynchronize();
    cout << endl;

    cout << "Checking Correctness with Gunrock..." << endl;
    checkTCCorrectnessWithSlabGraph(device_tc_array, filenameforCorrection);
}

void triangleCountSortedVertexCentric(ll totalvertices, ll *csr_offsets, ll *csr_edges,  string &filenameforCorrection){
    // Timing Calculations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0.0;
    float time;

    unsigned int *device_tc_array;
    cudaMalloc(&device_tc_array, (totalvertices) * sizeof(unsigned int));

    unsigned blocks = ceil((double) totalvertices / BLOCKSIZE);

    cout << endl;
    cout << "Launching Sorted Vertex Centric TC Kernel" << endl;

    // Kernel for TC
    time = 0.0;
    cudaEventRecord(start);
    triangleCountSortedVertexCentricKernel<<<blocks, BLOCKSIZE>>>(csr_offsets, csr_edges, device_tc_array, totalvertices);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    totalTime += time;
    cudaDeviceSynchronize();

    cout << "Finished Kernel" << endl;

    cout << "Total Time: " << totalTime << endl;
    cout << "First 40 values of TC: ";
    printTC<<<1,1>>>(totalvertices, device_tc_array);
    cudaDeviceSynchronize();
    cout << endl;

    cout << "Checking Correctness with Gunrock..." << endl;
    checkTCCorrectnessWithSlabGraph(device_tc_array, filenameforCorrection);
}