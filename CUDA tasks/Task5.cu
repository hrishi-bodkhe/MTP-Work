#include <cuda.h>
#include "preprocessing.h"

__device__ float tc = 0;
__global__ void triangleCountParallel(ll vertices, ll *index, ll *headVertex){
    unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;
    if(u >= vertices) return;
    
    ll start_u = index[u];
    ll end_u = index[u + 1];

    for(ll i = start_u; i < end_u; ++i){
        ll t = headVertex[i];
        ll start_t = index[t];
        ll end_t = index[t + 1];

        for(ll j = start_u; j < end_u; ++j){
                ll r = headVertex[j];

                if(t == r) continue;

                int is_neighbour = 0;
                
                for(ll k = start_t; k < end_t; ++k){
                    if(headVertex[k] == r){
                        is_neighbour = 1;
                        break;
                    }
                }
                
                if(is_neighbour) atomicAdd(&tc, 1);
            }
    }
}

__global__ void printTriangles(){
    tc = tc / 6;
    printf("Using parallel version: %f\n", tc);
}

__global__ void triangleCountSerial(ll vertices, ll *index, ll *headVertex){
    float tcs = 0;

    for(ll p = 0; p < vertices; ++p){
        ll start_p = index[p];
        ll end_p = index[p + 1];

        for(ll i = start_p; i < end_p; ++i){
            ll t = headVertex[i];
            ll start_t = index[t];
            ll end_t = index[t + 1];

            for(ll j = start_p; j < end_p; ++j){
                ll r = headVertex[j];

                if(t == r) continue;

                int is_neighbour = 0;
                
                for(ll k = start_t; k < end_t; ++k){
                    if(headVertex[k] == r){
                        is_neighbour = 1;
                        break;
                    }
                }

                if(is_neighbour) ++tcs;
            }
        }
    }

    printf("Using serial version: %f\n", tcs / 6);
}

int main()
{
    int count = 0;
    int sortedOption;
    cout << "Do you want the edge list to be in sorted order? Enter 1 for Yes or 0 for No. ";
    cin >> sortedOption;
    // cout << sortedOption << endl;

    string file1 = "Graphs/chesapeake.mtx";
    string file2 = "Graphs/inf-luxembourg_osm.mtx";
    string file3 = "Graphs/delaunay_n17.mtx";
    string file4 = "Graphs/kron_g500-logn16.mtx";
    string file5 = "Graphs/rgg_n_2_16_s0.mtx";

    string mtxFilePath = file4;
    double totalTime = 0.0;

    ifstream file(mtxFilePath);

    if (!file.is_open())
    {
        cerr << "Failed to open the file." << endl;
        return 0;
    }

    ll totalVertices;
    ll temp1, totalEdges; // for skipping the first line vertices, edges
    vector<Edge> edgeList;
    string line;
    ll batchSize = INT_MAX;
    bool skipLineOne = true;
    ll prevEdgeCount = 0;

    // Keep count of vertices. Track Max Vertex
    map<ll, ll> vertexCount;
    ll maxVertex = 0;


    int batch = 1;

    ll *hvertexToIndexMap;
    ll *hindex;
    ll *hheadVertex;
    ll *hweights;

    // Copying CSR on GPU
    ll *dvertexToIndexMap;
    ll *dindex;
    ll *dheadVertex;
    ll *dweights;

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

            totalEdges = DIRECTED ? totalEdges : 2 * totalEdges;
            continue;
        }

        ll src, dest, wt;

        istringstream iss(line);
        if (WEIGHTED)
            iss >> src >> dest >> wt;
        else
            iss >> src >> dest;

        Edge e;
        e.src = src - 1;
        e.dest = dest - 1;
        e.wt = WEIGHTED ? wt : 1;

        edgeList.emplace_back(e);

        ++vertexCount[e.src];
        maxVertex = max(maxVertex, e.src);

        if (!DIRECTED)
        {
            e.src = dest - 1;
            e.dest = src - 1;
            e.wt = WEIGHTED ? wt : 1;

            ++vertexCount[e.src];
            maxVertex = max(maxVertex, e.src);

            edgeList.emplace_back(e);
        }

        if (edgeList.size() >= batchSize)
        {
            if (sortedOption)
                sort(edgeList.begin(), edgeList.end(), comp_Edges_and_dest);
            else
                sort(edgeList.begin(), edgeList.end(), comp_Edges);


            ll noOfedges = edgeList.size();
            ll vertices = vertexCount.size();

            hvertexToIndexMap = (ll *)malloc((vertices) * sizeof(ll));
            hindex = (ll *)malloc((vertices + 1) * sizeof(ll));
            hheadVertex = (ll *)malloc(noOfedges * sizeof(ll));
            hweights = (ll *)malloc(noOfedges * sizeof(ll));

            constructCSR(vertices, hindex, hheadVertex, hweights, DIRECTED, WEIGHTED, edgeList, vertexCount, hvertexToIndexMap);
            printCSR(vertices, hindex, hheadVertex, hweights, noOfedges, hvertexToIndexMap);

            cudaMalloc(&dvertexToIndexMap, (ll)(vertices) * sizeof(ll));
            cudaMalloc(&dindex, (ll)(vertices + 1) * sizeof(ll));
            cudaMalloc(&dheadVertex, (ll)(noOfedges) * sizeof(ll));
            cudaMalloc(&dweights, (ll)(noOfedges) * sizeof(ll));

            cudaMemcpy(dvertexToIndexMap, hvertexToIndexMap, (ll)(vertices) * sizeof(ll), cudaMemcpyHostToDevice);
            cudaMemcpy(dindex, hindex, (ll)(vertices + 1) * sizeof(ll), cudaMemcpyHostToDevice);
            cudaMemcpy(dheadVertex, hheadVertex, (ll)(noOfedges) * sizeof(ll), cudaMemcpyHostToDevice);
            cudaMemcpy(dweights, hweights, (ll)(noOfedges) * sizeof(ll), cudaMemcpyHostToDevice);

            unsigned nblocks = ceil((float)vertices / BLOCKSIZE);

            ++batch;
            prevEdgeCount += noOfedges;
            edgeList.clear();
            vertexCount.clear();
            maxVertex = 0;
        }
    }

    if (edgeList.size() > 0)
    {
        if (sortedOption)
            sort(edgeList.begin(), edgeList.end(), comp_Edges_and_dest);
        else
            sort(edgeList.begin(), edgeList.end(), comp_Edges);

        ll noOfedges = edgeList.size();
        ll vertices = vertexCount.size();

        hvertexToIndexMap = (ll *)malloc((vertices) * sizeof(ll));
        hindex = (ll *)malloc((vertices + 1) * sizeof(ll));
        hheadVertex = (ll *)malloc(noOfedges * sizeof(ll));
        hweights = (ll *)malloc(noOfedges * sizeof(ll));

        constructCSR(vertices, hindex, hheadVertex, hweights, DIRECTED, WEIGHTED, edgeList, vertexCount, hvertexToIndexMap);
        // printCSR(vertices, hindex, hheadVertex, hweights, noOfedges, hvertexToIndexMap);

        cudaMalloc(&dvertexToIndexMap, (ll)(vertices) * sizeof(ll));
        cudaMalloc(&dindex, (ll)(vertices + 1) * sizeof(ll));
        cudaMalloc(&dheadVertex, (ll)(noOfedges) * sizeof(ll));
        cudaMalloc(&dweights, (ll)(noOfedges) * sizeof(ll));

        cudaMemcpy(dvertexToIndexMap, hvertexToIndexMap, (ll)(vertices) * sizeof(ll), cudaMemcpyHostToDevice);
        cudaMemcpy(dindex, hindex, (ll)(vertices + 1) * sizeof(ll), cudaMemcpyHostToDevice);
        cudaMemcpy(dheadVertex, hheadVertex, (ll)(noOfedges) * sizeof(ll), cudaMemcpyHostToDevice);
        cudaMemcpy(dweights, hweights, (ll)(noOfedges) * sizeof(ll), cudaMemcpyHostToDevice);

        // /* Vertex Updates
        // ll maxSizeNeeded = nearestPowerOf2(maxVertex);

        // unsigned nblocks = ceil((float)vertices / BLOCKSIZE);

        edgeList.clear();
        vertexCount.clear();
    }

    file.close();

    // printCSR(totalVertices, hindex, hheadVertex, hweights, totalEdges, hvertexToIndexMap);
    cout << "Triangle Counting Stats: " << endl;
    clock_t start, end;
    double elapsedTime;

    start = clock();
    triangleCountSerial<<<1,1>>>(totalVertices, dindex, dheadVertex);
    cudaDeviceSynchronize();
    end = clock();
    elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    cout << "Time for serial version: " << elapsedTime << endl;
    
    unsigned nblocks = ceil((float)totalVertices/ BLOCKSIZE);

    start = clock();
    triangleCountParallel<<<nblocks, BLOCKSIZE>>>(totalVertices, dindex, dheadVertex);
    cudaDeviceSynchronize();
    end = clock();
    elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    printTriangles<<<1,1>>>();
    cudaDeviceSynchronize();
    cout << "Time for parallel version: " << elapsedTime << endl;
    return 0;
}