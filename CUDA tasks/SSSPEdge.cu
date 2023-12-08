#include <cuda.h>
#include "preprocessing.h"

__device__ int changed;

__global__ void ssspEdgeChild(ll edges, ll *dist, ll *sources, ll *headVertex, ll *weights, ll src){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= edges) return;

    ll p = sources[id];
    ll t = headVertex[id];
    ll wt = weights[id];

    if(dist[t] > dist[p] + wt){
        atomicMin(&dist[t], dist[p] + wt);
        changed = 1;
    }
}

__global__ void ssspEdgeParent(ll edges, ll *dist, ll *sources, ll *headVertex, ll *weights, ll src){
    dist[src] = 0;

    unsigned blocks = ceil((double)edges / BLOCKSIZE);

    changed = 1;

    while(changed){
        changed = 0;
        ssspEdgeChild<<<blocks, BLOCKSIZE>>>(edges, dist, sources, headVertex, weights, src);
        cudaDeviceSynchronize();

        if(changed == 0) break;
    }
}

__global__ void ssspEdgeInit(ll vertices, ll *dist){
    unsigned u = blockIdx.x * blockDim.x + threadIdx.x;

    if(u >= vertices) return;

    dist[u] = INT_MAX;
}

__global__ void printssspDist(ll totalVertices, ll *dist){
    for(ll u = 0; u < totalVertices; ++u)
        printf("%ld ", dist[u]);
    printf("\n");
}

int main()
{
    int count = 0;
    int sortedOption;
    cout << "Do you want the edge list to be in sorted order? Enter 1 for Yes or 0 for No. ";
    cin >> sortedOption;
    // cout << sortedOption << endl;

    string file1 = "../../Graphs/chesapeake.mtx";
    string file2 = "../../Graphs/inf-luxembourg_osm.mtx";
    string file3 = "../../Graphs/delaunay_n17.mtx";
    string file4 = "../../Graphs/kron_g500-logn16.mtx";
    string file5 = "../../Graphs/rgg_n_2_16_s0.mtx";
    string file6 = "../../Graphs/delaunay_n24.mtx";
    string file7 = "../../Graphs/inf-road_usa.mtx";
    string file8 = "../../Graphs/delaunay_n19.mtx";
    string file9 = "../../Graphs/delaunay_n20.mtx";

    string mtxFilePath = file3;
    double totalTime = 0.0;

    size_t freeMemBefore = calculateMemoryConsumption();
    std::cout << "Free GPU memory before kernel launch: " << freeMemBefore << " bytes" << std::endl;

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
    ll *hsources;

    // Copying CSR on GPU
    ll *dvertexToIndexMap;
    ll *dindex;
    ll *dheadVertex;
    ll *dweights;
    ll *dsources;

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
            hsources = (ll *)malloc(noOfedges * sizeof(ll));

            constructSrcCSR(vertices, hindex, hsources, hheadVertex, hweights, DIRECTED, WEIGHTED, edgeList, vertexCount, hvertexToIndexMap);
            // printCSR(vertices, hindex, hheadVertex, hweights, noOfedges, hvertexToIndexMap);

            cudaMalloc(&dvertexToIndexMap, (ll)(vertices) * sizeof(ll));
            cudaMalloc(&dindex, (ll)(vertices + 1) * sizeof(ll));
            cudaMalloc(&dheadVertex, (ll)(noOfedges) * sizeof(ll));
            cudaMalloc(&dweights, (ll)(noOfedges) * sizeof(ll));
            cudaMalloc(&dsources, (ll)(noOfedges) * sizeof(ll));

            cudaMemcpy(dvertexToIndexMap, hvertexToIndexMap, (ll)(vertices) * sizeof(ll), cudaMemcpyHostToDevice);
            cudaMemcpy(dindex, hindex, (ll)(vertices + 1) * sizeof(ll), cudaMemcpyHostToDevice);
            cudaMemcpy(dheadVertex, hheadVertex, (ll)(noOfedges) * sizeof(ll), cudaMemcpyHostToDevice);
            cudaMemcpy(dweights, hweights, (ll)(noOfedges) * sizeof(ll), cudaMemcpyHostToDevice);
            cudaMemcpy(dsources, hsources, (ll)(noOfedges) * sizeof(ll), cudaMemcpyHostToDevice);

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
        hsources = (ll *)malloc(noOfedges * sizeof(ll));

        constructSrcCSR(vertices, hindex, hsources, hheadVertex, hweights, DIRECTED, WEIGHTED, edgeList, vertexCount, hvertexToIndexMap);
        // printCSR(vertices, hindex, hheadVertex, hweights, noOfedges, hvertexToIndexMap);

        cudaMalloc(&dvertexToIndexMap, (ll)(vertices) * sizeof(ll));
        cudaMalloc(&dindex, (ll)(vertices + 1) * sizeof(ll));
        cudaMalloc(&dheadVertex, (ll)(noOfedges) * sizeof(ll));
        cudaMalloc(&dweights, (ll)(noOfedges) * sizeof(ll));
        cudaMalloc(&dsources, (ll)(noOfedges) * sizeof(ll));

        cudaMemcpy(dvertexToIndexMap, hvertexToIndexMap, (ll)(vertices) * sizeof(ll), cudaMemcpyHostToDevice);
        cudaMemcpy(dindex, hindex, (ll)(vertices + 1) * sizeof(ll), cudaMemcpyHostToDevice);
        cudaMemcpy(dheadVertex, hheadVertex, (ll)(noOfedges) * sizeof(ll), cudaMemcpyHostToDevice);
        cudaMemcpy(dweights, hweights, (ll)(noOfedges) * sizeof(ll), cudaMemcpyHostToDevice);
        cudaMemcpy(dsources, hsources, (ll)(noOfedges) * sizeof(ll), cudaMemcpyHostToDevice);

        // /* Vertex Updates
        // ll maxSizeNeeded = nearestPowerOf2(maxVertex);

        // unsigned nblocks = ceil((float)vertices / BLOCKSIZE);

        edgeList.clear();
        vertexCount.clear();
    }

    file.close();

    // printCSR(totalVertices, hindex, hheadVertex, hweights, totalEdges, hvertexToIndexMap);
    cout << "Graph Built" << endl;

    cout << "Edge Centric SSSP Stats: " << endl;

    ll src = 0;

    ssspEdge(totalVertices, totalEdges, dsources, dheadVertex, dweights, src);

    size_t freeMemAfter = calculateMemoryConsumption();
    std::cout << "Free GPU memory after kernel launch: " << freeMemAfter << " bytes" << std::endl;
    size_t memoryUsed = freeMemBefore - freeMemAfter;
    std::cout << "Memory used by the kernel: " << memoryUsed << " bytes" << std::endl;

    return 0;
}

void ssspEdge(ll totalVertices, ll totalEdges, ll *dsources, ll *dheadVertex, ll *dweights, ll src){
    ll *dist;
    // ll *pred;
    cudaMalloc(&dist, (ll)(totalVertices) * sizeof(ll));
    // cudaMalloc(&pred, (ll)(totalVertices) * sizeof(ll));

    unsigned nodeblocks = totalVertices / BLOCKSIZE;

    clock_t start, end;
    double elapsedTime = 0.0;

    start = clock();
    ssspEdgeInit<<<nodeblocks, BLOCKSIZE>>>(totalVertices, dist);
    cudaDeviceSynchronize();
    end = clock();
    elapsedTime += (double) (end - start) / CLOCKS_PER_SEC * 1000.0;

    start = clock();
    ssspEdgeParent<<<1, 1>>>(totalEdges, dist, dsources, dheadVertex, dweights, src);
    cudaDeviceSynchronize();
    end = clock();
    elapsedTime += (double)  (end - start) / CLOCKS_PER_SEC * 1000.0;

    printssspDist<<<1, 1>>>(totalVertices, dist);
    cudaDeviceSynchronize();

    cout << "Time for Edge Centric SSSP: " << elapsedTime << endl;
}

size_t calculateMemoryConsumption()
{
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    return freeMem;
}