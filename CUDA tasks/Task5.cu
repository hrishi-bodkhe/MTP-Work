#include <cuda.h>
#include "preprocessing.h"

__device__ float tc = 0;
__global__ void triangleCountParallel(ll vertices, ll *index, ll *headVertex)
{
    unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= vertices)
        return;

    ll start_u = index[u];
    ll end_u = index[u + 1];

    for (ll i = start_u; i < end_u; ++i)
    {
        ll t = headVertex[i];
        ll start_t = index[t];
        ll end_t = index[t + 1];

        for (ll j = start_u; j < end_u; ++j)
        {
            ll r = headVertex[j];

            if (t == r)
                continue;

            int is_neighbour = 0;

            for (ll k = start_t; k < end_t; ++k)
            {
                if (headVertex[k] == r)
                {
                    is_neighbour = 1;
                    break;
                }
            }

            if (is_neighbour)
                atomicAdd(&tc, 1);
        }
    }
}

__global__ void printTriangles()
{
    tc = tc / 6;
    printf("Using parallel version: %f\n", tc);
}

__global__ void triangleCountSerial(ll vertices, ll *index, ll *headVertex)
{
    float tcs = 0;

    for (ll p = 0; p < vertices; ++p)
    {
        ll start_p = index[p];
        ll end_p = index[p + 1];

        for (ll i = start_p; i < end_p; ++i)
        {
            ll t = headVertex[i];
            ll start_t = index[t];
            ll end_t = index[t + 1];

            for (ll j = start_p; j < end_p; ++j)
            {
                ll r = headVertex[j];

                if (t == r)
                    continue;

                int is_neighbour = 0;

                for (ll k = start_t; k < end_t; ++k)
                {
                    if (headVertex[k] == r)
                    {
                        is_neighbour = 1;
                        break;
                    }
                }

                if (is_neighbour)
                    ++tcs;
            }
        }
    }

    printf("Using serial version: %f\n", tcs / 6);
}

__global__ void pagerankInitSerial(ll totalVertices, double *pageRanks)
{
    for (ll u = 0; u < totalVertices; ++u)
    {
        pageRanks[u] = 1 / (double)totalVertices;
    }
}

__global__ void pagerankSerial(ll vertices, ll *index, ll *headVertex, double *pageranks)
{
    ll itr = 1;

    while (itr < MAX_ITRS)
    {
        for (ll u = 0; u < vertices; ++u)
        {
            double val = 0.0;

            ll start_u = index[u];
            ll end_u = index[u + 1];

            for (ll idx = start_u; idx < end_u; ++idx)
            {
                ll v = headVertex[idx];

                ll v_outdegree = index[v + 1] - index[v];

                if (v_outdegree > 0)
                    val += pageranks[v] / v_outdegree;
            }

            pageranks[u] = val * dampingFactor + (1 - dampingFactor) / vertices;
        }

        ++itr;
    }
}

__global__ void printPageranks(ll totalVertices, double *pageranks)
{
    printf("Pageranks are as follow:\n");

    for (ll u = 0; u < totalVertices; ++u)
    {
        printf("%f ", pageranks[u]);
    }

    printf("\n");
}

__global__ void pagerankParallel(ll vertices, ll *index, ll *headvertex, double *prevPagerank, double *currPagerank)
{
    unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u >= vertices)
        return;

    double val = 0.0;

    ll start_u = index[u];
    ll end_u = index[u + 1];

    for(ll idx = start_u; idx < end_u; ++idx){
        ll v = headvertex[idx];

        ll v_outdegree = index[v + 1] - index[v];

        if(v_outdegree > 0) val += prevPagerank[v] / (double)v_outdegree;
    }

    currPagerank[u] = val * dampingFactor + (1 - dampingFactor) / (double)vertices;
}

__global__ void pagerankInitParallel(ll vertices, double *pageRanks)
{
    unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u >= vertices)
        return;

    pageRanks[u] = 1 / (double)vertices;
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
    string file6 = "Graphs/delaunay_n24.mtx";
    string file7 = "Graphs/inf-road_usa.mtx";
    string file8 = "Graphs/delaunay_n19.mtx";
    string file9 = "Graphs/delaunay_n20.mtx";

    string mtxFilePath = file4;
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

    // cout << "Triangle Counting Stats: " << endl;
    clock_t start, end;
    double elapsedTime;

    // start = clock();
    // triangleCountSerial<<<1, 1>>>(totalVertices, dindex, dheadVertex);
    // cudaDeviceSynchronize();
    // end = clock();
    // elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    // cout << "Time for serial version: " << elapsedTime << endl;

    // // -----------------------CALCULATE MEMORY CONSUMPTION-------------------------------------------------------
    // size_t freeMemAfter = calculateMemoryConsumption();
    // std::cout << "Free GPU memory after kernel launch: " << freeMemAfter << " bytes" << std::endl;

    // // Calculate memory used by the kernel
    // size_t memoryUsed = freeMemBefore - freeMemAfter;
    // std::cout << "Memory used by the kernel: " << memoryUsed << " bytes" << std::endl;

    // unsigned nblocks = ceil((float)totalVertices / BLOCKSIZE);

    // start = clock();
    // triangleCountParallel<<<nblocks, BLOCKSIZE>>>(totalVertices, dindex, dheadVertex);
    // cudaDeviceSynchronize();
    // end = clock();
    // elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    // printTriangles<<<1, 1>>>();
    // cudaDeviceSynchronize();
    // cout << "Time for parallel version: " << elapsedTime << endl;

    // // -----------------------CALCULATE MEMORY CONSUMPTION-------------------------------------------------------
    // size_t freeMemAfter = calculateMemoryConsumption();
    // std::cout << "Free GPU memory after kernel launch: " << freeMemAfter << " bytes" << std::endl;

    // // Calculate memory used by the kernel
    // size_t memoryUsed = freeMemBefore - freeMemAfter;
    // std::cout << "Memory used by the kernel: " << memoryUsed << " bytes" << std::endl;

    //--------------------------------------PAGERANK---------------------------------------------

    cout << "RUNNING SERIAL VERSION" << endl;
    computePRSerial(totalVertices, dindex, dheadVertex);

    // cout << "RUNNING PARALLEL VERSION" << endl;
    // computePRParallel(totalVertices, dindex, dheadVertex);

    size_t freeMemAfter = calculateMemoryConsumption();
    std::cout << "Free GPU memory after kernel launch: " << freeMemAfter << " bytes" << std::endl;
    size_t memoryUsed = freeMemBefore - freeMemAfter;
    std::cout << "Memory used by the kernel: " << memoryUsed << " bytes" << std::endl;

    return 0;
}

void computePRParallel(ll vertices, ll *dindex, ll *dheadVertex)
{
    double time = 0;
    double elapsedTime = 0;

    double *prevPageRanks;
    cudaMalloc(&prevPageRanks, (double)vertices * sizeof(double));

    unsigned blocks = ceil((float)vertices / BLOCKSIZE);

    clock_t start, end;
    start = clock();
    pagerankInitParallel<<<blocks, BLOCKSIZE>>>(vertices, prevPageRanks);
    cudaDeviceSynchronize();
    end = clock();
    elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    time += elapsedTime;

    double *currPageRanks;
    cudaMalloc(&currPageRanks, (double)vertices * sizeof(double));

    int last = 0;

    start = clock();
    for (ll itr = 0; itr < MAX_ITRS; ++itr)
    {
        if (itr & 1)
        {
            pagerankParallel<<<blocks, BLOCKSIZE>>>(vertices, dindex, dheadVertex, currPageRanks, prevPageRanks);
            cudaDeviceSynchronize();
            last = 1;
        }
        else
        {
            pagerankParallel<<<blocks, BLOCKSIZE>>>(vertices, dindex, dheadVertex, prevPageRanks, currPageRanks);
            cudaDeviceSynchronize();
            last = 0;
        }
    }
    end = clock();
    elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    
    time += elapsedTime;
    // if (last)
    //     printPageranks<<<1, 1>>>(vertices, prevPageRanks);
    // else
    //     printPageranks<<<1, 1>>>(vertices, currPageRanks);
    // cudaDeviceSynchronize();

    cout << "Total time for page Rank using parallel version: " << time << endl;
}

void computePRSerial(ll vertices, ll *dindex, ll *dheadVertex)
{
    double time = 0;
    double elapsedTime = 0;

    double *prevPageRanks;
    cudaMalloc(&prevPageRanks, (double)vertices * sizeof(double));

    clock_t start, end;
    start = clock();
    pagerankInitSerial<<<1, 1>>>(vertices, prevPageRanks);
    cudaDeviceSynchronize();
    end = clock();
    elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    time += elapsedTime;

    start = clock();
    pagerankSerial<<<1, 1>>>(vertices, dindex, dheadVertex, prevPageRanks);
    cudaDeviceSynchronize();
    end = clock();
    elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    time += elapsedTime;

    printPageranks<<<1, 1>>>(vertices, prevPageRanks);
    cudaDeviceSynchronize();

    cout << "Total time for page Rank using serial version: " << time << endl;
}

size_t calculateMemoryConsumption()
{
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    return freeMem;
}
