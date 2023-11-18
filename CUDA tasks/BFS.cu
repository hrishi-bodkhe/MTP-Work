#include <cuda.h>
#include "preprocessing.h"



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

void bfsTD(ll totalVertices, ll *dindex, ll *dheadVertex){
    ll *dist;
    ll *pred;
    cudaMalloc(&dist, (ll)(totalVertices) * sizeof(ll));
    cudaMalloc(&pred, (ll)(totalVertices) * sizeof(ll));

    unsigned blocks = totalVertices / BLOCKSIZE;

    bfsTDInit<<<blocks, BLOCKSIZE>>>(totalVertices, dist, pred);

    int changed = 1;

    while(changed){
        changed = 0;

        bfsTDUpdate<<<blocks, BLOCKSIZE>>>(totalVertices, dist, pred, dindex, dheadVertex);

        
    }
}

size_t calculateMemoryConsumption()
{
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    return freeMem;
}
