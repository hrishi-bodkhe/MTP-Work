#include <cuda.h>
#include "preprocessing.h"

__global__ void pagerank(ll totalVertices, ll *outdegrees, double *prevPagerank, double *currPagerank, Node **adjList)
{
    unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u >= totalVertices)
        return;

    double val = 0.0;

    for (ll i = 0; i < totalVertices; ++i)
    {
        Node *node = adjList[i];

        while (node)
        {
            if (node->vertex == u)
                break;
            node = node->next;
        }

        if (node && outdegrees[i] > 0)
            val += prevPagerank[i] / (double)outdegrees[i];
    }

    currPagerank[u] = val * dampingFactor + (1 - dampingFactor) / (double)totalVertices;
}

__global__ void pagerankInit(ll totalVertices, double *pageRanks)
{
    unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u >= totalVertices)
        return;

    pageRanks[u] = 1 / (double)totalVertices;
}

__global__ void calculateOutdegree(ll totalVertices, ll *outdegrees, Node **adjList)
{
    unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u >= totalVertices)
        return;

    outdegrees[u] = 0;

    Node *temp = adjList[u];

    while (temp)
    {
        ++outdegrees[u];
        temp = temp->next;
    }
}

__global__ void calculateIndegree(ll totalVertices, ll *indegrees, Node **adjList)
{

    for (int u = 0; u < totalVertices; ++u)
    {
        Node *temp = adjList[u];

        while (temp)
        {
            ll v = temp->vertex;
            ++indegrees[v];
            temp = temp->next;
        }
    }
}

__global__ void generateAdjListParallel(ll vertices, ll *index, ll *headVertex, ll *weights, Node *nodequeue, Node **adjList, ll *vertexToIndexMap, ll prevEdgeCount)
{
    unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u >= vertices)
        return;
    // printf("%ld\n", u);

    ll u_data = vertexToIndexMap[u];
    // printf("%ld\n", u_data);

    ll startIdx = index[u];
    ll endIdx = index[u + 1];

    for (ll idx = startIdx; idx < endIdx; ++idx)
    {
        ll v = headVertex[idx];
        ll wt = weights[idx];

        Node *node = nodequeue + prevEdgeCount + idx;
        // printf("%p\n", node);
        // if(node == NULL) printf("ye");
        // else printf("no");

        // if(node == NULL) printf("%ld ",qIndex);
        // else printf("no ");

        node->vertex = v;
        node->wt = wt;
        node->next = NULL;

        Node *temp = adjList[u_data];
        // printf("%ld ", node->vertex);

        if (!temp)
            adjList[u_data] = node;
        else
        {
            node->next = temp;
            adjList[u_data] = node;
        }
    }
}

__global__ void copyAdjacencyList(Node **oldList, Node **newList, ll oldSize)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id > oldSize)
        return;

    Node *node = oldList[id];

    newList[id] = node;
}

__global__ void updateOldToNew(Node **deviceAdjList, Node **newdeviceAdjList, Node **temp)
{
    temp = deviceAdjList;
    deviceAdjList = newdeviceAdjList;
}

__global__ void printAdjListKernel(ll vertices, Node **adjList)
{
    printf("---------------------------------STARTED PRINTING--------------------------------------\n");
    for (ll u = 0; u < vertices; ++u)
    {
        printf("%ld: ", u);

        Node *temp = adjList[u];

        // if(temp) printf("%ld ", temp->vertex);

        while (temp)
        {
            printf("(%ld, %ld), ", temp->vertex, temp->wt);
            temp = temp->next;
        }

        printf("\n");
    }
}

__global__ void printIndegrees(ll totalVertices, ll *indegrees)
{
    for (ll u = 0; u < totalVertices; ++u)
    {
        printf("%ld ", indegrees[u]);
    }
    printf("\n");
}

__global__ void printOutdegrees(ll totalVertices, ll *outdegrees)
{
    for (ll u = 0; u < totalVertices; ++u)
    {
        printf("%ld ", outdegrees[u]);
    }
    printf("\n");
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

    string mtxFilePath = file6;
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
    ll batchSize = 10000;
    bool skipLineOne = true;
    ll prevEdgeCount = 0;

    // Keep count of vertices. Track Max Vertex
    map<ll, ll> vertexCount;
    ll maxVertex = 0;

    // Defining Edge queue
    Node *nodeQueue;

    // Defining deviceAdjList;
    Node **deviceAdjList;
    ll sizeOfAdjList = 0;

    int batch = 1;
    vector<double> timings;

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

            // Creating edge queue
            totalEdges = DIRECTED ? totalEdges : 2 * totalEdges;
            cudaMalloc((Node **)&nodeQueue, totalEdges * sizeof(Node));

            // Allocating space for adjacency list on device
            // cudaMalloc(&deviceAdjList, totalVertices * sizeof(Node *));
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

            // printEdgeList(edgeList);
            // ++count;
            // if(count == 2) break;

            ll noOfedges = edgeList.size();
            ll vertices = vertexCount.size();
            // ++count;
            // if(count == 3) {cout << vertices << endl; break;}

            ll *hvertexToIndexMap = (ll *)malloc((vertices) * sizeof(ll));
            ll *hindex = (ll *)malloc((vertices + 1) * sizeof(ll));
            ll *hheadVertex = (ll *)malloc(noOfedges * sizeof(ll));
            ll *hweights = (ll *)malloc(noOfedges * sizeof(ll));

            constructCSR(vertices, hindex, hheadVertex, hweights, DIRECTED, WEIGHTED, edgeList, vertexCount, hvertexToIndexMap);
            // printCSR(vertices, hindex, hheadVertex, hweights, noOfedges, hvertexToIndexMap);
            // ++count;
            // if(count == 3) break;

            // Copying CSR on GPU
            ll *dvertexToIndexMap;
            ll *dindex;
            ll *dheadVertex;
            ll *dweights;

            cudaMalloc(&dvertexToIndexMap, (ll)(vertices) * sizeof(ll));
            cudaMalloc(&dindex, (ll)(vertices + 1) * sizeof(ll));
            cudaMalloc(&dheadVertex, (ll)(noOfedges) * sizeof(ll));
            cudaMalloc(&dweights, (ll)(noOfedges) * sizeof(ll));

            cudaMemcpy(dvertexToIndexMap, hvertexToIndexMap, (ll)(vertices) * sizeof(ll), cudaMemcpyHostToDevice);
            cudaMemcpy(dindex, hindex, (ll)(vertices + 1) * sizeof(ll), cudaMemcpyHostToDevice);
            cudaMemcpy(dheadVertex, hheadVertex, (ll)(noOfedges) * sizeof(ll), cudaMemcpyHostToDevice);
            cudaMemcpy(dweights, hweights, (ll)(noOfedges) * sizeof(ll), cudaMemcpyHostToDevice);

            // /* Vertex Updates
            ll maxSizeNeeded = nearestPowerOf2(maxVertex);
            // cout << maxSizeNeeded << endl;

            if (sizeOfAdjList == 0)
            {
                cudaMalloc(&deviceAdjList, maxSizeNeeded * sizeof(Node *));
                sizeOfAdjList = maxSizeNeeded;
            }
            else if (maxVertex > sizeOfAdjList)
            {
                clock_t start, end;
                start = clock();
                Node **newdeviceAdjList;
                cudaMalloc(&newdeviceAdjList, maxSizeNeeded * sizeof(Node *));

                // copy old adjacency to new
                unsigned blocks = ceil((float)sizeOfAdjList / BLOCKSIZE);
                copyAdjacencyList<<<blocks, BLOCKSIZE>>>(deviceAdjList, newdeviceAdjList, sizeOfAdjList);
                cudaDeviceSynchronize();
                // cout << "here1" << endl;
                sizeOfAdjList = maxSizeNeeded;
                Node **temp;
                updateOldToNew<<<1, 1>>>(deviceAdjList, newdeviceAdjList, temp);
                // printAdjListKernel<<<1, 1>>>(sizeOfAdjList, deviceAdjList);
                cudaDeviceSynchronize();
                cudaFree(temp);
                end = clock();
                double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds
                totalTime += elapsedTime;
            }
            // */

            unsigned nblocks = ceil((float)vertices / BLOCKSIZE);
            // cout << nblocks <<endl; break;

            clock_t start, end;
            start = clock();
            generateAdjListParallel<<<nblocks, BLOCKSIZE>>>(vertices, dindex, dheadVertex, dweights, nodeQueue, deviceAdjList, dvertexToIndexMap, prevEdgeCount);
            cudaDeviceSynchronize();
            end = clock();
            double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds
            totalTime += elapsedTime;

            timings.emplace_back(elapsedTime);

            // printAdjListKernel<<<1, 1>>>(vertices, deviceAdjList);
            // cudaDeviceSynchronize();
            // ++count;
            // if(count == 3) break;

            // cout << "Time taken for Batch " << batch << ": " << elapsedTime << " ms" << endl;

            ++batch;
            prevEdgeCount += noOfedges;
            edgeList.clear();
            vertexCount.clear();
            cudaFree(dvertexToIndexMap);
            cudaFree(dindex);
            cudaFree(dheadVertex);
            cudaFree(dweights);
            free(hvertexToIndexMap);
            free(hindex);
            free(hheadVertex);
            free(hweights);
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

        ll *hvertexToIndexMap = (ll *)malloc((vertices) * sizeof(ll));
        ll *hindex = (ll *)malloc((vertices + 1) * sizeof(ll));
        ll *hheadVertex = (ll *)malloc(noOfedges * sizeof(ll));
        ll *hweights = (ll *)malloc(noOfedges * sizeof(ll));

        constructCSR(vertices, hindex, hheadVertex, hweights, DIRECTED, WEIGHTED, edgeList, vertexCount, hvertexToIndexMap);
        // printCSR(vertices, hindex, hheadVertex, hweights, noOfedges, hvertexToIndexMap);

        // Copying CSR on GPU
        ll *dvertexToIndexMap;
        ll *dindex;
        ll *dheadVertex;
        ll *dweights;

        cudaMalloc(&dvertexToIndexMap, (ll)(vertices) * sizeof(ll));
        cudaMalloc(&dindex, (ll)(vertices + 1) * sizeof(ll));
        cudaMalloc(&dheadVertex, (ll)(noOfedges) * sizeof(ll));
        cudaMalloc(&dweights, (ll)(noOfedges) * sizeof(ll));

        cudaMemcpy(dvertexToIndexMap, hvertexToIndexMap, (ll)(vertices) * sizeof(ll), cudaMemcpyHostToDevice);
        cudaMemcpy(dindex, hindex, (ll)(vertices + 1) * sizeof(ll), cudaMemcpyHostToDevice);
        cudaMemcpy(dheadVertex, hheadVertex, (ll)(noOfedges) * sizeof(ll), cudaMemcpyHostToDevice);
        cudaMemcpy(dweights, hweights, (ll)(noOfedges) * sizeof(ll), cudaMemcpyHostToDevice);

        // /* Vertex Updates
        ll maxSizeNeeded = nearestPowerOf2(maxVertex);
        // cout << maxSizeNeeded << endl;

        if (sizeOfAdjList == 0)
        {
            clock_t start, end;
            start = clock();
            cudaMalloc(&deviceAdjList, maxSizeNeeded * sizeof(Node *));
            sizeOfAdjList = maxSizeNeeded;
            end = clock();
            double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds
            totalTime += elapsedTime;
        }
        else if (maxVertex > sizeOfAdjList)
        {
            clock_t start, end;
            start = clock();
            Node **newdeviceAdjList;
            cudaMalloc(&newdeviceAdjList, maxSizeNeeded * sizeof(Node *));

            // copy old adjacency to new
            unsigned blocks = ceil((float)sizeOfAdjList / BLOCKSIZE);
            copyAdjacencyList<<<blocks, BLOCKSIZE>>>(deviceAdjList, newdeviceAdjList, sizeOfAdjList);
            cudaDeviceSynchronize();
            // cout << "here2" << endl;
            sizeOfAdjList = maxSizeNeeded;
            Node **temp;
            updateOldToNew<<<1, 1>>>(deviceAdjList, newdeviceAdjList, temp);
            // printAdjListKernel<<<1,1>>>(sizeOfAdjList, deviceAdjList);
            // cudaDeviceSynchronize();
            cudaFree(temp);
            end = clock();
            double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds
            totalTime += elapsedTime;
        }
        // */

        unsigned nblocks = ceil((float)vertices / BLOCKSIZE);

        clock_t start, end;
        start = clock();
        generateAdjListParallel<<<nblocks, BLOCKSIZE>>>(vertices, dindex, dheadVertex, dweights, nodeQueue, deviceAdjList, dvertexToIndexMap, prevEdgeCount);
        cudaDeviceSynchronize();
        end = clock();
        double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds
        totalTime += elapsedTime;
        timings.emplace_back(elapsedTime);
        // printAdjListKernel<<<1,1>>>(vertices, deviceAdjList);
        // cudaDeviceSynchronize();

        // cout << "Time taken for Batch " << batch << ": " << elapsedTime << " ms" << endl;

        edgeList.clear();
        vertexCount.clear();
        cudaFree(dvertexToIndexMap);
        cudaFree(dindex);
        cudaFree(dheadVertex);
        cudaFree(dweights);
        free(hvertexToIndexMap);
        free(hindex);
        free(hheadVertex);
        free(hweights);
    }

    file.close();
    cout << "Graph Built" << endl;
    // -------------------------------------- P A G E R A N K -----------------------------------------------

    ll *dindegree;
    ll *doutdegree;

    cudaMalloc(&dindegree, totalVertices * sizeof(ll));
    cudaMalloc(&doutdegree, totalVertices * sizeof(ll));

    calculateIndegree<<<1, 1>>>(totalVertices, dindegree, deviceAdjList);

    unsigned blocks = ceil((float)totalVertices / BLOCKSIZE);
    calculateOutdegree<<<blocks, BLOCKSIZE>>>(totalVertices, doutdegree, deviceAdjList);
    cudaDeviceSynchronize();

    // printIndegrees<<<1,1>>>(totalVertices, dindegree);
    // printOutdegrees<<<1,1>>>(totalVertices, doutdegree);
    // cudaDeviceSynchronize();
    
    computePagerank(totalVertices, doutdegree, deviceAdjList);

    // -----------------------CALCULATE MEMORY CONSUMPTION-------------------------------------------------------
    size_t freeMemAfter = calculateMemoryConsumption();
    std::cout << "Free GPU memory after kernel launch: " << freeMemAfter << " bytes" << std::endl;

    // Calculate memory used by the kernel
    size_t memoryUsed = freeMemBefore - freeMemAfter;
    std::cout << "Memory used by the kernel: " << memoryUsed << " bytes" << std::endl;

    return 0;
}

void computePagerank(ll totalVertices, ll *doutdegrees, Node **adjList)
{   
    double time = 0;
    double elapsedTime = 0;
    unsigned blocks = ceil((float)totalVertices / BLOCKSIZE);

    double *prevPageRanks;
    cudaMalloc(&prevPageRanks, (double)totalVertices * sizeof(double));
    clock_t start, end;
    start = clock();
    pagerankInit<<<blocks, BLOCKSIZE>>>(totalVertices, prevPageRanks);
    end = clock();
    elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    time += elapsedTime;
    // printPageranks<<<1, 1>>>(totalVertices, prevPageRanks);
    // cudaDeviceSynchronize();

    double *currPageRanks;
    cudaMalloc(&currPageRanks, (double)totalVertices * sizeof(double));

    int last = 0;

    start = clock();
    for (ll itr = 0; itr < MAX_ITRS; ++itr)
    {
        if (itr & 1)
        {
            pagerank<<<blocks, BLOCKSIZE>>>(totalVertices, doutdegrees, currPageRanks, prevPageRanks, adjList);
            cudaDeviceSynchronize();
            last = 1;
        }
        else
        {
            pagerank<<<blocks, BLOCKSIZE>>>(totalVertices, doutdegrees, prevPageRanks, currPageRanks, adjList);
            cudaDeviceSynchronize();
            last = 0;
        }
    }
    end = clock();
    elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;;
    time += elapsedTime;
    if (last)
        printPageranks<<<1, 1>>>(totalVertices, prevPageRanks);
    else
        printPageranks<<<1, 1>>>(totalVertices, currPageRanks);
    cudaDeviceSynchronize();

    cout << "Total time for page Rank: " << time << endl;
}

size_t calculateMemoryConsumption() {
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    return freeMem;
}