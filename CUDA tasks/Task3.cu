#include <cuda.h>
#include "preprocessing.h"

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

__global__ void batchDeletion(ll vertices, ll *index, ll *headVertex, ll *weights, Node **adjList, ll *vertexToIndexMap)
{
    unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u >= vertices)
        return;

    ll u_data = vertexToIndexMap[u];

    ll startIdx = index[u];
    ll endIdx = index[u + 1];

    for (ll idx = startIdx; idx < endIdx; ++idx)
    {
        ll v = headVertex[idx];
        ll wt = weights[idx];

        Node *temp = adjList[u_data];

        while (temp)
        {
            if (temp->vertex == v && temp->wt == wt)
            {
                temp->vertex = -1;
                temp->wt = -1;
                break;
            }
            temp = temp->next;
        }
    }
}

__global__ void edgeCentricBatchDeletion(ll edges, Edge *dedgeList, Node **adjList)
{
    unsigned int e = blockIdx.x * blockDim.x + threadIdx.x;

    if (e >= edges)
        return;

    Edge edge = dedgeList[e];

    ll u = edge.src;
    ll v = edge.dest;
    ll wt = edge.wt;

    Node *temp = adjList[u];
    while (temp)
    {
        if (temp->vertex == v && temp->wt == wt)
        {
            temp->vertex = -1;
            temp->wt = -1;
            return;
        }

        temp = temp->next;
    }
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

    string mtxFilePath = file5;
    double totalTime = 0.0;
    double totalDeletionTime = 0.0;
    double totalDeletionTimeD1 = 0.0;

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

            // /*
            // BATCH DELETION==========================================================================================================================================================================================

            clock_t startD, endD;
            startD = clock();
            batchDeletion<<<nblocks, BLOCKSIZE>>>(vertices, dindex, dheadVertex, dweights, deviceAdjList, dvertexToIndexMap);
            cudaDeviceSynchronize();
            endD = clock();
            double elapsedTimeD = (double)(endD - startD) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds
            // cout << elapsedTimeD << ' ';
            totalDeletionTime += elapsedTimeD;
            // */

            /*
            // Convert vector edgelist to array================================================
            ll sz = edgeList.size();
            Edge *hedgeList = (Edge *)malloc((sz) * sizeof(Edge));
            for (ll k = 0; k < sz; ++k)
            {
                hedgeList[k] = edgeList[k];
            }

            Edge *dedgeList;
            cudaMalloc(&dedgeList, sz * sizeof(Edge));
            cudaMemcpy(dedgeList, hedgeList, sz * sizeof(Edge), cudaMemcpyHostToDevice);
            */

            /*
            // EDGE CENTRIC BATCH DELETION
            ll edges = edgeList.size();
            unsigned nblocks1 = ceil((float)edges / BLOCKSIZE);

            clock_t startD1, endD1;
            startD1 = clock();
            edgeCentricBatchDeletion<<<nblocks1, BLOCKSIZE>>>(edges, dedgeList, deviceAdjList);
            cudaDeviceSynchronize();
            endD1 = clock();
            double elapsedTimeD1 = (double)(endD1 - startD1) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds
            // cout << elapsedTimeD1 << ' ';
            totalDeletionTimeD1 += elapsedTimeD1;
            */

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
            // cudaFree(dedgeList);
            free(hvertexToIndexMap);
            free(hindex);
            free(hheadVertex);
            free(hweights);
            // free(hedgeList);
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

        // /*
        // BATCH DELETION==========================================================================================================================================================================================

        clock_t startD, endD;
        startD = clock();
        batchDeletion<<<nblocks, BLOCKSIZE>>>(vertices, dindex, dheadVertex, dweights, deviceAdjList, dvertexToIndexMap);
        cudaDeviceSynchronize();
        endD = clock();
        double elapsedTimeD = (double)(endD - startD) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds
        // cout << elapsedTimeD << ' ';
        totalDeletionTime += elapsedTimeD;
        // */

        /*
        // Convert vector edgelist to array================================================
        ll sz = edgeList.size();
        Edge *hedgeList = (Edge *)malloc((sz) * sizeof(Edge));
        for (ll k = 0; k < sz; ++k)
        {
            hedgeList[k] = edgeList[k];
        }

        Edge *dedgeList;
        cudaMalloc(&dedgeList, sz * sizeof(Edge));
        cudaMemcpy(dedgeList, hedgeList, sz * sizeof(Edge), cudaMemcpyHostToDevice);

        // EDGE CENTRIC BATCH DELETION
        ll edges = edgeList.size();
        unsigned nblocks1 = ceil((float)edges / BLOCKSIZE);

        clock_t startD1, endD1;
        startD1 = clock();
        edgeCentricBatchDeletion<<<nblocks1, BLOCKSIZE>>>(edges, dedgeList, deviceAdjList);
        cudaDeviceSynchronize();
        endD1 = clock();
        double elapsedTimeD1 = (double)(endD1 - startD1) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds
        // cout << elapsedTimeD1 << ' ';
        totalDeletionTimeD1 += elapsedTimeD1;
        */

        edgeList.clear();
        vertexCount.clear();
        cudaFree(dvertexToIndexMap);
        cudaFree(dindex);
        cudaFree(dheadVertex);
        cudaFree(dweights);
        // cudaFree(dedgeList);
        free(hvertexToIndexMap);
        free(hindex);
        free(hheadVertex);
        free(hweights);
        // free(hedgeList);
    }

    file.close();

    printAdjListKernel<<<1, 1>>>(totalVertices, deviceAdjList);
    cudaDeviceSynchronize();

    // printTimings(timings);

    double avgTime = (double)totalTime / batch;
    // cout << "Total Time is: " << totalTime << endl;
    // cout << "Average Time is: " << avgTime << endl;

    cout << endl;
    cout << "Total Time for Vertex Centric Deletion is: " << totalDeletionTime << endl;
    double avgDeleteTime = (double)totalDeletionTime / batch;
    cout << "Average Time for vertex Centric Deletion is: " << avgDeleteTime << endl;

    // cout << endl;
    // cout << "Total Time for Edge Centric Deletion is: " << totalDeletionTimeD1 << endl;
    // double avgDeleteTimeD1 = (double)totalDeletionTimeD1 / batch;
    // cout << "Average Time for Edge Centric Deletion is: " << avgDeleteTimeD1 << endl;
    return 0;
}