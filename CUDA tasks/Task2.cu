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

int main()
{
    int count = 0;
    int sortedOption;
    cout << "Do you want the edge list to be in sorted order? Enter 1 for Yes or 0 for No. ";
    cin >> sortedOption;
    // cout << sortedOption << endl;

    string file1 = "inf-luxembourg_osm.mtx";
    string file2 = "chesapeake.mtx";
    string file3 = "delaunay_n17.mtx";
    string file4 = "kron_g500-logn16.mtx";
    string file5 = "rgg_n_2_16_s0.mtx";
    string file6 = "kron_g500-logn21.mtx";

    string mtxFilePath = file1;

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
    ll batchSize = 10;
    bool skipLineOne = true;
    ll prevEdgeCount = 0;

    // Keep count of vertices. Track Max Vertex
    map<ll, ll> vertexCount;
    ll maxVertex = 0;

    // Defining Edge queue
    Node *nodeQueue;

    // Defining deviceAdjList;
    Node **deviceAdjList;

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
            cudaMalloc(&deviceAdjList, totalVertices * sizeof(Node *));
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

            unsigned nblocks = ceil((float)vertices / BLOCKSIZE);
            // cout << nblocks <<endl; break;

            clock_t start, end;
            start = clock();
            generateAdjListParallel<<<nblocks, BLOCKSIZE>>>(vertices, dindex, dheadVertex, dweights, nodeQueue, deviceAdjList, dvertexToIndexMap, prevEdgeCount);
            cudaDeviceSynchronize();
            end = clock();
            double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds

            // printAdjListKernel<<<1, 1>>>(totalVertices, deviceAdjList);
            // cudaDeviceSynchronize();
            // ++count;
            // if(count == 3) break;

            // cout << "Time taken is: " << elapsedTime << " ms" << endl;
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

        unsigned nblocks = ceil((float)vertices / BLOCKSIZE);

        clock_t start, end;
        start = clock();
        generateAdjListParallel<<<nblocks, BLOCKSIZE>>>(vertices, dindex, dheadVertex, dweights, nodeQueue, deviceAdjList, dvertexToIndexMap, prevEdgeCount);
        cudaDeviceSynchronize();
        end = clock();
        double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds
        // printAdjListKernel<<<1,1>>>(totalVertices, deviceAdjList);
        cudaDeviceSynchronize();

        // cout << "Time taken is: " << elapsedTime << " ms" << endl;
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

    printAdjListKernel<<<1,1>>>(totalVertices, deviceAdjList);
    cudaDeviceSynchronize();
    return 0;
}