#include<cuda.h>
#include "preprocessing.h"

void ssspVertexCentric(ll totalVertices, ll *dindex, ll *dheadvertex, ll *dweights);

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

__global__ void printDist(ll vertices, ll *dist){
    for(ll u = 0; u < 10; ++u) printf("%lld ", dist[u]);
    printf("\n");
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
    ll *hheadvertex;
    ll *hweights;

    hindex = (ll *)malloc((totalVertices + 1) * sizeof(ll));
    hheadvertex = (ll *)malloc(totalEdges * sizeof(ll));
    hweights = (ll *)malloc(totalEdges * sizeof(ll));

    size_t initialFreeMemory, totalMemory;
    cudaMemGetInfo(&initialFreeMemory, &totalMemory);
    cout << "Initial Free Memory: " << initialFreeMemory / (1024 * 1024 * 1024) << " GB" << endl;

    buildCSR(totalVertices, totalEdges, edgeList, hindex, hheadvertex, hweights, degrees);

    ll *dindex;
    ll *dheadVertex;
    ll *dweights;

    cudaMalloc(&dindex, (ll)(totalVertices + 1) * sizeof(ll));
    cudaMalloc(&dheadVertex, (ll)(totalEdges) * sizeof(ll));
    cudaMalloc(&dweights, (ll)(totalEdges) * sizeof(ll));

    cudaMemcpy(dindex, hindex, (ll)(totalVertices + 1) * sizeof(ll), cudaMemcpyHostToDevice);
    cudaMemcpy(dheadVertex, hheadvertex, (ll)(totalEdges) * sizeof(ll), cudaMemcpyHostToDevice);
    cudaMemcpy(dweights, hweights, (ll)(totalEdges) * sizeof(ll), cudaMemcpyHostToDevice);

    cout << endl;
    cout << "Graph Built" << endl;
    cout << endl;

    ssspVertexCentric(totalVertices, dindex, dheadVertex, dweights);

    size_t finalFreeMemory;
    cudaMemGetInfo(&finalFreeMemory, &totalMemory);
    size_t consumedMemory = initialFreeMemory - finalFreeMemory;
    cout << "Final Free Memory: " << finalFreeMemory / (1024 * 1024 * 1024) << " GB" << endl;
    cout << "Consumed Memory: " << consumedMemory / (1024 * 1024) << " MB" << endl;

    return 0;
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