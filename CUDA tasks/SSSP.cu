#include<cuda.h>
#include "kernels.h"

int main(){
    srand(static_cast<unsigned int>(time(0)));
    int sortedOption = 0;
    int directed = 0;
    int weighted = 0;
    int algoChoice;
    string filename = "";

    if(!takeChoices(directed, weighted, algoChoice, filename, sortedOption)) return 0;

    ll totalVertices;
    ll temp1, totalEdges; // for skipping the first line vertices, edges
    vector<Edge> edgeList;
    string line;

    ll batchSize = INT_MAX;
    bool skipLineOne = true;

    unordered_map<ll, ll> degrees;

    ll maxDegree = INT_MIN;
    ll avgDegree = 0;

    ifstream file(filename);

    file.is_open();

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

    if(algoChoice == 1 || algoChoice == 3 || algoChoice == 4 || algoChoice == 5 || algoChoice == 6) hindex = (ll *)malloc((totalVertices + 1) * sizeof(ll));
    else if(algoChoice == 2) hsrc = (ll *)malloc((totalEdges) * sizeof (ll));
    hheadvertex = (ll *)malloc(totalEdges * sizeof(ll));
    hweights = (ll *)malloc(totalEdges * sizeof(ll));

    size_t initialFreeMemory, totalMemory;
    cudaMemGetInfo(&initialFreeMemory, &totalMemory);
    cout << "Initial Free Memory: " << initialFreeMemory / (1024 * 1024 * 1024) << " GB" << endl;

    if(algoChoice == 1 || algoChoice == 3 || algoChoice == 4 || algoChoice == 5 || algoChoice == 6) buildCSR(totalVertices, totalEdges, edgeList, hindex, hheadvertex, hweights, degrees);
    else if(algoChoice == 2) buildCOO(totalEdges, edgeList, hsrc, hheadvertex, hweights);

//    for(ll i = 0; i <= totalVertices; ++i) cout << hindex[i] << ' ';
//    cout << "-------" << endl;
//    for(ll i = 0; i < totalEdges; ++i) cout << hheadvertex[i] << ' ';
//    cout << "-------" << endl;

    ll *dindex;
    ll *dsrc;
    ll *dheadVertex;
    ll *dweights;

    if(algoChoice == 1 || algoChoice == 3 || algoChoice == 4 || algoChoice == 5 || algoChoice == 6) cudaMalloc(&dindex, (ll)(totalVertices + 1) * sizeof(ll));
    else if(algoChoice == 2) cudaMalloc(&dsrc, (ll)(totalEdges) * sizeof(ll));
    cudaMalloc(&dheadVertex, (ll)(totalEdges) * sizeof(ll));
    cudaMalloc(&dweights, (ll)(totalEdges) * sizeof(ll));

    if(algoChoice == 1 || algoChoice == 3 || algoChoice == 4 || algoChoice == 5 || algoChoice == 6) cudaMemcpy(dindex, hindex, (ll)(totalVertices + 1) * sizeof(ll), cudaMemcpyHostToDevice);
    else if(algoChoice == 2) cudaMemcpy(dsrc, hsrc, (ll)(totalEdges) * sizeof(ll), cudaMemcpyHostToDevice);
    cudaMemcpy(dheadVertex, hheadvertex, (ll)(totalEdges) * sizeof(ll), cudaMemcpyHostToDevice);
    cudaMemcpy(dweights, hweights, (ll)(totalEdges) * sizeof(ll), cudaMemcpyHostToDevice);

    cout << endl;
    cout << "Graph Built" << endl;
    cout << endl;

    ll src = 0;

    if(algoChoice == 1) ssspVertexCentric(totalVertices, dindex, dheadVertex, dweights, src);
    else if(algoChoice == 2) ssspEdgeCentric(totalVertices, totalEdges, dsrc, dheadVertex, dweights, src);
    else if(algoChoice == 3) ssspWorklist(totalVertices, totalEdges, dindex, dheadVertex, dweights, src);
    else if(algoChoice == 4) ssspWorklist2(totalVertices, totalEdges, dindex, dheadVertex, dweights, src);
    else if(algoChoice == 5) ssspBalancedWorklist(totalVertices, totalEdges, dindex, dheadVertex, dweights, src);
    else if(algoChoice == 6) ssspEdgeWorklistCentric(totalVertices, totalEdges, dindex, dheadVertex, dweights, src);
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