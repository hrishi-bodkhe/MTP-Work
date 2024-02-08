#include<cuda.h>
#include "kernels.h"

int main(){
    srand(static_cast<unsigned int>(time(0)));
    int sortedOption = 0;
    int directed = 0;
    int weighted = 0;
    int algoChoice;
    string filename = "";
    string filenameforCorrection = "";

    if(!takeChoices(directed, weighted, algoChoice, filename, sortedOption, filenameforCorrection)) return 0;

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
//        if(e.wt != 1) cout << e.wt << ' ';
        if(e.wt < 0) e.wt = 1;

        edgeList.push_back(e);

        degrees[e.src] += 1;
        maxDegree = max(degrees[e.src], maxDegree);
        avgDegree += 1;

        if (!directed)
        {
            e.src = dest - 1;
            e.dest = src - 1;
            e.wt = weighted ? wt : 1;
            if(e.wt < 0) e.wt = 1;

            edgeList.push_back(e);

            degrees[e.src] += 1;
            maxDegree = max(degrees[e.src], maxDegree);
            avgDegree += 1;
        }
    }
    cout << endl;

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

    if(algoChoice == 9) triangleCount(totalVertices, totalEdges, dindex, dheadVertex, filenameforCorrection);

    size_t finalFreeMemory;
    cudaMemGetInfo(&finalFreeMemory, &totalMemory);
    size_t consumedMemory = initialFreeMemory - finalFreeMemory;
    cout << "Final Free Memory: " << finalFreeMemory / (1024 * 1024 * 1024) << " GB" << endl;
    cout << "Consumed Memory: " << consumedMemory / (1024 * 1024) << " MB" << endl;

    return 0;
}