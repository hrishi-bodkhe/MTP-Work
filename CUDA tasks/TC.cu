#include<cuda.h>
#include "kernels.h"
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

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

//    filename = "input.txt";
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

//        if(e.src == e.dest) {
//            totalEdges -= 2;
//            continue;
//        }
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

    clock_t start = clock();
    if(sortedOption) sort(edgeList.begin(), edgeList.end(), comp_Edges_and_dest);
    clock_t end = clock();
    double elapsed_time = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    cout << "Sorting Time: " << elapsed_time << " ms" << endl;

//    clock_t start = clock();
//    thrust::host_vector<Edge> h_edges = edgeList; // Copy edge list to host vector
//    thrust::device_vector<Edge> d_edges = h_edges; // Copy host vector to device vector
//
//    thrust::sort(d_edges.begin(), d_edges.end()); // Sort on the GPU
//
//    h_edges = d_edges; // Copy sorted list back to host
//    edgeList.resize(h_edges.size());
////    edgeList = h_edges; // Copy back to original edge list
//    std::copy(h_edges.begin(), h_edges.end(), edgeList.begin());
//    clock_t end = clock();
//    double elapsed_time = 1000.0 * (end - start) / CLOCKS_PER_SEC;
//    cout << "Sorting Time on GPU: " << elapsed_time << " ms" << endl;

    ll duplicates = 0;
    for(ll i = 0; i < edgeList.size() - 1; ++i){
        Edge e1 = edgeList[i];
        Edge e2 = edgeList[i + 1];

        if(e1.src == e2.src && e1.dest == e2.dest){
            ++duplicates;
//            cout << "Duplicate Found" << endl;
        }
    }



    ll selfLoops = 0;
    for(ll i = 0; i < edgeList.size(); ++i){
        Edge e = edgeList[i];
        if(e.src == e.dest) {
            ++selfLoops;
//            cout << "Self Loop for vertex: " << e.src << endl;
        }
    }

    cout << "Total Self Loops: " << selfLoops << endl;
    cout << endl << "Duplicates before Removal: " << duplicates << endl;
/*
    set<Edge, compareForDuplicateRemoval> uniqueEdges(edgeList.begin(), edgeList.end());
    vector<Edge> uniqueEdgeList(uniqueEdges.begin(), uniqueEdges.end());

    totalEdges = uniqueEdgeList.size();

    duplicates = 0;
    for(ll i = 0; i < uniqueEdgeList.size() - 1; ++i){
        Edge e1 = uniqueEdgeList[i];
        Edge e2 = uniqueEdgeList[i + 1];

        if(e1.src == e2.src && e1.dest == e2.dest){
            ++duplicates;
//            cout << "Duplicate Found" << endl;
        }
    }

    cout << endl << "Duplicates after Removal: " << duplicates << endl;
*/
    ll *hindex;
    ll *hheadvertex;
    ll *hweights;
    ll *hsrc;

    hindex = (ll *)malloc((totalVertices + 1) * sizeof(ll));
    hheadvertex = (ll *)malloc(totalEdges * sizeof(ll));
    hweights = (ll *)malloc(totalEdges * sizeof(ll));
    if(algoChoice == 12) hsrc = (ll *)malloc((totalEdges) * sizeof (ll));

    size_t initialFreeMemory, totalMemory;
    cudaMemGetInfo(&initialFreeMemory, &totalMemory);
    cout << "Initial Free Memory: " << initialFreeMemory / (1024 * 1024 * 1024) << " GB" << endl;

    buildCSR(totalVertices, totalEdges, edgeList, hindex, hheadvertex, hweights, degrees);
    if(algoChoice == 12) buildCOO(totalEdges, edgeList, hsrc, hheadvertex, hweights);

    ll *dindex;
    ll *dsrc;
    ll *dheadVertex;
    ll *dweights;

    cudaMalloc(&dindex, (ll)(totalVertices + 1) * sizeof(ll));
    cudaMalloc(&dheadVertex, (ll)(totalEdges) * sizeof(ll));
    cudaMalloc(&dweights, (ll)(totalEdges) * sizeof(ll));
    if (algoChoice == 12) cudaMalloc(&dsrc, (ll)(totalEdges) * sizeof(ll));

    cudaMemcpy(dindex, hindex, (ll)(totalVertices + 1) * sizeof(ll), cudaMemcpyHostToDevice);
    cudaMemcpy(dheadVertex, hheadvertex, (ll)(totalEdges) * sizeof(ll), cudaMemcpyHostToDevice);
    cudaMemcpy(dweights, hweights, (ll)(totalEdges) * sizeof(ll), cudaMemcpyHostToDevice);
    if (algoChoice == 12) cudaMemcpy(dsrc, hsrc, (ll)(totalEdges) * sizeof(ll), cudaMemcpyHostToDevice);

    cout << endl;
    cout << "Graph Built" << endl;
    cout << endl;

//    cout << "Index: ";
//    for(int i = 0; i < totalVertices + 1; ++i)
//        cout << hindex[i] << ' ';
//    cout << endl;
//
//    cout << "Head Vertex: ";
//    for (int i = 0; i < totalEdges; ++i)
//        cout << hheadvertex[i] << ' ';
//    cout << endl;
    if(algoChoice == 6) ssspEdgeWorklistCentric(totalVertices, totalEdges, dindex, dheadVertex, dweights, 0, filenameforCorrection);
    if(algoChoice == 9) triangleCount(totalVertices, totalEdges, dindex, dheadVertex, filenameforCorrection);
    else if(algoChoice == 10) triangleCountEdgeCentric(totalVertices, totalEdges, dindex, dheadVertex, filenameforCorrection);
    else if(algoChoice == 11) triangleCountSortedVertexCentric(totalVertices, dindex, dheadVertex, filenameforCorrection);
    else if(algoChoice == 12) triangleCountEdgeCentricCOO(totalVertices, totalEdges, dindex, dheadVertex, dsrc, filenameforCorrection);

    size_t finalFreeMemory;
    cudaMemGetInfo(&finalFreeMemory, &totalMemory);
    size_t consumedMemory = initialFreeMemory - finalFreeMemory;
    cout << "Final Free Memory: " << finalFreeMemory / (1024 * 1024 * 1024) << " GB" << endl;
    cout << "Consumed Memory: " << consumedMemory / (1024 * 1024) << " MB" << endl;

    return 0;
}