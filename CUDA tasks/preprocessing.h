#ifndef PREPROCESSING_H
#define PREPROCESSING_H
#include <bits/stdc++.h>
#include <iostream>
#define ll long long
#define ld long double
#define BLOCKSIZE 1024
#define WEIGHTED 1
#define DIRECTED 0
#define MAX_ITRS 10
#define dampingFactor 0.85
//#define INT_MAX 1e6
using namespace std;

struct Edge
{
    ll src;  // source
    ll dest; // destination
    ll wt;   // weight
};

// Node structure for adjacency list
typedef struct Node
{
    ll vertex;
    ll wt;
    struct Node *next;
} Node;

int takeChoices(int& directed, int& weighted, int& algoChoice, string& filename, int& sortedOption, string& filenameforCorrection);

void readFile(string path, vector<Edge> &edgeList, ll &vertices, ll &edges, int &directed, int &weighted);

void printEdgeList(vector<Edge> &edgeList);

bool comp_Edges_and_dest(Edge &a, Edge &b);

bool comp_Edges(Edge &a, Edge &b);

void constructCSR(ll &vertices, ll *index, ll *headvertex, ll *weights, int directed, int weighted, vector<Edge> &edgeList, map<ll, ll> vertexCount, ll* vertexToIndexMap);

void printCSR(ll &vertices, ll *index, ll *headvertex, ll *weights, ll &edges, ll *vertexToIndexMap);

void printEdgeList(vector<Edge> &edgeList);

ll nearestPowerOf2(ll value);

void printTimings(vector<double>& timings);

void computePagerank(ll totalVertices, ll *doutdegrees, Node **adjList);

size_t calculateMemoryConsumption();

void computePRSerial(ll vertices, ll *dindex, ll *dheadVertex);

void computePRParallel(ll vertices, ll *dindex, ll *dheadVertex);

void constructSrcCSR(ll &vertices, ll *index, ll *sources, ll *headvertex, ll *weights, int directed, int weighted, vector<Edge> &edgeList, map<ll, ll> vertexCount, ll* vertexToIndexMap);

void ssspWorklist(ll totalVertices, ll totalEdges, ll *dindex, ll *dheadvertex, ll *dweights, ll src, string &filenameforCorrection);

void ssspWorklist2(ll totalVertices, ll totalEdges, ll *dindex, ll *dheadvertex, ll *dweights, ll src, string &filenameforCorrection);

void ssspBalancedWorklist(ll totalVertices, ll totalEdges, ll *dindex, ll *dheadVertex, ll *dweights, ll src, string &filenameforCorrection);

void ssspVertexCentric(ll totalVertices, ll *dindex, ll *dheadvertex, ll *dweights, ll src, string &filenameforCorrection);

void ssspEdgeCentric(ll totalVertices, ll totalEdges, ll *src, ll *dest, ll *weights, ll srcVertex, string &filenameforCorrection);

void buildCSR(ll vertices, ll edges, vector<Edge>& edgelist, ll *index, ll *headvertex, ll *weights, unordered_map<ll, ll>& degrees);

void buildCOO(ll edges, vector<Edge>& edgelist, ll *src, ll *dest, ll *weights);

void ssspSerial(ll totalVertices, ll *index, ll *headvertex, ll *weights, ll *dist, ll src);

void printssspCpu(ll totalVertices, ll *dist);

void ssspVertexCentricCorrectness(ll totalVertices, ll *dindex, ll *dheadvertex, ll *dweights, ll srcVertex, ll *wdist);

void ssspEdgeWorklistCentric(ll totalvertices, ll totalEdges, ll *csr_offsets, ll *csr_edges, ll *csr_weights, ll srcVertex, string &filenameforCorrection);

void ssspBucketWorklist(ll totalvertices, ll totaledges, ll *csr_offsets, ll *csr_edges, ll *csr_weights, ll srcVertex, string &filenameforCorrection);

void ssspBucketWorklist2(ll totalvertices, ll totaledges, ll *csr_offsets, ll *csr_edges, ll *csr_weights, ll srcVertex, string &filenameforCorrection);

void checkSSSPCorrectnessWithSlabGraph(ll *wdist, string &filename);

void checkTCCorrectnessWithSlabGraph(unsigned int *wdist, string &filename);

void triangleCount(ll totalvertices, ll totaledges, ll *csr_offsets, ll *csr_edges, string &filenameforCorrection);

void triangleCountEdgeCentric(ll totalvertices, ll totaledges, ll *csr_offsets, ll *csr_edges,  string &filenameforCorrection);

void triangleCountSortedVertexCentric(ll totalvertices, ll *csr_offsets, ll *csr_edges,  string &filenameforCorrection);

#endif