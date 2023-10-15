#ifndef PREPROCESSING_H
#define PREPROCESSING_H
#include <bits/stdc++.h>
#include <iostream>
#define ll long long
#define ld long double
#define BLOCKSIZE 1024
#define WEIGHTED 0
#define DIRECTED 0
#define MAX_ITRS 10
#define dampingFactor 0.85
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

#endif