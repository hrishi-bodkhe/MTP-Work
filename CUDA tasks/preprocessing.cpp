#include "preprocessing.h"

// Function to create a new node
Node *createNode(int v, int weighted, int wt)
{
    Node *node = (Node *)malloc(sizeof(Node));
    node->vertex = v;
    node->next = NULL;

    if (weighted == 1)
        node->wt = wt;
    else
        wt = 1;

    return node;
}

bool comp_Edges_and_dest(Edge &a, Edge &b)
{
    return a.src == b.src ? a.dest < b.dest : a.src < b.src;
}

bool comp_Edges(Edge &a, Edge &b)
{
    return a.src < b.src;
}

void constructCSR(ll &vertices, ll *index, ll *headvertex, ll *weights, int directed, int weighted, vector<Edge> &edgeList, map<ll, ll> vertexCount, ll* vertexToIndexMap)
{
    ll edges = edgeList.size();

    // constructing indices for index array
    index[0] = 0;
    int i = 1;
    for(auto& p: vertexCount){
        index[i] = p.second;
        ++i;
    }

    i = 0;
    for(auto& p: vertexCount){
        vertexToIndexMap[i++] = p.first;
    }

    for (ll j = 1; j < vertices + 1; ++j)
        index[j] += index[j - 1];

    // constructing the headvertex and weights array
    for (ll j = 0; j < edges; ++j)
    {
        Edge e = edgeList[j];
        headvertex[j] = e.dest;
        weights[j] = e.wt;
    }
}

void printCSR(ll &vertices, ll *index, ll *headvertex, ll *weights, ll &edges, ll *vertexToIndexMap)
{
    cout << "----------------------------------STARTED PRINTING CSR---------------------------------" << endl;
    cout << "Vertex Mapping: ";
    for(int i = 0; i < vertices; ++i)
        cout << vertexToIndexMap[i] << ' ';
    cout << endl;

    cout << "Index: ";
    for(int i = 0; i < vertices + 1; ++i)
        cout << index[i] << ' ';
    cout << endl;

    cout << "Head Vertex: ";
    for (int i = 0; i < edges; ++i)
        cout << headvertex[i] << ' ';
    cout << endl;

    cout << "Weights: ";
    for (int i = 0; i < edges; ++i)
        cout << weights[i] << ' ';
    cout << endl;
}

void printEdgeList(vector<Edge> &edgeList)
{
    cout << "-------------------------------STARTED PRINTING EDGELIST------------------------------" << endl;
    for (Edge e : edgeList)
        cout << e.src << ' ' << e.dest << ' ' << e.wt << endl;
}