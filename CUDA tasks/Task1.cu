#include <iostream>
#include <bits/stdc++.h>
#include <cuda.h>
#include <ctime>
#define ll long long
#define BLOCKSIZE 1024
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

// Graph structure with an array of adjacency lists
typedef struct Graph
{
    int numVertices;
    Node **adjList;
} Graph;

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

// Function to create a graph with a given number of vertices
Graph *createGraph(int vertices)
{
    Graph *graph = (Graph *)malloc(sizeof(Graph));
    graph->numVertices = vertices;

    graph->adjList = (Node **)malloc(vertices * sizeof(Node *));

    for (int i = 0; i < vertices; ++i)
        graph->adjList[i] = NULL;

    return graph;
}

void constructCSR(ll &vertices, vector<ll> &index, vector<ll> &headvertex, vector<ll> &weights, int directed, int weighted, vector<Edge> &edgeList);
void readFile(string path, vector<Edge> &edgeList, ll &vertices, ll &edges, int &directed, int &weighted);
bool comp_Edges(Edge &a, Edge &b);
bool comp_Edges_and_dest(Edge &a, Edge &b);

__global__ void printAdjListKernel(ll vertices, Node **adjList){
    for(ll u = 0; u < vertices; ++u){
        printf("%ld: ", u);

        Node* temp = adjList[u];

        // if(temp) printf("%ld ", temp->vertex);

        while(temp){
            printf("(%ld, %ld), ", temp->vertex, temp->wt);
            temp = temp->next;
        }

        printf("\n");
    }
}

__global__ void generateAdjListParallel(ll vertices, ll *index, ll *headVertex, ll *weights, Node **nodequeue, Node **adjList)
{
    unsigned int u = blockIdx.x * blockDim.x +  threadIdx.x;

    if (u >= vertices)
        return;

    adjList[u] = NULL;

    ll startIdx = index[u];
    ll endIdx = index[u + 1];

    for (ll idx = startIdx; idx < endIdx; ++idx)
    {
        ll v = headVertex[idx];
        ll wt = weights[idx];

        Node *node = nodequeue[idx];

        // if(node == NULL) printf("%ld ",qIndex);
        // else printf("no ");

        node->vertex = v;
        node->wt = wt;
        node->next = NULL;

        Node *temp = adjList[u];

        if (!temp)
            adjList[u] = node;
        else
        {
            node->next = temp;
            adjList[u] = node;
        }
    }

    
}

__global__ void generateAdjList(ll vertices, ll *index, ll *headVertex, ll *weights, Node **nodequeue, Node **adjList, ll edges)
{
    for (ll i = 0; i < vertices; ++i)
    {
        adjList[i] = NULL;
    }
    ll qIndex = 0;

    for (ll u = 0; u < vertices; ++u)
    {
        ll startIdx = index[u];
        ll endIdx = index[u + 1];

        for (ll idx = startIdx; idx < endIdx; ++idx)
        {
            ll v = headVertex[idx];
            ll wt = weights[idx];
            // printf("%ld ", qIndex);
            Node *node = nodequeue[qIndex];
            ++qIndex;

            // if(node == NULL) printf("%ld ",qIndex);
            // else printf("no ");

            node->vertex = v;
            node->wt = wt;
            node->next = NULL;

            Node *temp = adjList[u];

            if (!temp)
                adjList[u] = node;
            else
            {
                node->next = temp;
                adjList[u] = node;
            }
            // printf("%d ", qIndex);

            // qIndex = qIndex + 1;
        }
    }
}

__global__ void allocate(ll i, Node **nodequeue, Node *node)
{
    nodequeue[i] = node;
}

int main()
{

    int sortedOption;
    cout << "Do you want the edge list to be in sorted order? Enter 1 for Yes or 0 for No. ";
    cin >> sortedOption;
    // cout << sortedOption << endl;

    // read the file and store it in a container
    ll vertices, edges;
    int directed, weighted;

    vector<Edge> edgeList;

    string inputPath = "input.txt";

    readFile(inputPath, edgeList, vertices, edges, directed, weighted);

    if (sortedOption)
        sort(edgeList.begin(), edgeList.end(), comp_Edges_and_dest);
    else
        sort(edgeList.begin(), edgeList.end(), comp_Edges);

    vector<ll> index(vertices + 1, 0);
    vector<ll> headvertex;
    vector<ll> weights;

    constructCSR(vertices, index, headvertex, weights, directed, weighted, edgeList);

    ll hindex[vertices + 1];
    ll hheadVertex[headvertex.size()];
    ll hweights[weights.size()];

    for (ll i = 0; i < vertices + 1; ++i)
    {
        hindex[i] = index[i];
    }
    for (ll i = 0; i < headvertex.size(); ++i)
    {
        hheadVertex[i] = headvertex[i];
        hweights[i] = weights[i];
    }
    ll noOfedges = headvertex.size();

    // Copying CSR on GPU
    ll *dindex;
    ll *dheadVertex;
    ll *dweights;

    cudaMalloc(&dindex, (ll)index.size() * sizeof(ll));
    cudaMalloc(&dheadVertex, (ll)headvertex.size() * sizeof(ll));
    cudaMalloc(&dweights, (ll)weights.size() * sizeof(ll));

    cudaMemcpy(dindex, hindex, (ll)index.size() * sizeof(ll), cudaMemcpyHostToDevice);
    cudaMemcpy(dheadVertex, hheadVertex, (ll)headvertex.size() * sizeof(ll), cudaMemcpyHostToDevice);
    cudaMemcpy(dweights, hweights, (ll)weights.size() * sizeof(ll), cudaMemcpyHostToDevice);

    // Creating Edge queue
    Node **nodeQueue;
    cudaMalloc(&nodeQueue, noOfedges * sizeof(Node *));
    for (ll i = 0; i < noOfedges; ++i)
    {
        Node *node;
        cudaMalloc(&node, sizeof(Node));
        allocate<<<1, 1>>>(i, nodeQueue, node);
    }

    Node **dadjList;
    cudaMalloc(&dadjList, index.size() * sizeof(Node *));

    /*// Main 1
    clock_t start, end;
    start = clock();
    generateAdjList<<<1, 1>>>(vertices, dindex, dheadVertex, dweights, nodeQueue, dadjList, noOfedges);
    cudaDeviceSynchronize();
    end = clock();
    double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds
    */

    // Optional Task
    unsigned nblocks = ceil((float) vertices / BLOCKSIZE);
    
    clock_t start, end;
    start = clock();
    generateAdjListParallel<<<nblocks, BLOCKSIZE>>>(vertices, dindex, dheadVertex, dweights, nodeQueue, dadjList);
    cudaDeviceSynchronize();
    end = clock();
    double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds

    printAdjListKernel<<<1,1>>>(vertices, dadjList);
    cudaDeviceSynchronize();

    cout << "Time taken is: " << elapsedTime << " ms" << endl;
    return 0;
}

bool comp_Edges_and_dest(Edge &a, Edge &b)
{
    return a.src == b.src ? a.dest < b.dest : a.src < b.src;
}

bool comp_Edges(Edge &a, Edge &b)
{
    return a.src < b.src;
}

void readMTXFile(string path, vector<Edge> &edgeList, ll &vertices, ll &edges, int &directed, int &weighted)
{
    ifstream file(path);

    if (!file.is_open())
    {
        cout << "Could not open file\n";
        return;
    }

    file >> vertices >> edges >> directed >> weighted;

    if (weighted == 1)
    {
        for (ll i = 0; i < edges; ++i)
        {
            ll u, v, wt;

            file >> u >> v >> wt;

            Edge e;
            e.src = u - 1;
            e.dest = v - 1;
            e.wt = wt;

            edgeList.push_back(e);

            if (!directed)
            {
                e.src = v - 1;
                e.dest = u - 1;
                e.wt = wt;

                edgeList.push_back(e);
            }
        }
    }
    else
    {
        for (ll i = 0; i < edges; ++i)
        {
            ll u, v;

            file >> u >> v;

            Edge e;
            e.src = u - 1;
            e.dest = v - 1;
            e.wt = 1;

            edgeList.push_back(e);

            if (!directed)
            {
                e.src = v - 1;
                e.dest = u - 1;
                e.wt = 1;

                edgeList.push_back(e);
            }
        }
    }

    file.close();
}

void readFile(string path, vector<Edge> &edgeList, ll &vertices, ll &edges, int &directed, int &weighted)
{
    ifstream file(path);

    if (!file.is_open())
    {
        cout << "Could not open file\n";
        return;
    }

    file >> vertices >> edges >> directed >> weighted;

    if (weighted == 1)
    {
        for (ll i = 0; i < edges; ++i)
        {
            ll u, v, wt;

            file >> u >> v >> wt;

            Edge e;
            e.src = u - 1;
            e.dest = v - 1;
            e.wt = wt;

            edgeList.push_back(e);

            if (!directed)
            {
                e.src = v - 1;
                e.dest = u - 1;
                e.wt = wt;

                edgeList.push_back(e);
            }
        }
    }
    else
    {
        for (ll i = 0; i < edges; ++i)
        {
            ll u, v;

            file >> u >> v;

            Edge e;
            e.src = u - 1;
            e.dest = v - 1;
            e.wt = 1;

            edgeList.push_back(e);

            if (!directed)
            {
                e.src = v - 1;
                e.dest = u - 1;
                e.wt = 1;

                edgeList.push_back(e);
            }
        }
    }

    file.close();
}

void constructCSR(ll &vertices, vector<ll> &index, vector<ll> &headvertex, vector<ll> &weights, int directed, int weighted, vector<Edge> &edgeList)
{
    ll edges = edgeList.size();

    // constructing indices for index array
    for (ll i = 0; i < edges; ++i)
    {
        Edge e = edgeList[i];
        ll node = e.src;
        index[node + 1] += 1;
    }

    for (ll i = 1; i < vertices + 1; ++i)
        index[i] += index[i - 1];

    // constructing the headvertex and weights array
    for (ll i = 0; i < edges; ++i)
    {
        Edge e = edgeList[i];
        headvertex.push_back(e.dest);
        weights.push_back(e.wt);
    }
}