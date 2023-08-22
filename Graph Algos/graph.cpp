#include "config.h"
#include "bfs.h"
#include "dfs.h"
#include "sssp.h"
#include "connectedComp.h"
#include "scc.h"
#include "apsp.h"
#include "weaklycc.h"
#include "mst.h"
#include "mis.h"
#include "pagerank.h"

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

    // printEdgeList(edgeList);

    // construct CSR

    vector<ll> index(vertices + 1, 0);
    vector<ll> headvertex;
    vector<ll> weights;

    constructCSR(vertices, index, headvertex, weights, directed, weighted, edgeList);

    // //print CSR
    // printCSR(index, headvertex, weights);

    // vector<ll> dist(vertices, INT_MAX);
    // vector<ll> parent(vertices, -1);
    // vector<Node> property(vertices);

    // if(weighted) bfsCSRweighted(0, vertices, index, headvertex, weights, dist, parent);
    // else bfsCSR(0, vertices, index, headvertex, dist, parent);

    // dfsCSR(0, vertices, index, headvertex);
    // sssp(vertices, index, headvertex, weights, 0, dist, parent);

    // vector<ll> color(vertices, 0);
    // vector<ll> component(vertices, -1);
    // findConnComp(0, vertices, index, headvertex, color, component);

    // ll ans = scc(vertices, index, headvertex, edgeList, property);
    // cout << ans << endl;

    // vector<vector<ll>> distAPSP(vertices, vector<ll>(vertices, INT_MAX));
    // vector<vector<ll>> parentAPSP(vertices, vector<ll>(vertices, -1));
    // apsp(vertices, edgeList, distAPSP, parentAPSP, directed);
    // printAPSPData(distAPSP, parentAPSP);

    // vector<ll> color(vertices, 0);
    // vector<ll> component(vertices, -1);

    // ll weaklyConnComp = weaklycc(vertices, edgeList, color, component);
    // cout << weaklyConnComp << endl;

    // vector<Edge> mstEdges;
    // ll mstCost = BoruvkaMST(vertices, index, headvertex, weights, mstEdges);

    // vector<ll> maxIndependentSet;
    // MIS(vertices, index, headvertex, maxIndependentSet);

    vector<ld> pageRank(vertices);
    computePR(vertices, index, headvertex, pageRank);
    
    
    return 0;
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

void printCSR(vector<ll> &index, vector<ll> &headvertex, vector<ll> &weights)
{
    cout << "Index: ";
    for (auto &v : index)
        cout << v << ' ';
    cout << endl;

    cout << "Head Vertex: ";
    for (auto &v : headvertex)
        cout << v << ' ';
    cout << endl;

    cout << "Weights: ";
    for (auto &v : weights)
        cout << v << ' ';
    cout << endl;
}

bool comp_Edges_and_dest(Edge &a, Edge &b)
{
    return a.src == b.src ? a.dest < b.dest : a.src < b.src;
}

bool comp_Edges(Edge &a, Edge &b)
{
    return a.src < b.src;
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
            e.src = u;
            e.dest = v;
            e.wt = wt;

            edgeList.push_back(e);

            if (!directed)
            {
                e.src = v;
                e.dest = u;
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
            e.src = u;
            e.dest = v;
            e.wt = 1;

            edgeList.push_back(e);

            if (!directed)
            {
                e.src = v;
                e.dest = u;
                e.wt = 1;

                edgeList.push_back(e);
            }
        }
    }

    file.close();
}

void printEdgeList(vector<Edge> &edgeList)
{
    for (Edge e : edgeList)
        cout << e.src << ' ' << e.dest << ' ' << e.wt << endl;
}