#include "scc.h"
#include "config.h"

// Using Kosaraju's Algorithm

void reverseEdges(vector<Edge> &edgeList)
{
    for (Edge &e : edgeList)
    {
        int u = e.src;
        e.src = e.dest;
        e.dest = u;
    }
}

/*
void dfsForScc(ll u, vector<ll> &index, vector<ll> &headVertex, vector<ll> &color)
{
    color[u] = 1;

    ll startIdx = index[u];
    ll endIdx = index[u + 1];

    for (ll i = startIdx; i < endIdx; ++i)
    {
        ll v = headVertex[i];
        // cout << u << ' ' << v << endl;

        if (color[v] == 0)
            dfsForScc(v, index, headVertex, color);
    }

    color[u] = 2;
}
*/

void dfsForScc(ll u, vector<ll> &index, vector<ll> &headVertex, vector<Node>& property){
    property[u].color = 1;

    ll startIdx = index[u];
    ll endIdx = index[u + 1];

    for (ll i = startIdx; i < endIdx; ++i)
    {
        ll v = headVertex[i];
        // cout << u << ' ' << v << endl;

        if (property[v].color == 0)
            dfsForScc(v, index, headVertex, property);
    }

    property[u].color = 2;
}

ll scc(ll vertices, vector<ll> &index, vector<ll> &headVertex, vector<Edge> edgeList, vector<Node>& property)
{

    // Call DFS(G) and compute v.ftime for each vertex v ∈ V .
    dfsCSR(4, vertices, index, headVertex, property);

    // Compute G^T
    ll edges = edgeList.size();

    reverseEdges(edgeList);
    sort(edgeList.begin(), edgeList.end(), comp_Edges_and_dest);
    // for(auto& e: edgeList) cout << e.src << ' ' << e.dest << endl;

    vector<ll> indexRev(vertices + 1, 0);
    vector<ll> headVertexRev;
    vector<ll> dummy;

    constructCSR(vertices, indexRev, headVertexRev, dummy, 1, 0, edgeList);
    
    #ifdef DEBUG_SCC
    printCSR(indexRev, headVertexRev, dummy);
    #endif

    // Sort the vertices in the decreasing order of the value of ftime
    vector<Node> propertyCopy;
    propertyCopy = property;

    sort(propertyCopy.begin(), propertyCopy.end(), [](Node& a, Node& b){
        return a.finTime > b.finTime;
    });

    
    /*
    priority_queue<pair<int, int>> maxHeap;
    for (int i = 0; i < vertices; ++i)
        maxHeap.push({fintime[i], i});
    */

    // Call DFS(G^T) using sorted order of vertices
    ll strongComponents = 0;
    for(int i = 0; i < vertices; ++i) propertyCopy[i].color = 0;

    for(ll i = 0; i < vertices; ++i){
        Node& node = propertyCopy[i];
        ll u = node.vertex;

        if(node.color == 0){
            ++strongComponents;
            dfsForScc(u, indexRev, headVertexRev, propertyCopy);
        }
    }

    /*
    ll strongComponents = 0;
    vector<ll> color(vertices, 0);

    while (!maxHeap.empty())
    {
        auto &p = maxHeap.top();
        int u = p.second;
        maxHeap.pop();

        if (color[u] == 0)
        {
            // cout << u << endl;
            ++strongComponents;
            dfsForScc(u, indexRev, headVertexRev, color);
        }
    }
    */

    // Each spanning tree resulting from the depth-first search in Step 4 corresponds to an SCC
    return strongComponents;
}