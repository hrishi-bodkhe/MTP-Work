#include "mst.h"

DS::DisjointSet(ll n)
{
    size.resize(n + 1, 0);
    parent.resize(n + 1, 0);

    for (ll i = 0; i < n + 1; ++i)
        parent[i] = i;
}

ll DS::findParent(ll node)
{
    if (node == parent[node])
        return node;
    return parent[node] = findParent(parent[node]);
}

void DS::unionBySize(ll u, ll v)
{
    ll ulp_u = findParent(u);
    ll ulp_v = findParent(v);

    if (ulp_u == ulp_v)
        return;

    if (size[ulp_u] < size[ulp_v])
    {
        parent[ulp_u] = ulp_v;
        size[ulp_v] += size[ulp_u];
    }
    else
    {
        parent[ulp_v] = ulp_u;
        size[ulp_u] += size[ulp_v];
    }
}

ll DS::trees()
{
    ll n = parent.size();
    ll count = 0;

    for (ll i = 0; i < n; ++i)
    {
        if (parent[i] == i)
            ++count;
    }

    return count;
}

ll BoruvkaMST(ll vertices, vector<ll>& index, vector<ll>& headVertex, vector<ll>& weights, vector<Edge>& mstEdges)
{
    DisjointSet ds(vertices);

    ll mstCost = 0;

    ll numComponents = vertices;

    vector<vector<ll>> cheapestEdge(vertices, vector<ll>(3, -1));

    while(numComponents > 1){

        for(ll u = 0; u < vertices; ++u){
            ll startIdx = index[u];
            ll endIdx = index[u + 1];

            ll set1 = ds.findParent(u);

            for(ll i = startIdx; i < endIdx; ++i){
                ll v = headVertex[i];
                ll wt = weights[i];

                ll set2 = ds.findParent(v);

                if(set1 != set2){
                    if(cheapestEdge[set1][2] == -1 || cheapestEdge[set1][2] > wt){
                        cheapestEdge[set1] = {u, v, wt};
                    }

                    if(cheapestEdge[set2][2] == -1 || cheapestEdge[set2][2] > wt){
                        cheapestEdge[set2] = {u, v, wt};
                    }
                }
            }
        }

        for(ll node = 0; node < vertices; ++node){
            if(cheapestEdge[node][2] != -1){
                ll u = cheapestEdge[node][0];
                ll v = cheapestEdge[node][1];
                ll wt = cheapestEdge[node][2];

                ll set1 = ds.findParent(u);
                ll set2 = ds.findParent(v);

                if(set1 != set2){
                    ds.unionBySize(u, v);

                    mstCost += wt;

                    Edge e;
                    e.src = u;
                    e.dest = v;
                    e.wt = wt;

                    mstEdges.emplace_back(e);
                    --numComponents;
                }
            }
        }

        for(ll node = 0; node < vertices; ++node) cheapestEdge[node][2] = -1;
    }

    printMST(mstEdges, mstCost);

    return mstCost;
}

void printMST(vector<Edge>& mstEdges, ll mstCost){
    cout << "The MST cost is: " << mstCost << endl;
    cout << "The MST edges are: " << endl;
    for(auto& e: mstEdges){
        cout << e.src << ' ' << e.dest << ' ' << e.wt << endl;
    }
}