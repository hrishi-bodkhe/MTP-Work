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

void BoruvkaMST(ll vertices, vector<ll>& index, vector<ll>& headVertex, vector<ll>& weights)
{
    DisjointSet ds(vertices);

    vector<vector<ll>> minEdge(vertices, vector<ll>(3, -1));

    

}