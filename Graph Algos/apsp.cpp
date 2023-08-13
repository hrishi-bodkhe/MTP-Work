#include "apsp.h"

void printAPSPData(vector<vector<ll>>& dist, vector<vector<ll>>& parent){
    cout << "Distance Matrix: \n";
    for (auto &vec : dist)
    {
        for (ll i : vec)
            cout << i << ' ';
        cout << endl;
    }

    cout << "Parent Matrix: \n";
    for (auto &vec : parent)
    {
        for (ll i : vec)
            cout << i << ' ';
        cout << endl;
    }
}

void apsp(ll vertices, vector<Edge> &edgeList, vector<vector<ll>> &dist, vector<vector<ll>> &parent, int directed)
{
    for (int i = 0; i < vertices; ++i)
    {
        dist[i][i] = 0;
    }

    for (auto &e : edgeList)
    {
        ll u = e.src;
        ll v = e.dest;
        ll wt = e.wt;

        dist[u][v] = wt;

        if (!directed)
            dist[v][u] = wt;
    }

    for (ll k = 0; k < vertices; ++k)
    {
        for (ll i = 0; i < vertices; ++i)
        {
            for (ll j = 0; j < vertices; ++j)
            {
                if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX && ((dist[i][k] + dist[k][j]) < dist[i][j]))
                {
                    dist[i][j] = dist[i][k] + dist[k][j];
                    parent[i][j] = k;
                }
            }
        }
    }
}