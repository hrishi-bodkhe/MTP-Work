#include "betweennessCentrality.h"

void vertexBetweennessCentrality(ll vertices, vector<ll> &index, vector<ll> &headVertex)
{
    // dist[v][p] stores shortest distance to vertex p from v
    // spnum[v][p] stores the number of shortest paths between vertices v & p
    // BC[v] stores betweenness centrality of vertex v

    vector<vector<ll>> dist(vertices, vector<ll>(vertices, INF));
    vector<vector<ll>> spnum(vertices, vector<ll>(vertices, 0));

    for (ll i = 0; i < vertices; ++i)
        dist[i][i] = 0;

    for (ll v = 0; v < vertices; ++v)
    {
        ll startIdx = index[v];
        ll endIdx = index[v + 1];

        for (ll idx = startIdx; idx < endIdx; ++idx)
        {
            ll p = headVertex[idx];

            dist[v][p] = 1;
            spnum[v][p] = 1;
        }
    }

    for (ll k = 0; k < vertices; ++k)
    {

        for (ll i = 0; i < vertices; ++i)
        {

            for (ll j = 0; j < vertices; ++j)
            {

                if (dist[i][k] + dist[k][j] < dist[i][j])
                {
                    dist[i][j] = dist[i][k] + dist[k][j];
                    spnum[i][j] = spnum[i][k] * spnum[k][j];
                }
                else if (dist[i][k] + dist[k][j] == dist[i][j])
                {
                    spnum[i][j] += spnum[i][k] * spnum[k][j];
                }
            }
        }
    }

    // debug
    /*
    for(ll i = 0; i < vertices; ++i){
        for(ll j = 0; j < vertices; ++j) cout << dist[i][j] << ' ';
        cout << endl;
    }

    for(ll i = 0; i < vertices; ++i){
        for(ll j = 0; j < vertices; ++j) cout << spnum[i][j] << ' ';
        cout << endl;
    }
    */

    vector<ld> BC(vertices, 0.0);
    for (ll v = 0; v < vertices; ++v)
    {
        for (ll s = 0; s < vertices; ++s)
        {
            for (ll t = 0; t < vertices; ++t)
            {
                if (s != t && dist[s][t] == dist[s][v] + dist[v][t])
                {
                    BC[v] += ((ld)(spnum[s][v] * spnum[v][t]) / spnum[s][t]);
                }
            }
        }
    }

    printVBC(vertices, BC);
}

void printVBC(ll vertices, vector<ld> &BC)
{
    for (ll i = 0; i < vertices; ++i)
        cout << BC[i] << ' ';
    cout << endl;
}