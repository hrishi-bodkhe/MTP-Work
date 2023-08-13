#include "sssp.h"

void sssp(ll vertices, vector<ll> &index, vector<ll> &headVertex, vector<ll> &weight, ll src, vector<ll> &dist, vector<ll> &parent)
{
	dist[src] = 0;

	int changed;

	while (1)
	{
		changed = 0;
		for (ll u = 0; u < vertices; ++u)
		{
			ll startIdx = index[u];
			ll endIdx = index[u + 1];

			ll edges = endIdx - startIdx;
			if (edges == 0)
				continue;

			for (ll i = startIdx; i < endIdx; ++i)
			{
				ll v = headVertex[i];
				ll wt = weight[i];

				if (dist[v] > dist[u] + wt)
				{
					dist[v] = dist[u] + wt;
					parent[v] = u;
					changed = 1;
				}
			}
		}

		if (changed == 0)
			break;
	}

	cout << "Distances: ";
	for (ll i = 0; i < vertices; ++i)
		cout << dist[i] << ' ';
	cout << "\nParents: ";
	for (ll i = 0; i < vertices; ++i)
		cout << parent[i] << ' ';
	cout << '\n';
}