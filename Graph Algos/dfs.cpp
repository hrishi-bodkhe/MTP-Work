#include "dfs.h"

void traverse(ll u, vector<ll> &index, vector<ll> &headVertex, ll color[], ll *clock, vector<ll> &starttime, vector<ll> &fintime)
{
	*clock = *clock + 1;
	starttime[u] = *clock;
	color[u] = 1;

	ll startIdx = index[u];
	ll endIdx = index[u + 1];

	for (ll i = startIdx; i < endIdx; ++i)
	{
		ll v = headVertex[i];

		if (color[v] == 0)
			traverse(v, index, headVertex, color, clock, starttime, fintime);
	}

	color[u] = 2;
	*clock = *clock + 1;
	fintime[u] = *clock;
}

void dfsCSR(ll src, ll vertices, vector<ll> &index, vector<ll> &headVertex, vector<ll> &starttime, vector<ll> &fintime)
{
	// 0 - white
	// 1 - grey
	// 2 - black

	ll clock = 0;
	ll color[vertices];

	for (ll u = 0; u < vertices; ++u)
	{
		color[u] = 0;
		starttime[u] = -1;
		fintime[u] = -1;
	}

	traverse(src, index, headVertex, color, &clock, starttime, fintime);

	for (ll u = 0; u < vertices; ++u)
	{
		if (color[u] == 0)
			traverse(u, index, headVertex, color, &clock, starttime, fintime);
	}

	// cout << "\n";
	// cout << "Start Times: ";
	// for(ll i = 0; i < vertices; ++i) cout << starttime[i] << ' ';
	// cout << "\nFinish Times: ";
	// for(ll i = 0; i < vertices; ++i) cout << fintime[i] << ' ';
	// cout << "\n" ;
}