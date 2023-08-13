#include "connectedComp.h"

void traverseConnComp(ll u, vector<ll> &index, vector<ll> &headVertex, vector<ll> &color, ll &compNum, vector<ll> &component)
{
	color[u] = 1;
	component[u] = compNum;

	ll startIdx = index[u];
	ll endIdx = index[u + 1];

	for (ll i = startIdx; i < endIdx; ++i)
	{
		ll v = headVertex[i];

		if (color[v] == 0)
			traverseConnComp(v, index, headVertex, color, compNum, component);
	}

	color[u] = 2;
}

void findConnComp(ll src, ll vertices, vector<ll> &index, vector<ll> &headVertex, vector<ll> &color, vector<ll> &component)
{
	// 0 - white
	// 1 - grey
	// 2 - black

	ll count = 0;

	for (ll u = 0; u < vertices; ++u)
	{
		if (color[u] == 0)
		{
			traverseConnComp(u, index, headVertex, color, count, component);
			++count;
		}
	}

	cout << endl;
	for (ll i = 0; i < vertices; ++i)
		cout << i << " is in component: " << component[i] << endl;
}