#include "kcore.h"

void kCore(ll vertices, vector<ll>& index, vector<ll>& headVertex, ll k){
    vector<ll> degree(vertices, 0);
    ll minDegree = INT_MAX;
    ll startVertex;
    findDegrees(vertices, index, degree, minDegree, startVertex);
	
	//debug
	/*
	for(ll i = 0; i < vertices; ++i) cout << degree[i] << ' ';
	cout << endl;
	cout << "Min Degree: "  << minDegree << endl;
	cout << "StartVertex: " << startVertex << endl;
	*/

	vector<int> vis(vertices, 0);
	dfsForKCore(startVertex, index, headVertex, vis, degree, k);

	//debug
	/*
	for(ll i = 0; i < vertices; ++i) cout << degree[i] << ' ';
	cout << endl;
	*/

	//if graph is disconnected
	for(ll i = 0; i < vertices; ++i){
		if(!vis[i]) dfsForKCore(i, index, headVertex, vis, degree, k);
	}

	//debug
	/*
	for(ll i = 0; i < vertices; ++i) cout << degree[i] << ' ';
	cout << endl;
	*/

	for(ll i = 0; i < vertices; ++i){
		if(degree[i] < k) continue;

		ll count = 0;

		ll startIdx = index[i];
		ll endIdx = index[i + 1];

		for(ll j = startIdx; j < endIdx; ++j){
			ll node = headVertex[j];

			if(degree[node] >= k) ++count;
		}

		if(count < k) degree[i] = count;
	}

	printKCores(vertices, index, headVertex, degree, k);
}

void printKCores(ll vertices, vector<ll>& index, vector<ll>& headVertex, vector<ll>& degree, ll k){
	for(ll i = 0; i < vertices; ++i){
		if(degree[i] < k) continue;

		cout << i << ": ";

		ll startIdx = index[i];
		ll endIdx = index[i + 1];

		for(ll v = startIdx; v < endIdx; ++v){
			ll node = headVertex[v];

			if(degree[node] >= k) cout << node << ' ';
		}
		cout << endl;
	}
}

void dfsForKCore(ll src, vector<ll>& index, vector<ll>& headVertex, vector<int>& vis, vector<ll>& degree, ll k){
	vis[src] = true;

	ll startIdx = index[src];
	ll endIdx = index[src + 1];

	for(ll i = startIdx; i < endIdx; ++i){
		ll node = headVertex[i];

		if(degree[src] < k){
			--degree[node];
		}

		if(!vis[node]) dfsForKCore(node, index, headVertex, vis, degree, k);
	}
}

void findDegrees(ll vertices, vector<ll>& index, vector<ll>& degree, ll& minDegree, ll& startVertex){
    for(ll i = 0; i < vertices; ++i) {
        degree[i] =  index[i + 1] - index[i];
        
        if(minDegree > degree[i]){
            startVertex = i;
            minDegree = degree[i];
        }
    }
}

