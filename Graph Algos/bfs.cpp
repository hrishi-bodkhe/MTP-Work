#include"bfs.h"

void bfsCSRweighted(ll src, ll vertices, vector<ll>& index, vector<ll>& headVertex, vector<ll>& weight, vector<ll>& dist, vector<ll>& parent){
    for(ll i = 0; i < vertices; ++i) {
        dist[i] = INT_MAX;
        parent[i] = -1;
    }
    dist[src] = 0;
    parent[src] = 0;

    int changed;
    while(1){
        changed = 0;
        
        for(ll u = 0; u < vertices; ++u){
            ll start = index[u];
            ll end = index[u + 1];

            ll edges = end - start;
            if(edges == 0) continue;

			for(ll j = start; j < end; ++j){
				ll v = headVertex[j];
                ll wt = weight[j];

				if(dist[u] != INT_MAX && dist[v] > dist[u] + wt){
					dist[v] = dist[u] + wt;
					parent[v] = u;
					changed = 1;
				}
			}
		}

        if(changed == 0) break;
    }

    cout << "\nDistance Array: ";
    for(ll i = 0; i < vertices; ++i) cout << dist[i] << ' ';
    cout << "\nParent Array: ";
    for(ll i = 0; i < vertices; ++i) cout << parent[i] << ' ';
    cout << '\n';
}

void bfsCSR(ll src, ll vertices, vector<ll>& index, vector<ll>& headVertex,  vector<ll>& dist, vector<ll>& parent){
    for(ll i = 0; i < vertices; ++i) {
        dist[i] = INT_MAX;
        parent[i] = -1;
    }
    dist[src] = 0;
    parent[src] = 0;

    int changed;
    while(1){
        changed = 0;
        
        for(ll u = 0; u < vertices; ++u){
            ll start = index[u];
            ll end = index[u + 1];

            ll edges = end - start;
            if(edges == 0) continue;

			for(ll j = start; j < end; ++j){
				ll v = headVertex[j];

				if(dist[u] != INT_MAX && dist[v] > dist[u] + 1){
					dist[v] = dist[u] + 1;
					parent[v] = u;
					changed = 1;
				}
			}
		}

        if(changed == 0) break;
    }

    cout << "\nDistance Array: ";
    for(ll i = 0; i < vertices; ++i) cout << dist[i] << ' ';
    cout << "\nParent Array: ";
    for(ll i = 0; i < vertices; ++i) cout << parent[i] << ' ';
    cout << '\n';
}