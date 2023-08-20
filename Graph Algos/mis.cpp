#include "mis.h"

void MIS(ll vertices, vector<ll>& index, vector<ll>& headVertex, vector<ll>& maxIndpendentSet){
    vector<ll> visited(vertices, 0);
    for(ll u = 0; u < vertices; ++u){
        if(!visited[u]){
            visited[u] = 1;
            maxIndpendentSet.emplace_back(u);
        }

        ll startIdx = index[u];
        ll endIdx = index[u + 1];

        for(ll i = startIdx; i < endIdx; ++i){
            ll v = headVertex[i];
            visited[v] = 1;
        }
    }

    printMIS(maxIndpendentSet);
}

void printMIS(vector<ll>& maxIndependentSet){
    cout << "The MIS contains the following vertices: ";
    
    ll n = maxIndependentSet.size();
    for(ll i = 0; i < n; ++i) cout << maxIndependentSet[i] << ' ';
    cout << endl;
}