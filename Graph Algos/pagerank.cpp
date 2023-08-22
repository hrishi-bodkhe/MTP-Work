#include "pagerank.h"

void findPageRank(ll node, vector<ld>& pageRank, ll vertices, vector<ll>& index, vector<ll>& headVertex){
    ld val = 0.0;

    ll startIdx = index[node];
    ll endIdx = index[node + 1];

    for(ll i = startIdx; i < endIdx; ++i){
        ll v = headVertex[i];
        ll outdegree = index[v + 1] - index[v];

        if(outdegree){
            val += pageRank[v] / outdegree;
        }
    }

    pageRank[node] = val * damping_factor + (1 - damping_factor) / vertices;
}

void computePR(ll vertices, vector<ll>& index, vector<ll>& headVertex, vector<ld>& pageRank){
    for(ll node = 0; node < vertices; ++node) pageRank[node] = 1.0 / vertices;

    ll itr = 1;
    while(itr < MAX_ITRS){
        for(ll node = 0; node < vertices; ++node){
            findPageRank(node, pageRank, vertices, index, headVertex);
        }
        ++itr;
    }

    printPR(vertices, pageRank);
}

void printPR(ll vertices, vector<ld>& pageRank){
    for(ll i = 0; i < vertices; ++i) cout << pageRank[i] << ' ';
    cout << endl;
}