#include "weaklycc.h"

ll weaklycc(ll vertices, vector<Edge>& edgeList, vector<ll>& color, vector<ll>& component){
    ll edges = edgeList.size();
    for(int i = 0; i < edges; ++i){
        auto& e = edgeList[i];
        Edge _e;
        _e.src = e.dest;
        _e.dest = e.src;
        _e.wt = 1;
        edgeList.push_back(_e);
    }

    sort(edgeList.begin(), edgeList.end(), comp_Edges_and_dest);

    vector<ll> index(vertices + 1, 0);
    vector<ll> headVertex;
    vector<ll> dummy;

    constructCSR(vertices, index, headVertex, dummy, 0, 0, edgeList);

    return findConnComp(0, vertices, index, headVertex, color, component);
}