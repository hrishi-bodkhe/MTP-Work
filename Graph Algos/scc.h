#ifndef SCC_H
#define SCC_H
#include "preprocessing.h"
#include "dfs.h"

void reverseEdges(vector<Edge> &edgeList);

void dfsForScc(ll u, vector<ll> &index, vector<ll> &headVertex, ll color[]);

ll scc(ll vertices, vector<ll> &index, vector<ll> &headVertex, vector<Edge> edgeList, vector<ll> &starttime, vector<ll> &fintime);

#endif