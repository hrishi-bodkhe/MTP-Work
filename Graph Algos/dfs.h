#ifndef DFS_H
#define DFS_H
#include"preprocessing.h"

void traverse(ll u, vector<ll>& index, vector<ll>& headVertex, ll color[], ll *clock, ll starttime[], ll fintime[]);
void dfsCSR(ll src, ll vertices, vector<ll>& index, vector<ll>& headVertex);

#endif