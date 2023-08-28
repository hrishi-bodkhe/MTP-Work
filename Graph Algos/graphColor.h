#ifndef GRAPH_COLOR
#define GRAPH_COLOR
#include "preprocessing.h"

bool isNeighbour(ll u, ll v, vector<ll>& index, vector<ll>& headVertex);

void assignColor(ll k, vector<ll>& index, vector<ll>& headVertex, vector<ll>& colors);

void graphColoring(ll vertices, vector<ll>& index, vector<ll>& headVertex, vector<ll>& colors);

void printColors(ll vertices, vector<ll>& colors);

#endif