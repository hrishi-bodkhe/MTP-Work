#ifndef BETWEENNESSCENTRALITY_H
#define BETWEENNESSCENTRALITY_H
#include "preprocessing.h"

const int INF = numeric_limits<int>::max();

void vertexBetweennessCentrality(ll vertices, vector<ll> &index, vector<ll> &headVertex);

void printVBC(ll vertices, vector<ld>& BC);

#endif