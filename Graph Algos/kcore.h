#ifndef KCORE_H
#define KCORE_H
#include "preprocessing.h"

void kCore(ll vertices, vector<ll>& index, vector<ll>& headVertex, ll k);

void printKCores(ll vertices, vector<ll>& index, vector<ll>& headVertex, vector<ll>& degree, ll k);

void dfsForKCore(ll src, vector<ll>& index, vector<ll>& headVertex, vector<int>& vis, vector<ll>& degree, ll k);

void findDegrees(ll vertices, vector<ll>& index, vector<ll>& degree, ll& minDegree, ll& startVertex);

#endif