#ifndef CONNCOMP_H
#define CONNCOMP_H
#include"preprocessing.h"

void traverseConnComp(ll u, vector<ll>& index, vector<ll>& headVertex, vector<ll>& color, ll& compNum, vector<ll>& component);

void findConnComp(ll src, ll vertices, vector<ll>& index, vector<ll>& headVertex, vector<ll>& color, vector<ll>& component);

#endif