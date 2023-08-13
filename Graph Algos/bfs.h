#ifndef BFS_H
#define BFS_H
#include "preprocessing.h"

void bfsCSRweighted(ll src, ll vertices, vector<ll> &index, vector<ll> &headVertex, vector<ll> &weight, vector<ll> &dist, vector<ll> &parent);

void bfsCSR(ll src, ll vertices, vector<ll> &index, vector<ll> &headVertex, vector<ll> &dist, vector<ll> &parent);

#endif