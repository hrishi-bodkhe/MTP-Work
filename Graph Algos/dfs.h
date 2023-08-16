#ifndef DFS_H
#define DFS_H
#include "preprocessing.h"

// void traverse(ll u, vector<ll> &index, vector<ll> &headVertex, ll color[], ll *clock, vector<ll> &starttime, vector<ll> &fintime);
void traverse(ll u, vector<ll> &index, vector<ll> &headVertex, vector<Node>& property, ll &clock);

// void dfsCSR(ll src, ll vertices, vector<ll> &index, vector<ll> &headVertex, vector<ll> &starttime, vector<ll> &fintime);
void dfsCSR(ll src, ll vertices, vector<ll>& index, vector<ll>& headVertex, vector<Node>& property);

#endif