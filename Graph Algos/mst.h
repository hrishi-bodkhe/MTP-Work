#ifndef MST_H
#define MST_H
#include "preprocessing.h"
#define DisjointSet DS

struct DisjointSet
{
    vector<int> parent, size;

    DisjointSet(ll n);

    ll findParent(ll node);

    void unionBySize(ll u, ll v);

    ll trees();
};

ll BoruvkaMST(ll vertices, vector<ll>& index, vector<ll>& headVertex, vector<ll>& weights, vector<Edge>& mstEdges);

void printMST(vector<Edge>& mstEdges, ll mstCost);

#endif