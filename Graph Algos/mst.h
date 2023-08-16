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

#endif