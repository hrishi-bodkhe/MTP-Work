#ifndef PAGERANK_H
#define PAGERANK_H
#include "preprocessing.h"
#define MAX_ITRS 10
#define damping_factor 0.85

void findPageRank(ll node, vector<ld>& pageRank, ll vertices, vector<ll>& index, vector<ll>& headVertex);

void computePR(ll vertices, vector<ll>& index, vector<ll>& headVertex, vector<ld>& pageRank);

void printPR(ll vertices, vector<ld>& pageRank);

#endif