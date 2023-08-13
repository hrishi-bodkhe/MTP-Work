#ifndef APSP_H
#define APSP_H
#include "preprocessing.h"

void printAPSPData(vector<vector<ll>>& dist, vector<vector<ll>>& parent);

void apsp(ll vertices, vector<Edge>& edgeList, vector<vector<ll>>& dist, vector<vector<ll>>& parent, int directed);

#endif