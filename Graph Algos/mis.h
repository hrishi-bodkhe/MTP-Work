#ifndef MIS_H
#define MIS_H
#include "preprocessing.h"

void MIS(ll vertices, vector<ll>& index, vector<ll>& headVertex, vector<ll>& maxIndpendentSet);

void printMIS(vector<ll>& maxIndependentSet);

#endif