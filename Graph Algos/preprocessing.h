#ifndef PREPROCESSING_H
#define PREPROCESSING_H
#include <bits/stdc++.h>
#include <iostream>
#define ll long long
using namespace std;

struct Edge
{
    ll src;  // source
    ll dest; // destination
    ll wt;   // weight
};

void readFile(string path, vector<Edge> &edgeList, ll &vertices, ll &edges, int &directed, int &weighted);

void printEdgeList(vector<Edge> &edgeList);

bool comp_Edges_and_dest(Edge &a, Edge &b);

bool comp_Edges(Edge &a, Edge &b);

void constructCSR(ll &vertices, vector<ll> &index, vector<ll> &headvertex, vector<ll> &weights, int directed, int weighted, vector<Edge> &edgeList);

void printCSR(vector<ll> &index, vector<ll> &headvertex, vector<ll> &weights);

#endif