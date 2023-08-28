#include "graphColor.h"

bool isNeighbour(ll u, ll v, vector<ll>& index, vector<ll>& headVertex){
    ll startIdx = index[u];
    ll endIdx = index[u + 1];

    for(ll idx = startIdx; idx < endIdx; ++idx){
        if(headVertex[idx] == v) return true;
    }
    return false;
}

void assignColor(ll k, vector<ll>& index, vector<ll>& headVertex, vector<ll>& colors){
    colors[k] = 1;

    for(ll i = 0; i < k; ++i){
        if(isNeighbour(k, i, index, headVertex) && colors[k] == colors[i])
            colors[k] = colors[k] + 1;
    }
}

void graphColoring(ll vertices, vector<ll>& index, vector<ll>& headVertex, vector<ll>& colors){
    for(ll v = 0; v < vertices; ++v){
        colors[v] = 0; // Initialization of color
        
        for(ll i = 0; i < vertices; ++i){
            assignColor(i, index, headVertex, colors); // Color vertex Order.i
        }
    }

    printColors(vertices, colors);
}

void printColors(ll vertices, vector<ll>& colors){
    for(ll i = 0; i < vertices; ++i){
        cout << i << ": " << colors[i] << endl;
    }
}