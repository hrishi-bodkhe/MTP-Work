#include "triangleCount.h"

ll triangleCount(ll vertices, vector<ll>& index, vector<ll>& headVertex){
    ll tc = 0;

    for(ll p = 0; p < vertices; ++p){
        ll start_p = index[p];
        ll end_p = index[p + 1];

        for(ll i = start_p; i < end_p; ++i){
            ll t = headVertex[i];
            ll start_t = index[t];
            ll end_t = index[t + 1];

            for(ll j = start_p; j < end_p; ++j){
                ll r = headVertex[j];

                if(t == r) continue;

                int is_neighbour = 0;
                
                for(ll k = start_t; k < end_t; ++k){
                    if(headVertex[k] == r){
                        is_neighbour = 1;
                        break;
                    }
                }

                if(is_neighbour) ++tc;
            }
        }
    }

    return tc / 6;
}