#include<stdio.h>
#include<limits.h>
#include<stdlib.h>

void apsp(int n){
	int dist[n][n];
	int parent[n][n];

	for(int i = 0; i < n; ++i){
		for(int j = 0; j < n; ++j){
			if(i == j){ 
				dist[i][j] = 0;
				parent[i][j] = i;
			}
			else {
				dist[i][j] = INT_MAX;
				parent[i][j] = -1;
			}
		}
	}

	dist[0][1] = 2;
	dist[1][2] = 3;
	dist[2][0] = 5;
	dist[2][1] = 8;

	for(int k = 0; k < n; ++k){
		for(int i = 0; i < n; ++i){
			for(int j = 0; j < n; ++j){
				if(dist[i][k] != INT_MAX && dist[k][j] != INT_MAX && ((dist[i][k] + dist[k][j]) < dist[i][j])){
					dist[i][j] = dist[i][k] + dist[k][j];
					parent[i][j] = k;
				}
			}
		}

	}

	printf("Distance Matrix:\n");
	for(int i = 0; i < n; ++i){
		for(int j = 0; j < n; ++j)
			printf("%d ", dist[i][j]);
		printf("\n");
	}

	printf("\nParent Matrix:\n");

	for(int i = 0; i < n; ++i){
		for(int j = 0; j < n; ++j)
			printf("%d ", parent[i][j]);
		printf("\n");
	}

	printf("\n");
}

int main(){
	int n = 3;
	
	apsp(n);

	// printGraph(graph);
	

	return 0;
}