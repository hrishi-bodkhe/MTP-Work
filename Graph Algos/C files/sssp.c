#include <stdio.h>
#include <limits.h>
#include <stdlib.h>

// Node structure for adjacency list
typedef struct Node
{
	int vertex;
	int weight;
	struct Node *next;
} Node;

// Graph structure with an array of adjacency lists
typedef struct Graph
{
	int numVertices;
	Node **adjList;
} Graph;

// Function to create a new node
Node *createNode(int v, int weight)
{
	Node *node = (Node *)malloc(sizeof(Node));
	node->vertex = v;
	node->weight = weight;
	node->next = NULL;

	return node;
}

// Function to create a graph with a given number of vertices
Graph *createGraph(int vertices)
{
	Graph *graph = (Graph *)malloc(sizeof(Graph));
	graph->numVertices = vertices;

	graph->adjList = (Node **)malloc(vertices * sizeof(Node *));

	for (int i = 0; i < vertices; ++i)
		graph->adjList[i] = NULL;

	return graph;
}

// Function to add an edge to an undirected graph
void addEdge(Graph *graph, int u, int v, int wt)
{
	Node *node = createNode(v, wt);
	node->next = graph->adjList[u];
	graph->adjList[u] = node;

	node = createNode(u, wt);
	node->next = graph->adjList[v];
	graph->adjList[v] = node;
}

// Function to print the adjacency list representation of a graph
void printGraph(Graph *graph)
{
	if (!graph || !graph->adjList)
		return;

	for (int u = 0; u < graph->numVertices; ++u)
	{
		Node *node = graph->adjList[u];

		printf("For %d: ", u);
		while (node)
		{
			printf("(%d, %d)->", node->vertex, node->weight);
			node = node->next;
		}
		printf("\n");
	}
}

void freeGraph(Graph *graph)
{
	if (!graph || !graph->adjList)
		return;

	for (int i = 0; i < graph->numVertices; ++i)
	{
		Node *node = graph->adjList[i];

		while (node)
		{
			Node *prev = node;
			node = node->next;
			prev->next = NULL;
			free(prev);
		}
	}

	free(graph->adjList);
	free(graph);
}

void sssp(Graph *graph, int src)
{
	int n = graph->numVertices;
	int dist[n], parent[n];

	for (int i = 0; i < n; ++i)
	{
		dist[i] = INT_MAX;
		parent[i] = -1;
	}

	dist[src] = 0;

	int changed;

	while (1)
	{
		changed = 0;
		for (int u = 0; u < n; ++u)
		{
			Node *node = graph->adjList[u];
			while (node)
			{
				int v = node->vertex;
				int wt = node->weight;

				if (dist[v] > dist[u] + wt)
				{
					dist[v] = dist[u] + wt;
					parent[v] = u;
					changed = 1;
				}

				node = node->next;
			}
		}

		if (changed == 0)
			break;
	}

	for (int i = 0; i < n; ++i)
		printf("%d ", dist[i]);
	printf("\n");
	for (int i = 0; i < n; ++i)
		printf("%d ", parent[i]);
}
int main()
{
	int n = 5;
	Graph *graph = createGraph(n);

	addEdge(graph, 0, 1, 5);
	addEdge(graph, 0, 4, 2);
	addEdge(graph, 1, 2, 3);
	addEdge(graph, 1, 3, 2);
	addEdge(graph, 1, 4, 1);
	addEdge(graph, 2, 3, 1);
	addEdge(graph, 3, 4, 11);

	sssp(graph, 0);

	// printGraph(graph);

	return 0;
}