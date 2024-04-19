#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>

#include <unistd.h>
#include <thread>

#include <cuda_profiler_api.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define THREADS_PER_BLOCK 1024
#define VERTEX_BLOCK_SIZE 51000000
#define EDGE_BLOCK_SIZE 15
#define VERTEX_PREALLOCATE_LIST_SIZE 2000
#define EDGE_PREALLOCATE_LIST_SIZE 160000000
#define BATCH_SIZE 21
#define VERTEX_BATCH_SIZE 100


// d_source_vector_1 should be managed before running SSSP
// below flag should 0 for undirected graphs, 1 for directed graphs
#define DIRECTED 0

// #define BATCH_SIZE 340
// #define BATCH_SIZE 30491458
// #define BATCH_SIZE 32073440
// #define BATCH_SIZE 47997626
// #define BATCH_SIZE 85362744
// #define BATCH_SIZE 100663202
// #define BATCH_SIZE 108109320
// #define BATCH_SIZE 182084020
// #define BATCH_SIZE 265114400
// #define BATCH_SIZE 802465952
// #define BATCH_SIZE 936364282
#define BATCH_SIZE_DELETE 6
#define INCREMENTAL 1

#define SSSP_LOAD_FACTOR 8

#define PREALLOCATE_EDGE_BLOCKS 100
#define SEARCH_BLOCKS_COUNT 100
#define INFTY 1000000000
#define BIT_STRING_LOOKUP_SIZE 4096

#define CUDA_CHECK_ERROR() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
            exit(err); \
        } \
    } while(0)


// Issues faced
// Too big of an edge_preallocate_list giving errors

// Inserts
// -> Vertices => 1 thread per vertex block

struct graph_properties {

    unsigned long xDim;
    unsigned long yDim;
    unsigned long total_edges;

};

struct vertex_preallocated_queue {

    long front;
    long rear;
    unsigned long count;
    struct vertex_block *vertex_block_address[VERTEX_PREALLOCATE_LIST_SIZE];

};

struct edge_preallocated_queue {

    unsigned int front;
    unsigned int rear;
    unsigned long count;
    struct edge_block *edge_block_address[EDGE_PREALLOCATE_LIST_SIZE];

};

// below are structures for the data structure

struct edge {

    unsigned long destination_vertex;
    // unsigned long weight;
    // unsigned long timestamp;

};

struct edge_block {

    // Array of Structures (AoS) for each edge block
    struct edge edge_block_entry[EDGE_BLOCK_SIZE];
    unsigned short active_edge_count;
    struct edge_block *lptr;
    struct edge_block *rptr;
    struct edge_block *level_order_predecessor;

};

struct adjacency_sentinel {

    unsigned long edge_block_count;
    unsigned long active_edge_count;
    unsigned long last_insert_edge_offset;

    struct edge_block *last_insert_edge_block;
    struct edge_block *next;

};

struct adjacency_sentinel_new {

    unsigned long edge_block_count;
    unsigned long active_edge_count;
    unsigned long last_insert_edge_offset;

    struct edge_block *last_insert_edge_block;
    // struct edge_block *edge_block_address[100];
    struct edge_block *edge_block_address;

};

struct vertex_block {

    // Structure of Array (SoA) for each vertex block
    unsigned long vertex_id[VERTEX_BLOCK_SIZE];
    struct adjacency_sentinel *vertex_adjacency[VERTEX_BLOCK_SIZE];
    unsigned long active_vertex_count;
    struct vertex_block *next;

    // adjacency sentinel
    unsigned long edge_block_count[VERTEX_BLOCK_SIZE];
    unsigned long last_insert_edge_offset[VERTEX_BLOCK_SIZE];

};

struct vertex_dictionary_sentinel {

    unsigned long vertex_block_count;
    unsigned long vertex_count;
    unsigned long last_insert_vertex_offset;

    struct vertex_block *last_insert_vertex_block;
    struct vertex_block *next;

};

// include valid/invalid bit here
struct vertex_dictionary_structure {

    // // Structure of Array (SoA) for vertex dictionary
    // unsigned long vertex_id[VERTEX_BLOCK_SIZE];
    // struct adjacency_sentinel_new *vertex_adjacency[VERTEX_BLOCK_SIZE];
    // unsigned long edge_block_count[VERTEX_BLOCK_SIZE];
    // unsigned long active_vertex_count;

    // below is directly from edge sentinel node
    // unsigned long edge_block_count[VERTEX_BLOCK_SIZE];
    // unsigned int active_edge_count[VERTEX_BLOCK_SIZE];
    // unsigned long last_insert_edge_offset[VERTEX_BLOCK_SIZE];

    // struct edge_block *last_insert_edge_block[VERTEX_BLOCK_SIZE];
    // // struct edge_block *edge_block_address[100];
    // struct edge_block *edge_block_address[VERTEX_BLOCK_SIZE];

    // New Structure of Array (SoA) for vertex dictionary
    unsigned long *vertex_id;
    struct adjacency_sentinel_new *vertex_adjacency[VERTEX_BLOCK_SIZE];
    unsigned long *edge_block_count;
    unsigned long active_vertex_count;

    // // below is directly from edge sentinel node
    // // unsigned long edge_block_count[VERTEX_BLOCK_SIZE];
    unsigned int *active_edge_count;
    unsigned long *last_insert_edge_offset;

    struct edge_block **last_insert_edge_block;
    // struct edge_block *edge_block_address[100];
    struct edge_block **edge_block_address;

};

struct vertex_dictionary_structure_new {

    // Structure of Array (SoA) for vertex dictionary
    unsigned long vertex_id[VERTEX_BLOCK_SIZE];
    struct adjacency_sentinel_new *vertex_adjacency[VERTEX_BLOCK_SIZE];
    unsigned long edge_block_count[VERTEX_BLOCK_SIZE];
    unsigned long active_vertex_count;

};

// thrust::copy(h_source_degrees_new.begin(), h_source_degrees_new.end(), d_source_degrees_new.begin());
// thrust::copy(h_csr_offset_new.begin(), h_csr_offset_new.end(), d_csr_offset_new.begin());
// thrust::copy(h_csr_edges_new.begin(), h_csr_edges_new.end(), d_csr_edges_new.begin());
// thrust::copy(h_edge_blocks_count.begin(), h_edge_blocks_count.end(), d_edge_blocks_count.begin());
// thrust::copy(h_prefix_sum_edge_blocks_new.begin(), h_prefix_sum_edge_blocks_new.end(), d_prefix_sum_edge_blocks_new.begin());

struct batch_update_data {

    unsigned long *csr_offset;
    unsigned long *csr_edges;
    unsigned long *prefix_sum_edge_blocks;

};

// global variables
__device__ struct vertex_dictionary_sentinel d_v_d_sentinel;
__device__ struct vertex_preallocated_queue d_v_queue;

__device__ struct adjacency_sentinel d_a_sentinel;
__device__ struct edge_preallocated_queue d_e_queue;

__device__ unsigned long bit_string_lookup[BIT_STRING_LOOKUP_SIZE];

// below function generates the bit string for the parameter val
__device__ unsigned long traversal_string(unsigned long val, unsigned long *length) {

    unsigned long temp = val;
    unsigned long bit_string = 0;
    *length = 0;

    while(temp > 1) {

        // bit_string = ((temp % 2) * pow(10, iteration++)) + bit_string;
        if(temp % 2)
            bit_string = (bit_string * 10) + 2;
        else
            bit_string = (bit_string * 10) + 1;

        // bit_string = (bit_string * 10) + (temp % 2);
        temp = temp / 2;
        // (*length)++;
        // *length = *length + 1;


    }

    // printf("%lu\n", iteration);

    return bit_string;
}

// below function generates the lookup table for bit strings, which is used during traversals
// launched with number of threads equal to size of the lookup table
__global__ void build_bit_string_lookup() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < BIT_STRING_LOOKUP_SIZE) {

        unsigned long length = 0;
        // value passed to traversal string is (id + 1) since we count edge blocks from 1
        bit_string_lookup[id] = traversal_string(id + 1, &length);

    }

}

__global__ void print_bit_string() {

    for(unsigned long i = 0 ; i < 100 ; i++)
        printf("\nBit String for %lu is %lu\n", i + 1, bit_string_lookup[i]);

}

// __device__ void push_to_vertex_preallocate_queue(struct vertex_block *device_vertex_block) {

//     if( (d_v_queue.rear + 1) % VERTEX_PREALLOCATE_LIST_SIZE == d_v_queue.front ) {
//         printf("Vertex queue Full, front = %ld, rear = %ld\n", d_v_queue.front, d_v_queue.rear);

//         return;
//     }
//     else if (d_v_queue.front == -1)
//         d_v_queue.front = 0;

//     d_v_queue.rear = (d_v_queue.rear + 1) % VERTEX_PREALLOCATE_LIST_SIZE;
//     d_v_queue.vertex_block_address[d_v_queue.rear] = device_vertex_block;
//     d_v_queue.count++;

//     printf("Inserted %p to the vertex queue, front = %ld, rear = %ld\n", d_v_queue.vertex_block_address[d_v_queue.rear], d_v_queue.front, d_v_queue.rear);

// }

// __device__ struct vertex_block* pop_from_vertex_preallocate_queue(unsigned long pop_count) {

//     if(d_v_queue.front == -1) {
//         printf("Vertex queue empty, front = %ld, rear = %ld\n", d_v_queue.front, d_v_queue.rear);
//         return NULL;
//     }
//     else {

//         struct vertex_block *device_vertex_block = d_v_queue.vertex_block_address[d_v_queue.front];
//         d_v_queue.vertex_block_address[d_v_queue.front] = NULL;
//         d_v_queue.count -= pop_count;
//         printf("Popped %p from the vertex queue, front = %ld, rear = %ld\n", device_vertex_block, d_v_queue.front, d_v_queue.rear);

//         if(d_v_queue.front == d_v_queue.rear) {

//             d_v_queue.front = -1;
//             d_v_queue.rear = -1;

//         }
//         else
//             d_v_queue.front = (d_v_queue.front + 1) % VERTEX_PREALLOCATE_LIST_SIZE;

//         return device_vertex_block;
//     }

// }

__device__ struct vertex_block* parallel_pop_from_vertex_preallocate_queue(unsigned long pop_count, unsigned long id) {

    struct vertex_block *device_vertex_block;



    if((d_v_queue.count < pop_count) || (d_v_queue.front == -1)) {
        // printf("Vertex queue empty, front = %ld, rear = %ld\n", d_v_queue.front, d_v_queue.rear);
        return NULL;
    }

    else {

        device_vertex_block = d_v_queue.vertex_block_address[d_v_queue.front + id];
        d_v_queue.vertex_block_address[d_v_queue.front + id] = NULL;
        // printf("Popped %p from the vertex queue, placeholders front = %ld, rear = %ld\n", device_vertex_block, d_v_queue.front, d_v_queue.rear);

    }

    __syncthreads();

    if(id == 0) {

        d_v_queue.count -= pop_count;

        // printf("Vertex Queue before, front = %ld, rear = %ld\n", d_v_queue.front, d_v_queue.rear);

        if((d_v_queue.front + pop_count - 1) % VERTEX_PREALLOCATE_LIST_SIZE == d_v_queue.rear) {

            d_v_queue.front = -1;
            d_v_queue.rear = -1;

        }
        else
            d_v_queue.front = (d_v_queue.front + pop_count) % VERTEX_PREALLOCATE_LIST_SIZE;

        // printf("Vertex Queue before, front = %ld, rear = %ld\n", d_v_queue.front, d_v_queue.rear);

    }

    return device_vertex_block;
}

// __device__ void push_to_edge_preallocate_queue(struct edge_block *device_edge_block) {

//     if( (d_e_queue.rear + 1) % EDGE_PREALLOCATE_LIST_SIZE == d_e_queue.front ) {
//         printf("Edge queue Full, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);

//         return;
//     }
//     else if (d_e_queue.front == -1)
//         d_e_queue.front = 0;

//     d_e_queue.rear = (d_e_queue.rear + 1) % EDGE_PREALLOCATE_LIST_SIZE;
//     d_e_queue.edge_block_address[d_e_queue.rear] = device_edge_block;
//     d_e_queue.count++;

//     printf("Inserted %p to the edge queue, front = %ld, rear = %ld\n", d_e_queue.edge_block_address[d_e_queue.rear], d_e_queue.front, d_e_queue.rear);

// }

// __device__ struct edge_block* pop_from_edge_preallocate_queue(unsigned long pop_count) {

//     if(d_e_queue.front == -1) {
//         printf("Edge queue empty, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);
//         return NULL;
//     }
//     else {

//         struct edge_block *device_edge_block = d_e_queue.edge_block_address[d_e_queue.front];
//         d_e_queue.edge_block_address[d_e_queue.front] = NULL;
//         d_e_queue.count -= pop_count;
//         printf("Popped %p from the edge queue, front = %ld, rear = %ld\n", device_edge_block, d_e_queue.front, d_e_queue.rear);

//         if(d_e_queue.front == d_e_queue.rear) {

//             d_e_queue.front = -1;
//             d_e_queue.rear = -1;

//         }
//         else
//             d_e_queue.front = (d_e_queue.front + 1) % EDGE_PREALLOCATE_LIST_SIZE;

//         return device_edge_block;
//     }

// }

__device__ unsigned k1counter = 0;
__device__ unsigned k2counter = 0;

__device__ void parallel_pop_from_edge_preallocate_queue(struct edge_block** device_edge_block, unsigned long pop_count, unsigned long* d_prefix_sum_edge_blocks, unsigned long current_vertex, unsigned long thread_blocks, unsigned long edge_blocks_used_present, unsigned long edge_blocks_required) {



    if((d_e_queue.count < pop_count) || (d_e_queue.front == -1)) {
        ;
        // printf("Edge queue empty, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);
        // return NULL;
    }

    else {

        unsigned long start_index;
        // if(id == 0)
        //     start_index = 0;
        // else
        //     start_index = d_prefix_sum_edge_blocks[id - 1];


        start_index = d_prefix_sum_edge_blocks[current_vertex];

        // unsigned long end_index = d_prefix_sum_edge_blocks[id];
        unsigned long end_index = start_index + edge_blocks_required;

        if(start_index < end_index)
            device_edge_block[0] = d_e_queue.edge_block_address[d_e_queue.front + start_index];

        // unsigned long j = 0;

        // // printf("Thread #%lu, start_index is %lu and end_index is %lu\n", id, start_index, end_index);

        // for(unsigned long i = start_index ; i < end_index ; i++) {

        //     device_edge_block[j] = d_e_queue.edge_block_address[d_e_queue.front + i];
        //     d_e_queue.edge_block_address[d_e_queue.front + i] = NULL;

        //     // printf("Popped %p from the edge queue, placeholders front = %ld, rear = %ld\n", device_edge_block[j], d_e_queue.front, d_e_queue.rear);

        //     j++;
        // }

        // if(id == 0) {
        //     printf("Popped %p from the edge queue, start_index = %ld, end_index = %ld\n", device_edge_block[j-1], start_index, end_index);
        //     // printf("Queue address = %p, index is %lu\n", d_e_queue.edge_block_address[0], d_e_queue.front + start_index);
        // }

    }

    __syncthreads();


}

__device__ struct edge_block* pop_edge_block_address(unsigned long pop_count, unsigned long* d_prefix_sum_edge_blocks, unsigned long current_vertex) {



    if((d_e_queue.count < pop_count) || (d_e_queue.front == -1)) {
        // ;
        // printf("Edge queue empty, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);
        return NULL;
    }

    else {

        // unsigned long start_index;

        // start_index = d_prefix_sum_edge_blocks[current_vertex];

        // unsigned long end_index = start_index + edge_blocks_required;

        // if(start_index < end_index)
        return d_e_queue.edge_block_address[d_e_queue.front + d_prefix_sum_edge_blocks[current_vertex]];

    }

    // __syncthreads();


}

__device__ struct edge_block* parallel_pop_from_edge_preallocate_queue_v1(unsigned long pop_count, unsigned long* d_prefix_sum_edge_blocks, unsigned long source_vertex, unsigned long index_counter) {

    // struct edge_block *device_edge_block;



    return d_e_queue.edge_block_address[d_e_queue.front + d_prefix_sum_edge_blocks[source_vertex] + index_counter - 1];
    // return d_e_queue.edge_block_address[d_e_queue.front + index_counter];
}

// __global__ void push_preallocate_list_to_device_queue_kernel(struct vertex_block** d_vertex_preallocate_list, struct edge_block** d_edge_preallocate_list, struct adjacency_sentinel** d_adjacency_sentinel_list, unsigned long vertex_blocks_count_init, unsigned long *edge_blocks_count_init, unsigned long total_edge_blocks_count_init) {
// __global__ void push_preallocate_list_to_device_queue_kernel(struct vertex_block* d_vertex_preallocate_list, struct edge_block* d_edge_preallocate_list, struct adjacency_sentinel** d_adjacency_sentinel_list, unsigned long vertex_blocks_count_init, unsigned long *edge_blocks_count_init, unsigned long total_edge_blocks_count_init) {

//     // some inits, don't run the below two initializations again or queue will fail
//     d_v_queue.front = -1;
//     d_v_queue.rear = -1;
//     d_e_queue.front = -1;
//     d_e_queue.rear = -1;

//     printf("Pushing vertex blocks to vertex queue\n");

//     for(unsigned long i = 0 ; i < vertex_blocks_count_init ; i++) {

//         // printf("%lu -> %p\n", i, d_vertex_preallocate_list[i]);
//         printf("%lu -> %p\n", i, d_vertex_preallocate_list + i);

//         // d_vertex_preallocate_list[i]->active_vertex_count = 1909;

//         // push_to_vertex_preallocate_queue(d_vertex_preallocate_list[i]);
//         push_to_vertex_preallocate_queue(d_vertex_preallocate_list + i);

//     }

//     printf("Pushing edge blocks to edge queue\n");

//     for(unsigned long i = 0 ; i < total_edge_blocks_count_init ; i++) {

//         // printf("%lu -> %p\n", i, d_edge_preallocate_list[i]);
//         printf("%lu -> %p\n", i, d_edge_preallocate_list + i);

//         // push_to_edge_preallocate_queue(d_edge_preallocate_list[i]);
//         push_to_edge_preallocate_queue(d_edge_preallocate_list + i);

//     }

// }

// __device__ unsigned long d_search_flag;

__global__ void data_structure_init(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long *d_vertex_id, unsigned long *d_edge_block_count, unsigned int *d_active_edge_count, unsigned long *d_last_insert_edge_offset, struct edge_block **d_last_insert_edge_block, struct edge_block **d_edge_block_address) {

    d_v_queue.front = -1;
    d_v_queue.rear = -1;
    d_v_queue.count = 0;
    d_e_queue.front = -1;
    d_e_queue.rear = -1;
    d_e_queue.count = 0;

    device_vertex_dictionary->active_vertex_count = 0;

    device_vertex_dictionary->vertex_id = d_vertex_id;
    device_vertex_dictionary->edge_block_count = d_edge_block_count;
    device_vertex_dictionary->active_edge_count = d_active_edge_count;
    device_vertex_dictionary->last_insert_edge_offset = d_last_insert_edge_offset;
    device_vertex_dictionary->last_insert_edge_block = d_last_insert_edge_block;
    device_vertex_dictionary->edge_block_address = d_edge_block_address;
    // printf("At data structure init \n");
    // d_search_flag = 0;

}

// __global__ void parallel_push_vertex_preallocate_list_to_device_queue(struct vertex_block* d_vertex_preallocate_list, struct adjacency_sentinel** d_adjacency_sentinel_list, unsigned long vertex_blocks_count_init) {

//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     if(id < vertex_blocks_count_init) {


//         printf("%lu -> %p\n", id, d_vertex_preallocate_list + id);


//         unsigned long free_blocks = VERTEX_PREALLOCATE_LIST_SIZE - d_v_queue.count;

//         if( (free_blocks < vertex_blocks_count_init) || (d_v_queue.rear + vertex_blocks_count_init) % VERTEX_PREALLOCATE_LIST_SIZE == d_v_queue.front ) {
//             printf("Vertex queue Full, front = %ld, rear = %ld\n", d_v_queue.front, d_v_queue.rear);

//             return;
//         }


//         d_v_queue.vertex_block_address[id] = d_vertex_preallocate_list + id;



//     }

// }

__global__ void parallel_push_edge_preallocate_list_to_device_queue(struct edge_block* d_edge_preallocate_list, struct adjacency_sentinel** d_adjacency_sentinel_list, unsigned long *edge_blocks_count_init, unsigned long total_edge_blocks_count_init) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < total_edge_blocks_count_init) {


        // printf("%lu -> %p\n", id, d_edge_preallocate_list + id);



        unsigned long free_blocks = EDGE_PREALLOCATE_LIST_SIZE - d_e_queue.count;

        if( (free_blocks < total_edge_blocks_count_init) || (d_e_queue.rear + total_edge_blocks_count_init) % EDGE_PREALLOCATE_LIST_SIZE == d_e_queue.front ) {
            // printf("Edge queue Full, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);

            return;
        }

        d_e_queue.edge_block_address[id] = d_edge_preallocate_list + id;


    }

}

__global__ void parallel_push_edge_preallocate_list_to_device_queue_v1(struct edge_block* d_edge_preallocate_list, unsigned long total_edge_blocks_count_init) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < total_edge_blocks_count_init) {


        // printf("%lu -> %p\n", id, d_edge_preallocate_list + id);



        unsigned long free_blocks = EDGE_PREALLOCATE_LIST_SIZE - d_e_queue.count;

        if( (free_blocks < total_edge_blocks_count_init) || (d_e_queue.rear + total_edge_blocks_count_init) % EDGE_PREALLOCATE_LIST_SIZE == d_e_queue.front ) {
            // printf("Edge queue Full, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);

            return;
        }

        d_e_queue.edge_block_address[id] = d_edge_preallocate_list + id;

        // if(id == 0) {
        //     printf("%lu -> %p\n", id, d_edge_preallocate_list + id);
        //     printf("Queue is %p\n", d_e_queue.edge_block_address[0]);
        // }
    }

}

__global__ void parallel_push_queue_update(unsigned long total_edge_blocks_count_init) {

    // if (d_v_queue.front == -1)
    //     d_v_queue.front = 0;

    // d_v_queue.rear = (d_v_queue.rear + vertex_blocks_count_init) % VERTEX_PREALLOCATE_LIST_SIZE;
    // // d_v_queue.vertex_block_address[d_v_queue.rear] = device_vertex_block;
    // d_v_queue.count += vertex_blocks_count_init;

    if (d_e_queue.front == -1)
        d_e_queue.front = 0;

    d_e_queue.rear = (d_e_queue.rear + (unsigned int)total_edge_blocks_count_init) % EDGE_PREALLOCATE_LIST_SIZE;
    // d_v_queue.vertex_block_address[d_v_queue.rear] = device_vertex_block;
    d_e_queue.count += (unsigned int)total_edge_blocks_count_init;

    printf("Queue front is %lu(%p) and rear is %lu(%p)\n", d_e_queue.front, d_e_queue.edge_block_address[d_e_queue.front], d_e_queue.rear, d_e_queue.edge_block_address[d_e_queue.rear]);

}


// __global__ void vertex_dictionary_init(struct vertex_block* d_vertex_preallocate_list, struct adjacency_sentinel* d_adjacency_sentinel_list, unsigned long vertex_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long size, unsigned long *edge_blocks_count_init) {

//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     unsigned int current;
//     __shared__ unsigned int lockvar;
//     lockvar = 0;

//     __syncthreads();

//     d_v_d_sentinel.vertex_count = 99;

//     if(id < vertex_blocks_count_init) {

//         // critical section start



//         struct vertex_block *device_vertex_block;





//         device_vertex_block = parallel_pop_from_vertex_preallocate_queue( vertex_blocks_count_init, id);

//         // parallel test code end

//         // assigning first vertex block to the vertex sentinel node
//         if(id == 0)
//             d_v_d_sentinel.next = device_vertex_block;

//         printf("%lu\n", id);
//         printf("---------\n");

//         printf("%lu -> %p\n", id, device_vertex_block);

//         unsigned long start_index = id * VERTEX_BLOCK_SIZE;
//         unsigned long end_index = (start_index + VERTEX_BLOCK_SIZE - 1) < size ? start_index + VERTEX_BLOCK_SIZE : size;
//         unsigned long j = 0;

//         // device_vertex_block->active_vertex_count = 0;

//         for( unsigned long i = start_index ; i < end_index ; i++ ) {

//             device_vertex_block->vertex_id[j] = i + 1;
//             device_vertex_block->active_vertex_count++;

//             // device_vertex_block->vertex_adjacency[j] = d_adjacency_sentinel_list[i];
//             device_vertex_block->vertex_adjacency[j] = d_adjacency_sentinel_list + i;

//             device_vertex_block->edge_block_count[j] = edge_blocks_count_init[i];

//             printf("%lu from thread %lu, start = %lu and end = %lu\n", device_vertex_block->vertex_id[j], id, start_index, end_index);
//             j++;

//         }


//         if(id < vertex_blocks_count_init - 1)
//             (d_vertex_preallocate_list + id)->next = d_vertex_preallocate_list + id + 1;
//         else
//             (d_vertex_preallocate_list + id)->next = NULL;


//     }

// }

// __global__ void parallel_vertex_dictionary_init(struct adjacency_sentinel* d_adjacency_sentinel_list, unsigned long vertex_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long vertex_size, unsigned long *edge_blocks_count_init, struct vertex_dictionary_structure *device_vertex_dictionary) {

//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     if(id < vertex_size) {



//         if(id == 0)
//             device_vertex_dictionary->active_vertex_count += vertex_size;

//         device_vertex_dictionary->vertex_id[id] = id + 1;
//         // device_vertex_dictionary->active_vertex_count++;
//         device_vertex_dictionary->vertex_adjacency[id] = d_adjacency_sentinel_list + id;
//         device_vertex_dictionary->edge_block_count[id] = edge_blocks_count_init[id];




//     }

// }

__global__ void parallel_vertex_dictionary_init_v1(struct adjacency_sentinel_new* d_adjacency_sentinel_list, unsigned long vertex_size, unsigned long *edge_blocks_count_init, struct vertex_dictionary_structure *device_vertex_dictionary) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {



        if(id == 0)
            device_vertex_dictionary->active_vertex_count += vertex_size;

        device_vertex_dictionary->vertex_id[id] = id + 1;
        // device_vertex_dictionary->active_vertex_count++;
        // device_vertex_dictionary->vertex_adjacency[id] = d_adjacency_sentinel_list + id;
        // device_vertex_dictionary->edge_block_count[id] = edge_blocks_count_init[id];




    }

    // __syncthreads();

    // if(id == 0)
    //     printf("Checkpoint VD\n");

}

__device__ unsigned int lockvar = 0;


// __global__ void adjacency_list_init(struct edge_block** d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long *d_source, unsigned long *d_destination, unsigned long total_edge_blocks_count_init, unsigned long vertex_size, unsigned long edge_size) {
// __global__ void adjacency_list_init(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long *d_source, unsigned long *d_destination, unsigned long total_edge_blocks_count_init, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks) {


//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     unsigned int current;
//     // __shared__ unsigned int lockvar;
//     // lockvar = 0;

//     __syncthreads();

//     // printf("K1 counter = %u\n", atomicAdd((unsigned *)&k1counter, 1));

//     if(id < vertex_size) {

//         printf("Prefix sum of %ld is %ld\n", id, d_prefix_sum_edge_blocks[id]);

//         unsigned long current_vertex = id + 1;
//         unsigned long index = 0;
//         struct vertex_block *ptr = d_v_d_sentinel.next;

//         // locating vertex in the vertex dictionary
//         while(ptr != NULL) {

//             unsigned long flag = 1;

//             for(unsigned long i = 0 ; i < VERTEX_BLOCK_SIZE ; i++) {

//                 if(ptr->vertex_id[i] == current_vertex) {
//                     index = i;
//                     flag = 0;
//                     printf("Matched at %lu\n", index);
//                     break;
//                 }

//             }

//             if(flag)
//                 ptr = ptr->next;
//             else
//                 break;


//         }

//         if(ptr != NULL)
//             printf("ID = %lu, Vertex = %lu, Adjacency Sentinel = %p and edge blocks = %lu\n", id, ptr->vertex_id[index], ptr->vertex_adjacency[index], ptr->edge_block_count[index]);

//         unsigned long edge_blocks_required = ptr->edge_block_count[index];

//         // critical section start

//         // temporary fix, this can't be a constant sized one
//         struct edge_block *device_edge_block[100];


//         // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {
//         parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_init, d_prefix_sum_edge_blocks, id, thread_blocks);

//         // critical section end
//         if(threadIdx.x == 0)
//             printf("ID\tIteration\tGPU address\n");

//         for(unsigned long i = 0 ; i < edge_blocks_required ; i++)
//             printf("%lu\t%lu\t\t%p\n", id, i, device_edge_block[i]);

//         // adding the edge blocks
//         struct adjacency_sentinel *vertex_adjacency = ptr->vertex_adjacency[index];

//         if(edge_blocks_required > 0) {

//             struct edge_block *prev, *curr;

//             prev = NULL;
//             curr = NULL;


//             for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

//                 curr = device_edge_block[i];

//                 if(prev != NULL)
//                     prev->next = curr;

//                 prev = curr;

//             }

//             if(edge_blocks_required > 0) {
//                 vertex_adjacency->next = device_edge_block[0];
//                 curr->next = NULL;
//             }

//             unsigned long edge_block_entry_count = 0;
//             unsigned long edge_block_counter = 0;

//             curr = vertex_adjacency->next;
//             vertex_adjacency->active_edge_count = 0;

//             for(unsigned long i = 0 ; i < edge_size ; i++) {

//                 if(d_source[i] == current_vertex){

//                     // printf("Source = %lu and Destination = %lu\n", d_source[i], d_destination[i]);

//                     // insert here
//                     curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_destination[i];
//                     vertex_adjacency->active_edge_count++;

//                     if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {

//                         curr = curr->next;
//                         edge_block_counter++;
//                         edge_block_entry_count = 0;
//                     }

//                 }

//             }

//         }

//     }

// }

// __global__ void adjacency_list_init_modded(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long *d_source, unsigned long *d_destination, unsigned long total_edge_blocks_count_init, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks, struct vertex_dictionary_structure *device_vertex_dictionary) {

//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     unsigned int current;
//     // __shared__ unsigned int lockvar;
//     // lockvar = 0;

//     __syncthreads();

//     // printf("K1 counter = %u\n", atomicAdd((unsigned *)&k1counter, 1));

//     if(id < vertex_size) {

//         printf("Prefix sum of %ld is %ld\n", id, d_prefix_sum_edge_blocks[id]);

//         unsigned long current_vertex = id + 1;

//         unsigned long edge_blocks_required = device_vertex_dictionary->edge_block_count[id];

//         // critical section start

//         // temporary fix, this can't be a constant sized one
//         struct edge_block *device_edge_block[100];


//         parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_init, d_prefix_sum_edge_blocks, id, thread_blocks);

//         // critical section end
//         if(threadIdx.x == 0)
//             printf("ID\tIteration\tGPU address\n");

//         for(unsigned long i = 0 ; i < edge_blocks_required ; i++)
//             printf("%lu\t%lu\t\t%p\n", id, i, device_edge_block[i]);

//         // adding the edge blocks
//         // struct adjacency_sentinel *vertex_adjacency = ptr->vertex_adjacency[index];
//         struct adjacency_sentinel *vertex_adjacency = device_vertex_dictionary->vertex_adjacency[id];

//         if(edge_blocks_required > 0) {

//             struct edge_block *prev, *curr;

//             prev = NULL;
//             curr = NULL;


//             for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

//                 curr = device_edge_block[i];

//                 if(prev != NULL)
//                     prev->next = curr;

//                 prev = curr;

//             }

//             if(edge_blocks_required > 0) {
//                 vertex_adjacency->next = device_edge_block[0];
//                 curr->next = NULL;
//             }

//             unsigned long edge_block_entry_count = 0;
//             unsigned long edge_block_counter = 0;

//             curr = vertex_adjacency->next;
//             vertex_adjacency->active_edge_count = 0;

//             for(unsigned long i = 0 ; i < edge_size ; i++) {

//                 if(d_source[i] == current_vertex){

//                     // printf("Source = %lu and Destination = %lu\n", d_source[i], d_destination[i]);

//                     // insert here
//                     curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_destination[i];
//                     vertex_adjacency->active_edge_count++;

//                     if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {

//                         curr = curr->next;
//                         edge_block_counter++;
//                         edge_block_entry_count = 0;
//                     }

//                 }

//             }

//         }

//     }

// }

// __global__ void adjacency_list_init_modded_v1(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long *d_source, unsigned long *d_destination, unsigned long total_edge_blocks_count_init, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_prefix_sum_vertex_degrees, unsigned long thread_blocks, struct vertex_dictionary_structure *device_vertex_dictionary) {

//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     unsigned int current;
//     // __shared__ unsigned int lockvar;
//     // lockvar = 0;

//     __syncthreads();


//     if(id < vertex_size) {

//         printf("Prefix sum of %ld is %ld\n", id, d_prefix_sum_edge_blocks[id]);

//         unsigned long current_vertex = id + 1;

//         // unsigned long edge_blocks_required = ptr->edge_block_count[index];
//         unsigned long edge_blocks_required = device_vertex_dictionary->edge_block_count[id];

//         // critical section start

//         // temporary fix, this can't be a constant sized one
//         struct edge_block *device_edge_block[100];




//         // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {
//         parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_init, d_prefix_sum_edge_blocks, id, thread_blocks);


//         // critical section end
//         if(threadIdx.x == 0)
//             printf("ID\tIteration\tGPU address\n");

//         for(unsigned long i = 0 ; i < edge_blocks_required ; i++)
//             printf("%lu\t%lu\t\t%p\n", id, i, device_edge_block[i]);

//         // adding the edge blocks
//         // struct adjacency_sentinel *vertex_adjacency = ptr->vertex_adjacency[index];
//         struct adjacency_sentinel_new *vertex_adjacency = device_vertex_dictionary->vertex_adjacency[id];

//         if(edge_blocks_required > 0) {

//             struct edge_block *prev, *curr;

//             prev = NULL;
//             curr = NULL;


//             for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

//                 curr = device_edge_block[i];

//                 if(prev != NULL)
//                     prev->next = curr;

//                 prev = curr;

//             }

//             if(edge_blocks_required > 0) {
//                 vertex_adjacency->next = device_edge_block[0];
//                 curr->next = NULL;
//             }

//             unsigned long edge_block_entry_count = 0;
//             unsigned long edge_block_counter = 0;

//             curr = vertex_adjacency->next;
//             vertex_adjacency->active_edge_count = 0;

//             unsigned long start_index;

//             if(current_vertex != 1)
//                 start_index = d_prefix_sum_vertex_degrees[current_vertex - 2];
//             else
//                 start_index = 0;

//             unsigned long end_index = d_prefix_sum_vertex_degrees[current_vertex - 1];

//             printf("Current vertex = %lu, start = %lu, end = %lu\n", current_vertex, start_index, end_index);



//             for(unsigned long i = start_index ; i < end_index ; i++) {

//                 // if(d_source[i] == current_vertex){

//                     // printf("Source = %lu and Destination = %lu\n", d_source[i], d_destination[i]);

//                     // insert here
//                     curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_destination[i];
//                     vertex_adjacency->active_edge_count++;

//                     if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {

//                         curr = curr->next;
//                         edge_block_counter++;
//                         edge_block_entry_count = 0;
//                     }

//                 // }

//             }

//         }

//     }

// }

// __global__ void adjacency_list_init_modded_v2(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, unsigned long *d_source, unsigned long *d_destination, unsigned long total_edge_blocks_count_init, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_prefix_sum_vertex_degrees, unsigned long thread_blocks, struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long batch_number, unsigned long batch_size) {

//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     unsigned int current;
//     // __shared__ unsigned int lockvar;
//     // lockvar = 0;

//     __syncthreads();


//     if(id < vertex_size) {

//         // printf("Prefix sum of %ld is %ld\n", id, d_prefix_sum_edge_blocks[id]);

//         unsigned long current_vertex = id + 1;

//         // unsigned long edge_blocks_required = ptr->edge_block_count[index];
//         unsigned long edge_blocks_required = device_vertex_dictionary->edge_block_count[id];

//         // critical section start

//         struct edge_block *device_edge_block_base;

//         if(batch_number == 0) {
//             // temporary fix, this can't be a constant sized one
//             // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {
//             struct edge_block *device_edge_block[4];
//             parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_init, d_prefix_sum_edge_blocks, id, thread_blocks);
//             device_edge_block_base = device_edge_block[0];
//         }

//         // critical section end
//         // if(threadIdx.x == 0)
//         //     printf("ID\tIteration\tGPU address\n");

//         // if(id == 0) {

//         //     for(unsigned long i = 0 ; i < batch_size ; i++)
//         //         printf("%lu and %lu\n", d_source[i], d_destination[i]);

//         // }

//         // for(unsigned long i = 0 ; i < edge_blocks_required ; i++)
//         //     printf("%lu\t%lu\t\t%p\n", id, i, device_edge_block[i]);

//         // adding the edge blocks
//         // struct adjacency_sentinel *vertex_adjacency = ptr->vertex_adjacency[index];
//         struct adjacency_sentinel_new *vertex_adjacency = device_vertex_dictionary->vertex_adjacency[id];

//         if(edge_blocks_required > 0) {

//             // one thread inserts sequentially all destination vertices of a source vertex.

//             struct edge_block *prev, *curr;

//             // prev = NULL;
//             // curr = NULL;


//             // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

//             //     curr = device_edge_block[i];

//             //     if(prev != NULL)
//             //         prev->next = curr;

//             //     prev = curr;

//             // }

//             // if(edge_blocks_required > 0) {
//             //     vertex_adjacency->next = device_edge_block[0];
//             //     curr->next = NULL;
//             // }

//             // unsigned long edge_block_entry_count = 0;
//             // unsigned long edge_block_counter = 0;
//             unsigned long edge_block_entry_count;
//             unsigned long edge_block_counter;

//             if(batch_number == 0) {

//                 edge_block_entry_count = 0;
//                 edge_block_counter = 0;
//                 vertex_adjacency->active_edge_count = 0;

//                 // curr = device_edge_block[edge_block_counter];
//                 curr = device_edge_block_base;
//                 // vertex_adjacency->edge_block_address[edge_block_counter] = curr;
//                 vertex_adjacency->edge_block_address = curr;
//                 vertex_adjacency->last_insert_edge_block = curr;

//             }

//             else {

//                 edge_block_entry_count = vertex_adjacency->last_insert_edge_offset;
//                 edge_block_counter = vertex_adjacency->edge_block_count;
//                 curr = vertex_adjacency->last_insert_edge_block;
//                 curr->active_edge_count = 0;


//             }

//             // curr = vertex_adjacency->next;

//             unsigned long start_index;

//             if(current_vertex != 1)
//                 start_index = d_prefix_sum_vertex_degrees[current_vertex - 2];
//             else
//                 start_index = 0;

//             unsigned long end_index = d_prefix_sum_vertex_degrees[current_vertex - 1];

//             // printf("Current vertex = %lu, start = %lu, end = %lu\n", current_vertex, start_index, end_index);

//             // unsigned long current_edge_block = 0;
//             // curr = device_edge_block[edge_block_counter];
//             // // vertex_adjacency->edge_block_address[edge_block_counter] = curr;
//             // vertex_adjacency->edge_block_address = curr;
//             // vertex_adjacency->last_insert_edge_block = curr;

//             // __syncthreads();

//             // if(id == 0)
//             //     printf("Checkpoint AL beg\n");

//             // unsigned long edge_counter = 0;

//             // if(id == 0) {
//             //     // printf("Edge counter is %lu\n", vertex_adjacency->active_edge_count);
//             //     printf("Start index is %lu and end index is %lu\n", start_index, end_index);
//             //     printf("Device edge block address is %p\n", curr);
//             // }

//             for(unsigned long i = start_index ; i < end_index ; i++) {

//                 // printf("Checkpoint 0\n");

//                 // if(d_source[i] == current_vertex){

//                 // printf("Thread = %lu and Source = %lu and Destination = %lu\n", id, d_source[i], d_destination[i]);
//                 // printf("Checkpoint 1\n");

//                 // insert here

//                 // if(curr == NULL)
//                 //     printf("Hit here at %lu\n", id);

//                 curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_destination[i];
//                 // printf("Checkpoint 2\n");

//                 // if(id == 0) {
//                 //     printf("Entry is %lu\n", curr->edge_block_entry[edge_block_entry_count].destination_vertex);
//                 // }
//                 // printf("Entry is %lu\n", curr->edge_block_entry[edge_block_entry_count].destination_vertex);
//                 vertex_adjacency->active_edge_count++;
//                 curr->active_edge_count++;
//                 vertex_adjacency->last_insert_edge_offset++;
//                 // printf("Checkpoint 3\n");

//                 if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {

//                     // curr = curr->next;
//                     // edge_block_counter++;
//                     edge_block_entry_count = 0;
//                     curr = curr + 1;
//                     // curr = device_edge_block[++edge_block_counter];
//                     curr->active_edge_count = 0;
//                     vertex_adjacency->last_insert_edge_block = curr;
//                     vertex_adjacency->edge_block_count++;
//                     vertex_adjacency->last_insert_edge_offset = 0;
//                     // vertex_adjacency->edge_block_address[edge_block_counter] = curr;

//                 }

//                 // }

//             }

//             // printf("Success\n");

//         }

//     }

//     // __syncthreads();

//     // if(id == 0)
//     //     printf("Checkpoint AL\n");

// }

// __global__ void adjacency_list_init_modded_v3(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, unsigned long total_edge_blocks_count_init, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks, struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long batch_number, unsigned long batch_size, unsigned long start_index_batch, unsigned long end_index_batch, unsigned long* d_csr_offset, unsigned long* d_csr_edges) {

//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     unsigned int current;
//     // __shared__ unsigned int lockvar;
//     // lockvar = 0;

//     __syncthreads();


//     if(id < vertex_size) {

//         // printf("Prefix sum of %ld is %ld\n", id, d_prefix_sum_edge_blocks[id]);

//         unsigned long current_vertex = id + 1;

//         // unsigned long edge_blocks_required = ptr->edge_block_count[index];
//         unsigned long edge_blocks_required = device_vertex_dictionary->edge_block_count[id];

//         // critical section start

//         struct edge_block *device_edge_block_base;

//         if(batch_number == 0) {
//             // temporary fix, this can't be a constant sized one
//             // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {
//             struct edge_block *device_edge_block[4];
//             parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_init, d_prefix_sum_edge_blocks, id, thread_blocks);
//             device_edge_block_base = device_edge_block[0];
//         }

//         // critical section end
//         // if(threadIdx.x == 0)
//         //     printf("ID\tIteration\tGPU address\n");

//         // if(id == 0) {

//         //     for(unsigned long i = 0 ; i < batch_size ; i++)
//         //         printf("%lu and %lu\n", d_source[i], d_destination[i]);

//         // }

//         // for(unsigned long i = 0 ; i < edge_blocks_required ; i++)
//         //     printf("%lu\t%lu\t\t%p\n", id, i, device_edge_block[i]);

//         // adding the edge blocks
//         // struct adjacency_sentinel *vertex_adjacency = ptr->vertex_adjacency[index];
//         struct adjacency_sentinel_new *vertex_adjacency = device_vertex_dictionary->vertex_adjacency[id];

//         if(edge_blocks_required > 0) {

//             // one thread inserts sequentially all destination vertices of a source vertex.

//             struct edge_block *prev, *curr;

//             // prev = NULL;
//             // curr = NULL;


//             // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

//             //     curr = device_edge_block[i];

//             //     if(prev != NULL)
//             //         prev->next = curr;

//             //     prev = curr;

//             // }

//             // if(edge_blocks_required > 0) {
//             //     vertex_adjacency->next = device_edge_block[0];
//             //     curr->next = NULL;
//             // }

//             // unsigned long edge_block_entry_count = 0;
//             // unsigned long edge_block_counter = 0;
//             unsigned long edge_block_entry_count;
//             unsigned long edge_block_counter;

//             if(batch_number == 0) {

//                 edge_block_entry_count = 0;
//                 edge_block_counter = 0;
//                 vertex_adjacency->active_edge_count = 0;

//                 // curr = device_edge_block[edge_block_counter];
//                 curr = device_edge_block_base;
//                 // vertex_adjacency->edge_block_address[edge_block_counter] = curr;
//                 vertex_adjacency->edge_block_address = curr;
//                 vertex_adjacency->last_insert_edge_block = curr;

//             }

//             else {

//                 edge_block_entry_count = vertex_adjacency->last_insert_edge_offset;
//                 edge_block_counter = vertex_adjacency->edge_block_count;
//                 curr = vertex_adjacency->last_insert_edge_block;
//                 curr->active_edge_count = 0;


//             }

//             // curr = vertex_adjacency->next;

//             // test code v3 start

//             unsigned long start_index = start_index_batch;
//             unsigned long end_index = end_index_batch;

//             if(start_index_batch <= d_csr_offset[id])
//                 start_index = d_csr_offset[id];
//             // else
//             //     end_index = end_index_batch;

//             // test code v3 end

//             if(end_index_batch >= d_csr_offset[id + 1])
//                 end_index = d_csr_offset[id + 1];
//             // else
//             //     end_index = end_index_batch;

//             // test code v3 end

//             // unsigned long start_index;

//             // if(current_vertex != 1)
//             //     start_index = d_prefix_sum_vertex_degrees[current_vertex - 2];
//             // else
//             //     start_index = 0;

//             // unsigned long end_index = d_prefix_sum_vertex_degrees[current_vertex - 1];

//             // printf("Current vertex = %lu, start = %lu, end = %lu\n", current_vertex, start_index, end_index);

//             // unsigned long current_edge_block = 0;
//             // curr = device_edge_block[edge_block_counter];
//             // // vertex_adjacency->edge_block_address[edge_block_counter] = curr;
//             // vertex_adjacency->edge_block_address = curr;
//             // vertex_adjacency->last_insert_edge_block = curr;

//             // __syncthreads();

//             // if(id == 0)
//             //     printf("Checkpoint AL beg\n");

//             // unsigned long edge_counter = 0;

//             // if(id == 0) {
//             //     // printf("Edge counter is %lu\n", vertex_adjacency->active_edge_count);
//             //     printf("Start index is %lu and end index is %lu\n", start_index, end_index);
//             //     printf("Device edge block address is %p\n", curr);
//             // }

//             for(unsigned long i = start_index ; i < end_index ; i++) {

//                 // printf("Checkpoint 0\n");

//                 // if(d_source[i] == current_vertex){

//                 // printf("Thread = %lu and Source = %lu and Destination = %lu\n", id, d_source[i], d_destination[i]);
//                 // printf("Checkpoint 1\n");

//                 // insert here

//                 // if(curr == NULL)
//                 //     printf("Hit here at %lu\n", id);
//                 // curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_destination[i];
//                 curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_csr_edges[i];
//                 // printf("Checkpoint 2\n");

//                 // if(id == 0) {
//                 //     printf("Entry is %lu\n", curr->edge_block_entry[edge_block_entry_count].destination_vertex);
//                 // }
//                 // printf("Entry is %lu\n", curr->edge_block_entry[edge_block_entry_count].destination_vertex);
//                 vertex_adjacency->active_edge_count++;
//                 curr->active_edge_count++;
//                 vertex_adjacency->last_insert_edge_offset++;
//                 // printf("Checkpoint 3\n");

//                 if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {

//                     // curr = curr->next;
//                     // edge_block_counter++;
//                     edge_block_entry_count = 0;
//                     curr = curr + 1;
//                     // curr = device_edge_block[++edge_block_counter];
//                     curr->active_edge_count = 0;
//                     vertex_adjacency->last_insert_edge_block = curr;
//                     vertex_adjacency->edge_block_count++;
//                     vertex_adjacency->last_insert_edge_offset = 0;
//                     // vertex_adjacency->edge_block_address[edge_block_counter] = curr;

//                 }

//                 // }

//             }

//             // printf("Success\n");

//         }

//     }

//     // __syncthreads();

//     // if(id == 0)
//     //     printf("Checkpoint AL\n");

// }

__device__ void insert_edge_block_to_CBT(struct edge_block *root, unsigned long bit_string, unsigned long length, struct edge_block *new_block, struct edge_block *last_insert_edge_block, unsigned long i) {

    struct edge_block *curr = root;

    // if(length > 0) {

    // for(unsigned long i = 0 ; i < length - 1 ; i++) {
    for( ; bit_string > 10  ; bit_string /= 10) {

        // if(bit_string % 2)
        //     curr = curr->rptr;
        // else
        //     curr = curr->lptr;

        if(bit_string % 2)
            curr = curr->lptr;
        else
            curr = curr->rptr;

        // bit_string /= 10;

    }

    // }

    new_block->lptr = NULL;
    new_block->rptr = NULL;

    if(i)
        new_block->level_order_predecessor = new_block - 1;
    else
        new_block->level_order_predecessor = last_insert_edge_block;

    // printf("Checkpoint\n");

    if(bit_string % 2)
        curr->lptr = new_block;
    else
        curr->rptr = new_block;

}

__device__ void insert_edge_block_to_CBT_v2(struct edge_block *root, unsigned long bit_string, unsigned long length, struct edge_block *new_block, struct edge_block *last_insert_edge_block, unsigned long batch_index_counter, unsigned long global_index_counter, unsigned long current_edge_block_count, unsigned long total_edge_blocks_new, unsigned long source_vertex, unsigned long id) {

    struct edge_block *curr = root;

    // indices from 0
    unsigned long current_capacity = (current_edge_block_count) * 2 + 2;

    // printf("ID=%lu, source=%lu, batch_index_counter=%lu, global_index_counter=%lu, current_capacity=%lu, current_edge_block_count=%lu, total_edge_blocks_new=%lu\n", id, source_vertex, batch_index_counter, global_index_counter, current_capacity, current_edge_block_count, total_edge_blocks_new);

    if(global_index_counter <= current_capacity) {

        // if(length > 0) {

        // for(unsigned long i = 0 ; i < length - 1 ; i++) {
        for( ; bit_string > 10  ; bit_string /= 10) {

            // if(bit_string % 2)
            //     curr = curr->rptr;
            // else
            //     curr = curr->lptr;

            if(bit_string % 2)
                curr = curr->lptr;
            else
                curr = curr->rptr;

            // bit_string /= 10;

        }

        // }

        // if(batch_index_counter)
        //     new_block->level_order_predecessor = new_block - 1;
        // else
        //     new_block->level_order_predecessor = last_insert_edge_block;

        // printf("Checkpoint\n");

        if(bit_string % 2)
            curr->lptr = new_block;
        else
            curr->rptr = new_block;

        // if((global_index_counter * 2) <= total_edge_blocks_new) {

        //     new_block->lptr = NULL;
        //     new_block->rptr = NULL;

        // }

        // else if(((global_index_counter * 2) + 2) <= total_edge_blocks_new) {

        //     new_block->lptr = curr + (global_index_counter * 2) + 1 - global_index_counter;
        //     new_block->rptr = curr + (global_index_counter * 2) + 2 - global_index_counter;

        // }

        // else {

        //     new_block->lptr = curr + (global_index_counter * 2) + 1 - global_index_counter;

        // }
    }

    if(batch_index_counter)
        new_block->level_order_predecessor = new_block - 1;
    else
        new_block->level_order_predecessor = last_insert_edge_block;

    // if((global_index_counter * 2) <= total_edge_blocks_new) {

    //     new_block->lptr = NULL;
    //     new_block->rptr = NULL;

    // }

    if(((global_index_counter * 2) + 2) <= total_edge_blocks_new) {

        new_block->lptr = new_block + (global_index_counter * 2) + 1 - global_index_counter;
        new_block->rptr = new_block + (global_index_counter * 2) + 2 - global_index_counter;
        // printf("ID=%lu was here, first if\n", id);

    }

    else if(((global_index_counter * 2) + 1) <= total_edge_blocks_new) {

        new_block->lptr = new_block + (global_index_counter * 2) + 1 - global_index_counter;
        new_block->rptr = NULL;
        // printf("ID=%lu was here, second if, curr=%p, val=%lu, new_block->lptr=%p\n", id, new_block, (global_index_counter * 2) + 1 - global_index_counter, new_block->lptr);

    }

    else {

        new_block->lptr = NULL;
        new_block->rptr = NULL;
        // printf("ID=%lu was here, third if\n", id);

    }

    // else {

    //     unsigned long parent = ceil(double(global_index_counter) / 2) - 1;

    // }

}

__device__ void inorderTraversalTemp(struct edge_block *root) {

    if(root == NULL) {
        printf("Hit\n");
        return;
    }

    else {

        printf("Hit 1\n");
        printf("Root %p contents are %lu and %lu, pointers are %p and %p, active_edge_count is %lu\n", root, root->edge_block_entry[0].destination_vertex, root->edge_block_entry[1].destination_vertex, root->lptr, root->rptr, root->active_edge_count);
        printf("Hit 2\n");

        inorderTraversalTemp(root->lptr);

        printf("\nedge block edge count = %lu, ", root->active_edge_count);

        for(unsigned long j = 0 ; j < root->active_edge_count ; j++) {

            printf("%lu ", root->edge_block_entry[j].destination_vertex);

            // edge_block_entry_count++;

            // if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
            //     // itr = itr->next;
            //     // itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[++edge_block_counter];
            //     itr = itr + 1;
            //     edge_block_entry_count = 0;
            // }
        }

        // printf("\n");

        // inorderTraversalTemp(root->rptr);

    }

    // unsigned long edge_block_counter = 0;
    // unsigned long edge_block_entry_count = 0;
    // // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[edge_block_counter];
    // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address;


    // for(unsigned long j = 0 ; j < device_vertex_dictionary->vertex_adjacency[i]->active_edge_count ; j++) {

    //     printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);

    //     // edge_block_entry_count++;

    //     if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
    //         // itr = itr->next;
    //         // itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[++edge_block_counter];
    //         itr = itr + 1;
    //         edge_block_entry_count = 0;
    //     }
    // }

}

__global__ void adjacency_list_init_modded_v4(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, unsigned long total_edge_blocks_count_init, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks, struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long batch_number, unsigned long batch_size, unsigned long start_index_batch, unsigned long end_index_batch, unsigned long* d_csr_offset, unsigned long* d_csr_edges) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int current;
    // __shared__ unsigned int lockvar;
    // lockvar = 0;

    __syncthreads();


    if(id < vertex_size) {

        // printf("Prefix sum of %ld is %ld\n", id, d_prefix_sum_edge_blocks[id]);

        unsigned long current_vertex = id + 1;

        // unsigned long edge_blocks_required = ptr->edge_block_count[index];
        // unsigned long edge_blocks_required = device_vertex_dictionary->edge_block_count[id];
        unsigned long edge_blocks_required = 0;

        // critical section start

        struct edge_block *device_edge_block_base;

        // v4 test code start

        struct adjacency_sentinel_new *vertex_adjacency = device_vertex_dictionary->vertex_adjacency[id];
        struct edge_block *curr;

        unsigned long space_remaining = 0;
        unsigned long edge_blocks_used_present = 0;

        if((vertex_adjacency->edge_block_address != NULL) && (vertex_adjacency->last_insert_edge_offset != 0)) {

            space_remaining = EDGE_BLOCK_SIZE - device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset;

        }

        edge_blocks_used_present = device_vertex_dictionary->vertex_adjacency[id]->edge_block_count;

        // printf("%lu\n", edge_blocks_required);
        unsigned long start_index = start_index_batch;
        unsigned long end_index = end_index_batch;

        unsigned long new_edges_count = 0;
        // if(end_index > start_index)
        //     new_edges_count = end_index - start_index;

        if(start_index_batch <= d_csr_offset[id])
            start_index = d_csr_offset[id];
        // else
        //     end_index = end_index_batch;

        // test code v3 end

        if(end_index_batch >= d_csr_offset[id + 1])
            end_index = d_csr_offset[id + 1];

        // below check since end_index and start_index are unsigned longs
        if(end_index > start_index) {

            // if((end_index - start_index) > space_remaining) {
            new_edges_count = end_index - start_index;
            // edge_blocks_required = ceil(double((end_index - start_index) - space_remaining) / EDGE_BLOCK_SIZE);
            if(new_edges_count > space_remaining)
                edge_blocks_required = ceil(double(new_edges_count - space_remaining) / EDGE_BLOCK_SIZE);
            // else
            //     edge_blocks_required = ceil(double(new_edges_count - space_remaining) / EDGE_BLOCK_SIZE);
            // }

        }

        else
            edge_blocks_required = 0;

        // printf("thread #%lu, start_index is %lu, end_index is %lu, end_index_batch is %lu, d_csr_offset[id+1] is %lu, edge_blocks_required is %lu, edge_blocks_used is %lu, space_remaining is %lu\n", id, start_index, end_index, end_index_batch, d_csr_offset[id + 1], edge_blocks_required, edge_blocks_used_present, space_remaining);

        // printf("Checkpointer 1\n");

        // printf("%lu\n", edge_blocks_required);

        // v4 test code end

        struct edge_block *device_edge_block[PREALLOCATE_EDGE_BLOCKS];

        if(edge_blocks_required > 0) {

            parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_init, d_prefix_sum_edge_blocks, id, thread_blocks, edge_blocks_used_present, edge_blocks_required);
            vertex_adjacency->edge_block_count += edge_blocks_required;
            device_edge_block_base = device_edge_block[0];

        }




        // printf("Checkpointer 2\n");

        // create complete binary tree for first time
        if((edge_blocks_required > 0) && (vertex_adjacency->edge_block_address == NULL)) {

            curr = device_edge_block[0];
            // vertex_adjacency->edge_block_address = curr;

            // unsigned long array[] = {1,2,3,4,5,6,7,8,9,10,11,12};
            // unsigned long len = 12;

            // struct node *addresses = (struct node*)malloc(len * sizeof(struct node));

            // unsigned long inserted = 0;

            // struct node *root = addresses;
            // struct node *curr = root;

            unsigned long i = 0;
            unsigned long curr_index = 0;

            if(edge_blocks_required > 1) {

                for(i = 0 ; i < (edge_blocks_required / 2) - 1 ; i++) {

                    // curr->value = array[i];
                    curr->lptr = *(device_edge_block + (2 * i) + 1);
                    curr->rptr = *(device_edge_block + (2 * i) + 2);

                    // printf("Inserted internal node %p\n", curr);

                    curr_index++;
                    curr = *(device_edge_block + curr_index);

                }

                if(edge_blocks_required % 2) {

                    // curr->value = array [i];
                    curr->lptr = *(device_edge_block + (2 * i) + 1);
                    curr->rptr = *(device_edge_block + (2 * i++) + 2);

                    // printf("Inserted internal node v1 %p\n", curr);

                }
                else {
                    // curr->value = array [i];
                    curr->lptr = *(device_edge_block + (2 * i++) + 1);

                    // printf("Inserted internal node v2 %p\n", curr);

                }

                curr_index++;
                curr = *(device_edge_block + curr_index);

            }

            // printf("Checkpoint %lu\n", edge_blocks_required);

            for( ; i < edge_blocks_required ; i++) {

                // curr->value = array[i];
                curr->lptr = NULL;
                curr->rptr = NULL;

                // printf("Inserted leaf node %p\n", curr);

                curr_index++;
                curr = *(device_edge_block + curr_index);

            }

        }

        else if((edge_blocks_required > 0) && (vertex_adjacency->edge_block_address != NULL)) {

            // printf("Checkpoint update %lu\n", edge_blocks_required);

            for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

                unsigned long current_edge_block = vertex_adjacency->edge_block_count - edge_blocks_required + i + 1;
                unsigned long length = 0;
                unsigned long bit_string = traversal_string(current_edge_block, &length);

                // printf("Disputed address %p\n", device_edge_block[i]);

                // printf("Checkpoint cbt1 for thread #%lu, bit_string for %lu is %lu with length %lu\n", id, current_edge_block, bit_string, length);
                // insert_edge_block_to_CBT(vertex_adjacency->edge_block_address, bit_string, length, device_edge_block[i]);
                // printf("Checkpoint cbt2\n");

            }

        }

        // if(batch_number == 0) {
        //     // temporary fix, this can't be a constant sized one
        //     // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {
        //     struct edge_block *device_edge_block[4];
        //     parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_init, d_prefix_sum_edge_blocks, id, thread_blocks);
        //     device_edge_block_base = device_edge_block[0];
        // }

        // critical section end
        // if(threadIdx.x == 0)
        //     printf("ID\tIteration\tGPU address\n");

        // if(id == 0) {

        //     for(unsigned long i = 0 ; i < batch_size ; i++)
        //         printf("%lu and %lu\n", d_source[i], d_destination[i]);

        // }

        // for(unsigned long i = 0 ; i < edge_blocks_required ; i++)
        //     printf("%lu\t%lu\t\t%p\n", id, i, device_edge_block[i]);

        // adding the edge blocks
        // struct adjacency_sentinel *vertex_adjacency = ptr->vertex_adjacency[index];

        if(new_edges_count > 0) {

            // one thread inserts sequentially all destination vertices of a source vertex.

            // struct edge_block *prev, *curr;

            // prev = NULL;
            // curr = NULL;


            // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

            //     curr = device_edge_block[i];

            //     if(prev != NULL)
            //         prev->next = curr;

            //     prev = curr;

            // }

            // if(edge_blocks_required > 0) {
            //     vertex_adjacency->next = device_edge_block[0];
            //     curr->next = NULL;
            // }

            // unsigned long edge_block_entry_count = 0;
            // unsigned long edge_block_counter = 0;
            unsigned long edge_block_entry_count;
            unsigned long edge_block_counter;


            if(vertex_adjacency->edge_block_address == NULL) {

                edge_block_entry_count = 0;
                edge_block_counter = 0;
                vertex_adjacency->active_edge_count = 0;
                // vertex_adjacency->edge_block_count = 1;

                // curr = device_edge_block[edge_block_counter];
                curr = device_edge_block[0];
                curr->active_edge_count = 0;
                // vertex_adjacency->edge_block_address[edge_block_counter] = curr;
                vertex_adjacency->edge_block_address = curr;
                vertex_adjacency->last_insert_edge_block = curr;

            }

            else {

                edge_block_entry_count = vertex_adjacency->last_insert_edge_offset;
                // edge_block_counter = vertex_adjacency->edge_block_count;
                edge_block_counter = 0;
                curr = vertex_adjacency->last_insert_edge_block;
                // curr->active_edge_count = 0;

                if(space_remaining == 0) {
                    curr = device_edge_block[0];
                    curr->active_edge_count = 0;
                    edge_block_entry_count = 0;
                    edge_block_counter = 1;
                }
                // else {
                //     curr = vertex_adjacency->last_insert_edge_block;
                // }
            }

            // curr = vertex_adjacency->next;

            // test code v3 start

            // unsigned long start_index = start_index_batch;
            // unsigned long end_index = end_index_batch;

            // if(start_index_batch <= d_csr_offset[id])
            //     start_index = d_csr_offset[id];
            // // else
            // //     end_index = end_index_batch;

            // // test code v3 end

            // if(end_index_batch >= d_csr_offset[id + 1])
            //     end_index = d_csr_offset[id + 1];
            // else
            //     end_index = end_index_batch;

            // test code v3 end

            // unsigned long start_index;

            // if(current_vertex != 1)
            //     start_index = d_prefix_sum_vertex_degrees[current_vertex - 2];
            // else
            //     start_index = 0;

            // unsigned long end_index = d_prefix_sum_vertex_degrees[current_vertex - 1];

            // printf("Current vertex = %lu, start = %lu, end = %lu\n", current_vertex, start_index, end_index);

            // unsigned long current_edge_block = 0;
            // curr = device_edge_block[edge_block_counter];
            // // vertex_adjacency->edge_block_address[edge_block_counter] = curr;
            // vertex_adjacency->edge_block_address = curr;
            // vertex_adjacency->last_insert_edge_block = curr;

            // __syncthreads();

            // if(id == 0)
            //     printf("Checkpoint AL beg\n");

            // unsigned long edge_counter = 0;

            // if(id == 0) {
            //     // printf("Edge counter is %lu\n", vertex_adjacency->active_edge_count);
            //     printf("Start index is %lu and end index is %lu\n", start_index, end_index);
            //     printf("Device edge block address is %p\n", curr);
            // }

            // printf("Checkpoint\n");

            for(unsigned long i = start_index ; i < end_index ; i++) {

                // printf("Checkpoint 0\n");

                // if(d_source[i] == current_vertex){

                // printf("Thread = %lu and Source = %lu and Destination = %lu\n", id, id + 1, d_csr_edges[i]);
                // printf("Checkpoint 1\n");

                // insert here

                // if(curr == NULL)
                //     printf("Hit here at %lu\n", id);
                // curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_destination[i];
                curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_csr_edges[i];
                // printf("Checkpoint 2\n");

                // if(id == 0) {
                // }
                // printf("Entry is %lu\n", curr->edge_block_entry[edge_block_entry_count].destination_vertex);
                vertex_adjacency->active_edge_count++;
                curr->active_edge_count++;
                vertex_adjacency->last_insert_edge_offset++;
                // printf("Checkpoint 3\n");

                // printf("Entry is %lu for thread #%lu at %p, counter is %lu and %lu\n", curr->edge_block_entry[edge_block_entry_count].destination_vertex, id, curr, curr->active_edge_count, vertex_adjacency->edge_block_address->active_edge_count);


                if((i + 1 < end_index) && (++edge_block_entry_count >= EDGE_BLOCK_SIZE) && (edge_block_counter < edge_blocks_required)) {

                    // curr = curr->next;
                    // edge_block_counter++;
                    edge_block_entry_count = 0;


                    if((space_remaining != 0) && (edge_block_counter == 0))
                        curr = device_edge_block[0];
                    else
                        curr = curr + 1;

                    // printf("Hit for thread #%lu at %p\n", id, curr);

                    // curr = device_edge_block[++edge_block_counter];
                    ++edge_block_counter;
                    curr->active_edge_count = 0;
                    vertex_adjacency->last_insert_edge_block = curr;
                    // vertex_adjacency->edge_block_count++;
                    vertex_adjacency->last_insert_edge_offset = 0;
                    // vertex_adjacency->edge_block_address[edge_block_counter] = curr;

                }

                // }

            }

            // printf("Debug code start\n");
            // struct edge_block *curr = device_edge_block[0];

            // curr->rptr = NULL;
            // curr->lptr->lptr = NULL;
            // curr->lptr->rptr = NULL;

            // printf("Root %p contents are %lu and %lu, pointers are %p and %p, active_edge_count is %lu\n", curr, curr->edge_block_entry[0].destination_vertex, curr->edge_block_entry[1].destination_vertex, curr->lptr, curr->rptr, curr->active_edge_count);
            // curr = curr->lptr;
            // printf("Lptr %p contents are %lu and %lu, pointers are %p and %p, active_edge_count is %lu\n", curr, curr->edge_block_entry[0].destination_vertex, curr->edge_block_entry[1].destination_vertex, curr->lptr, curr->rptr, curr->active_edge_count);


            // inorderTraversalTemp(device_edge_block[0]);

            // printf("Success\n");

        }
        // printf("Checkpoint final thread#%lu\n", id);

    }

    // __syncthreads();

    // if(id == 0)

}

__global__ void adjacency_list_init_modded_v5(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, unsigned long total_edge_blocks_count_batch, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks, struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long batch_number, unsigned long batch_size, unsigned long start_index_batch, unsigned long end_index_batch, unsigned long* d_csr_offset, unsigned long* d_csr_edges, unsigned long *d_source_degrees) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int current;

    __syncthreads();


    if(id < vertex_size) {

        // printf("ID is %lu, edge_block_count now is %lu\n", id, device_vertex_dictionary->edge_block_count[id]);


        unsigned long current_vertex = id + 1;

        unsigned long edge_blocks_required = 0;

        // printf("Checkpoint for id %lu\n", id);

        // critical section start

        struct edge_block *device_edge_block_base;

        // v4 test code start

        // struct adjacency_sentinel_new *vertex_adjacency = device_vertex_dictionary->vertex_adjacency[id];
        struct edge_block *curr;

        unsigned long space_remaining = 0;
        unsigned long edge_blocks_used_present = 0;

        if((device_vertex_dictionary->edge_block_address[id] != NULL) && (device_vertex_dictionary->last_insert_edge_offset[id] != 0)) {

            space_remaining = EDGE_BLOCK_SIZE - device_vertex_dictionary->last_insert_edge_offset[id];
            // printf("id=%lu, last_insert_edge_offset is %lu\n", id, device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset);
        }

        edge_blocks_used_present = device_vertex_dictionary->edge_block_count[id];

        // printf("ID is %lu, edge_block_count now is %lu, edge_blocks_used is %lu\n", id, device_vertex_dictionary->edge_block_count[id], edge_blocks_used_present);


        // printf("%lu\n", edge_blocks_required);
        unsigned long start_index = start_index_batch;
        unsigned long end_index = end_index_batch;

        unsigned long new_edges_count = 0;

        // if(start_index_batch <= d_csr_offset[id])
        //     start_index = d_csr_offset[id];

        // if(end_index_batch >= d_csr_offset[id + 1])
        //     end_index = d_csr_offset[id + 1];

        start_index = d_csr_offset[id];
        end_index = d_csr_offset[id + 1];

        // below check since end_index and start_index are unsigned longs
        if(end_index > start_index) {

            // new_edges_count = end_index - start_index;
            new_edges_count = d_source_degrees[id];
            // edge_blocks_required = ceil(double((end_index - start_index) - space_remaining) / EDGE_BLOCK_SIZE);
            if(new_edges_count > space_remaining)
                edge_blocks_required = ceil(double(new_edges_count - space_remaining) / EDGE_BLOCK_SIZE);

        }

        else
            edge_blocks_required = 0;

        // struct edge_block *device_edge_block[PREALLOCATE_EDGE_BLOCKS];
        struct edge_block *device_edge_block[1];

        if(edge_blocks_required > 0) {

            parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_batch, d_prefix_sum_edge_blocks, id, thread_blocks, edge_blocks_used_present, edge_blocks_required);
            device_vertex_dictionary->edge_block_count[id] += edge_blocks_required;
            device_edge_block_base = device_edge_block[0];

            // printf("ID is %lu, edge_block_count now is %lu, edge_blocks_required is %lu\n", id, device_vertex_dictionary->edge_block_count[id], edge_blocks_required);
            // printf("Checkpoint ID is %lu, device edge block base is %p\n", id, device_edge_block_base);

        }



        // printf("thread #%lu, start_index is %lu, end_index is %lu, end_index_batch is %lu, d_csr_offset[id+1] is %lu, edge_blocks_required is %lu, edge_blocks_used is %lu, space_remaining is %lu\n", id, start_index, end_index, end_index_batch, d_csr_offset[id + 1], edge_blocks_required, edge_blocks_used_present, space_remaining);

        // create complete binary tree for first time
        if((edge_blocks_required > 0) && (device_vertex_dictionary->edge_block_address[id] == NULL)) {

            curr = device_edge_block[0];

            unsigned long i = 0;
            unsigned long curr_index = 0;

            if(edge_blocks_required > 1) {

                for(i = 0 ; i < (edge_blocks_required / 2) - 1 ; i++) {

                    // curr->value = array[i];
                    curr->lptr = *device_edge_block + (2 * i) + 1;
                    curr->rptr = *device_edge_block + (2 * i) + 2;

                    // printf("Inserted internal node %p\n", curr);

                    // if(i) {

                    curr->level_order_predecessor = *device_edge_block + i - 1;

                    // }

                    curr_index++;
                    curr = *device_edge_block + curr_index;

                }

                if(edge_blocks_required % 2) {

                    // curr->value = array [i];
                    curr->lptr = *device_edge_block + (2 * i) + 1;
                    curr->rptr = *device_edge_block + (2 * i++) + 2;

                    curr->level_order_predecessor = *device_edge_block + i - 1;


                    // printf("Inserted internal node v1 %p for id %lu\n", curr, id);

                }
                else {
                    // curr->value = array [i];
                    curr->lptr = *device_edge_block + (2 * i++) + 1;

                    curr->level_order_predecessor = *device_edge_block + i - 1;


                    // printf("Inserted internal node v2 %p for id %lu, lptr is %p\n", curr, id, *device_edge_block + (2 * (i - 1)) + 1);

                }

                curr_index++;
                curr = *device_edge_block + curr_index;

            }

            // printf("Checkpoint %lu\n", edge_blocks_required);

            for( ; i < edge_blocks_required ; i++) {

                // curr->value = array[i];
                curr->lptr = NULL;
                curr->rptr = NULL;

                curr->level_order_predecessor = *device_edge_block + i - 1;


                // printf("Inserted leaf node %p\n", curr);

                curr_index++;
                curr = *device_edge_block + curr_index;

            }

        }

        else if((edge_blocks_required > 0) && (device_vertex_dictionary->edge_block_address[id] != NULL)) {

            // printf("Checkpoint update %lu\n", edge_blocks_required);

            for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

                unsigned long current_edge_block = device_vertex_dictionary->edge_block_count[id] - edge_blocks_required + i + 1;
                unsigned long length = 0;
                // unsigned long bit_string = traversal_string(current_edge_block, &length);
                unsigned long bit_string = bit_string_lookup[current_edge_block - 1];

                // printf("Disputed address %p for id %lu\n", *device_edge_block + i, id);

                // if(batch_number)
                //     goto exit_insert;

                // printf("Checkpoint cbt1 for thread #%lu, bit_string for %lu is %lu with length %lu\n", id, current_edge_block, bit_string, length);
                insert_edge_block_to_CBT(device_vertex_dictionary->edge_block_address[id], bit_string, length, *device_edge_block + i, device_vertex_dictionary->last_insert_edge_block[id], i);
                // printf("Checkpoint cbt2\n");

            }

        }

        // printf("Checkpoint for id %lu\n", id);
        // if(batch_number)
        //     goto exit_insert;


        if(new_edges_count > 0) {

            unsigned long edge_block_entry_count;
            unsigned long edge_block_counter;

            if(device_vertex_dictionary->edge_block_address[id] == NULL) {

                edge_block_entry_count = 0;
                edge_block_counter = 0;
                device_vertex_dictionary->active_edge_count[id] = 0;
                // vertex_adjacency->edge_block_count = 1;

                // curr = device_edge_block[edge_block_counter];
                curr = device_edge_block[0];
                curr->active_edge_count = 0;
                // vertex_adjacency->edge_block_address[edge_block_counter] = curr;
                device_vertex_dictionary->edge_block_address[id] = curr;
                device_vertex_dictionary->last_insert_edge_block[id] = curr;

            }

            else {

                edge_block_entry_count = device_vertex_dictionary->last_insert_edge_offset[id];
                // edge_block_counter = vertex_adjacency->edge_block_count;
                edge_block_counter = 0;
                curr = device_vertex_dictionary->last_insert_edge_block[id];
                // curr->active_edge_count = 0;

                if(space_remaining == 0) {
                    curr = device_edge_block[0];
                    curr->active_edge_count = 0;
                    device_vertex_dictionary->last_insert_edge_block[id] = curr;
                    edge_block_entry_count = 0;
                    edge_block_counter = 1;
                }
                // else {
                //     curr = vertex_adjacency->last_insert_edge_block;
                // }
            }



            // unsigned long previous_edge = 0;

            // printf("ID is %lu, start_index=%lu, end_index=%lu\n", id, start_index, end_index);

            for(unsigned long i = start_index ; i < end_index ; i++) {

                // if((curr == NULL) || (vertex_adjacency == NULL) || (edge_block_entry_count >= EDGE_BLOCK_SIZE) || (i >= batch_size))
                //     printf("Hit disupte null\n");

                // if(edge_block_entry_count >= EDGE_BLOCK_SIZE)
                //     printf("Hit dispute margin\n");
                // if(d_csr_edges[i] != previous_edge)

                if(d_csr_edges[i] == INFTY)
                    continue;

                curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_csr_edges[i];

                // previous_edge = d_csr_edges[i];

                // if(d_csr_edges[i] == 0)
                //     printf("Hit dispute\n");



                device_vertex_dictionary->active_edge_count[id]++;
                curr->active_edge_count++;
                device_vertex_dictionary->last_insert_edge_offset[id]++;
                // printf("Checkpoint 3\n");

                // printf("Entry is %lu for thread #%lu at %p, counter is %lu and %lu\n", curr->edge_block_entry[edge_block_entry_count].destination_vertex, id, curr, curr->active_edge_count, vertex_adjacency->edge_block_address->active_edge_count);

                // edge_block_entry_count++;


                // if(edge_block_entry_count >= EDGE_BLOCK_SIZE)
                //     edge_block_entry_count = 0;

                // continue;
                // if(id == 3) {

                //     printf("iteration %lu, edge_block_edge_count is %lu, curr is %p\n", i, curr->active_edge_count, curr);

                // }

                // if(curr == device_vertex_dictionary->edge_block_address[3])
                //     printf("Hit at id %lu\n", id);

                // if((i + 1 < end_index) && (++edge_block_entry_count >= EDGE_BLOCK_SIZE) && (edge_block_counter < edge_blocks_required)) {
                if((++edge_block_entry_count >= EDGE_BLOCK_SIZE) && (i + 1 < end_index) && (edge_block_counter < edge_blocks_required)) {



                    // curr = curr->next;
                    // edge_block_counter++;
                    edge_block_entry_count = 0;


                    // printf("at dispute\n");
                    // if((curr == NULL) || (device_edge_block[0] == NULL) || (vertex_adjacency == NULL))
                    //     printf("Dispute caught\n");

                    if((space_remaining != 0) && (edge_block_counter == 0))
                        curr = device_edge_block[0];
                    else
                        curr = curr + 1;



                    // printf("Hit for thread #%lu at %p\n", id, curr);

                    // curr = device_edge_block[++edge_block_counter];
                    ++edge_block_counter;

                    if(edge_block_counter >= edge_blocks_required)
                        continue;

                    curr->active_edge_count = 0;
                    // printf("%p active edge count made 0 by ID %lu, edge_block_counter is %lu and edge_blocks_required is %lu\n", curr, id, edge_block_counter, edge_blocks_required);
                    device_vertex_dictionary->last_insert_edge_block[id] = curr;
                    // vertex_adjacency->edge_block_count++;
                    device_vertex_dictionary->last_insert_edge_offset[id] = 0;
                    // vertex_adjacency->edge_block_address[edge_block_counter] = curr;



                }

                // }

            }


            if(device_vertex_dictionary->last_insert_edge_offset[id] == EDGE_BLOCK_SIZE)
                device_vertex_dictionary->last_insert_edge_offset[id] = 0;

        }
        // printf("Checkpoint final thread#%lu\n", id);

    }

    exit_insert:

    // __syncthreads();

    // if(id == 0) {
    //     printf("Checkpoint final\n");
    // }

}

__global__ void adjacency_list_kernel_VC_LB(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, unsigned long total_edge_blocks_count_batch, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks, struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long batch_number, unsigned long batch_size, unsigned long start_index_batch, unsigned long end_index_batch, unsigned long* d_csr_offset, unsigned long* d_csr_edges, unsigned long *d_source_degrees, unsigned long load_factor) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // unsigned int current;

    // __syncthreads();


    if(id < ceil((double)vertex_size / load_factor)) {

        // printf("ID is %lu, edge_block_count now is %lu\n", id, device_vertex_dictionary->edge_block_count[id]);
        unsigned long vertex_start = id * load_factor;
        unsigned long vertex_end = (id + 1) * load_factor;

        if(vertex_end > vertex_size)
            vertex_end = vertex_size;

        for(unsigned long current_vertex = vertex_start ; current_vertex < vertex_end ; current_vertex++) {

            // unsigned long current_vertex = id + 1;

            unsigned long edge_blocks_required = 0;

            // printf("Checkpoint for id %lu\n", id);

            // critical section start

            struct edge_block *device_edge_block_base;

            // v4 test code start

            // struct adjacency_sentinel_new *vertex_adjacency = device_vertex_dictionary->vertex_adjacency[id];
            struct edge_block *curr;

            unsigned long space_remaining = 0;
            unsigned long edge_blocks_used_present = 0;

            if((device_vertex_dictionary->edge_block_address[current_vertex] != NULL) && (device_vertex_dictionary->last_insert_edge_offset[current_vertex] != 0)) {

                space_remaining = EDGE_BLOCK_SIZE - device_vertex_dictionary->last_insert_edge_offset[current_vertex];
                // printf("id=%lu, last_insert_edge_offset is %lu\n", id, device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset);
            }

            edge_blocks_used_present = device_vertex_dictionary->edge_block_count[current_vertex];

            // printf("current_vertex is %lu, edge_block_count now is %lu, edge_blocks_used is %lu\n", current_vertex, device_vertex_dictionary->edge_block_count[current_vertex], edge_blocks_used_present);


            // printf("%lu\n", edge_blocks_required);
            unsigned long start_index = start_index_batch;
            unsigned long end_index = end_index_batch;

            unsigned long new_edges_count = 0;

            // if(start_index_batch <= d_csr_offset[current_vertex])
            //     start_index = d_csr_offset[current_vertex];

            // if(end_index_batch >= d_csr_offset[current_vertex + 1])
            //     end_index = d_csr_offset[current_vertex + 1];

            start_index = d_csr_offset[current_vertex];
            end_index = d_csr_offset[current_vertex + 1];

            // below check since end_index and start_index are unsigned longs
            if(end_index > start_index) {

                // new_edges_count = end_index - start_index;
                new_edges_count = d_source_degrees[current_vertex];
                // edge_blocks_required = ceil(double((end_index - start_index) - space_remaining) / EDGE_BLOCK_SIZE);
                if(new_edges_count > space_remaining)
                    edge_blocks_required = ceil(double(new_edges_count - space_remaining) / EDGE_BLOCK_SIZE);

            }

            else
                edge_blocks_required = 0;

            // struct edge_block *device_edge_block[PREALLOCATE_EDGE_BLOCKS];
            struct edge_block *device_edge_block[1];

            if(edge_blocks_required > 0) {

                parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_batch, d_prefix_sum_edge_blocks, current_vertex, thread_blocks, edge_blocks_used_present, edge_blocks_required);
                device_vertex_dictionary->edge_block_count[current_vertex] += edge_blocks_required;
                device_edge_block_base = device_edge_block[0];

                // printf("current_vertex is %lu, edge_block_count now is %lu, edge_blocks_required is %lu\n", current_vertex, device_vertex_dictionary->edge_block_count[current_vertex], edge_blocks_required);
                // printf("Checkpoint current_vertex is %lu, device edge block base is %p\n", current_vertex, device_edge_block_base);

            }



            // printf("thread #%lu, start_index is %lu, end_index is %lu, end_index_batch is %lu, d_csr_offset[current_vertex+1] is %lu, edge_blocks_required is %lu, edge_blocks_used is %lu, space_remaining is %lu\n", current_vertex, start_index, end_index, end_index_batch, d_csr_offset[current_vertex + 1], edge_blocks_required, edge_blocks_used_present, space_remaining);

            // create complete binary tree for first time
            if((edge_blocks_required > 0) && (device_vertex_dictionary->edge_block_address[current_vertex] == NULL)) {

                curr = device_edge_block[0];

                unsigned long i = 0;
                unsigned long curr_index = 0;

                if(edge_blocks_required > 1) {

                    for(i = 0 ; i < (edge_blocks_required / 2) - 1 ; i++) {

                        // curr->value = array[i];
                        curr->lptr = *device_edge_block + (2 * i) + 1;
                        curr->rptr = *device_edge_block + (2 * i) + 2;

                        // printf("Inserted internal node %p\n", curr);

                        // if(i) {

                        curr->level_order_predecessor = *device_edge_block + i - 1;

                        // }

                        curr_index++;
                        curr = *device_edge_block + curr_index;

                    }

                    if(edge_blocks_required % 2) {

                        // curr->value = array [i];
                        curr->lptr = *device_edge_block + (2 * i) + 1;
                        curr->rptr = *device_edge_block + (2 * i++) + 2;

                        curr->level_order_predecessor = *device_edge_block + i - 1;


                        // printf("Inserted internal node v1 %p for current_vertex %lu\n", curr, current_vertex);

                    }
                    else {
                        // curr->value = array [i];
                        curr->lptr = *device_edge_block + (2 * i++) + 1;

                        curr->level_order_predecessor = *device_edge_block + i - 1;


                        // printf("Inserted internal node v2 %p for current_vertex %lu, lptr is %p\n", curr, current_vertex, *device_edge_block + (2 * (i - 1)) + 1);

                    }

                    curr_index++;
                    curr = *device_edge_block + curr_index;

                }

                // printf("Checkpoint %lu\n", edge_blocks_required);

                for( ; i < edge_blocks_required ; i++) {

                    // curr->value = array[i];
                    curr->lptr = NULL;
                    curr->rptr = NULL;

                    curr->level_order_predecessor = *device_edge_block + i - 1;


                    // printf("Inserted leaf node %p\n", curr);

                    curr_index++;
                    curr = *device_edge_block + curr_index;

                }

            }

            else if((edge_blocks_required > 0) && (device_vertex_dictionary->edge_block_address[current_vertex] != NULL)) {

                // printf("Checkpoint update %lu\n", edge_blocks_required);

                for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

                    unsigned long current_edge_block = device_vertex_dictionary->edge_block_count[current_vertex] - edge_blocks_required + i + 1;
                    unsigned long length = 0;
                    // unsigned long bit_string = traversal_string(current_edge_block, &length);
                    unsigned long bit_string = bit_string_lookup[current_edge_block - 1];

                    // printf("Disputed address %p for id %lu\n", *device_edge_block + i, id);

                    // if(batch_number)
                    //     goto exit_insert;

                    // printf("Checkpoint cbt1 for thread #%lu, bit_string for %lu is %lu with length %lu\n", id, current_edge_block, bit_string, length);
                    insert_edge_block_to_CBT(device_vertex_dictionary->edge_block_address[current_vertex], bit_string, length, *device_edge_block + i, device_vertex_dictionary->last_insert_edge_block[current_vertex], i);
                    // printf("Checkpoint cbt2\n");

                }

            }

            // printf("Checkpoint for id %lu\n", id);
            // if(batch_number)
            //     goto exit_insert;


            if(new_edges_count > 0) {

                unsigned long edge_block_entry_count;
                unsigned long edge_block_counter;

                if(device_vertex_dictionary->edge_block_address[current_vertex] == NULL) {

                    edge_block_entry_count = 0;
                    edge_block_counter = 0;
                    device_vertex_dictionary->active_edge_count[current_vertex] = 0;
                    // vertex_adjacency->edge_block_count = 1;

                    // curr = device_edge_block[edge_block_counter];
                    curr = device_edge_block[0];
                    curr->active_edge_count = 0;
                    // vertex_adjacency->edge_block_address[edge_block_counter] = curr;
                    device_vertex_dictionary->edge_block_address[current_vertex] = curr;
                    device_vertex_dictionary->last_insert_edge_block[current_vertex] = curr;

                }

                else {

                    edge_block_entry_count = device_vertex_dictionary->last_insert_edge_offset[current_vertex];
                    // edge_block_counter = vertex_adjacency->edge_block_count;
                    edge_block_counter = 0;
                    curr = device_vertex_dictionary->last_insert_edge_block[current_vertex];
                    // curr->active_edge_count = 0;

                    if(space_remaining == 0) {
                        curr = device_edge_block[0];
                        curr->active_edge_count = 0;
                        device_vertex_dictionary->last_insert_edge_block[current_vertex] = curr;
                        edge_block_entry_count = 0;
                        edge_block_counter = 1;
                    }
                    // else {
                    //     curr = vertex_adjacency->last_insert_edge_block;
                    // }
                }



                // unsigned long previous_edge = 0;

                // printf("ID is %lu, start_index=%lu, end_index=%lu\n", id, start_index, end_index);

                for(unsigned long i = start_index ; i < end_index ; i++) {

                    // if((curr == NULL) || (vertex_adjacency == NULL) || (edge_block_entry_count >= EDGE_BLOCK_SIZE) || (i >= batch_size))
                    //     printf("Hit disupte null\n");

                    // if(edge_block_entry_count >= EDGE_BLOCK_SIZE)
                    //     printf("Hit dispute margin\n");
                    // if(d_csr_edges[i] != previous_edge)

                    if(d_csr_edges[i] == INFTY)
                        continue;

                    curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_csr_edges[i];

                    // previous_edge = d_csr_edges[i];

                    // if(d_csr_edges[i] == 0)
                    //     printf("Hit dispute\n");



                    device_vertex_dictionary->active_edge_count[current_vertex]++;
                    curr->active_edge_count++;
                    device_vertex_dictionary->last_insert_edge_offset[current_vertex]++;
                    // printf("Checkpoint 3\n");

                    // printf("Entry is %lu for thread #%lu at %p, counter is %lu and %lu\n", curr->edge_block_entry[edge_block_entry_count].destination_vertex, id, curr, curr->active_edge_count, vertex_adjacency->edge_block_address->active_edge_count);

                    // edge_block_entry_count++;


                    // if(edge_block_entry_count >= EDGE_BLOCK_SIZE)
                    //     edge_block_entry_count = 0;

                    // continue;
                    // if(id == 3) {

                    //     printf("iteration %lu, edge_block_edge_count is %lu, curr is %p\n", i, curr->active_edge_count, curr);

                    // }

                    // if(curr == device_vertex_dictionary->edge_block_address[3])
                    //     printf("Hit at id %lu\n", id);

                    // if((i + 1 < end_index) && (++edge_block_entry_count >= EDGE_BLOCK_SIZE) && (edge_block_counter < edge_blocks_required)) {
                    if((++edge_block_entry_count >= EDGE_BLOCK_SIZE) && (i + 1 < end_index) && (edge_block_counter < edge_blocks_required)) {



                        // curr = curr->next;
                        // edge_block_counter++;
                        edge_block_entry_count = 0;


                        // printf("at dispute\n");
                        // if((curr == NULL) || (device_edge_block[0] == NULL) || (vertex_adjacency == NULL))
                        //     printf("Dispute caught\n");

                        if((space_remaining != 0) && (edge_block_counter == 0))
                            curr = device_edge_block[0];
                        else
                            curr = curr + 1;



                        // printf("Hit for thread #%lu at %p\n", id, curr);

                        // curr = device_edge_block[++edge_block_counter];
                        ++edge_block_counter;

                        if(edge_block_counter >= edge_blocks_required)
                            continue;

                        curr->active_edge_count = 0;
                        // printf("%p active edge count made 0 by ID %lu, edge_block_counter is %lu and edge_blocks_required is %lu\n", curr, id, edge_block_counter, edge_blocks_required);
                        device_vertex_dictionary->last_insert_edge_block[current_vertex] = curr;
                        // vertex_adjacency->edge_block_count++;
                        device_vertex_dictionary->last_insert_edge_offset[current_vertex] = 0;
                        // vertex_adjacency->edge_block_address[edge_block_counter] = curr;



                    }

                    // }

                }


                if(device_vertex_dictionary->last_insert_edge_offset[current_vertex] == EDGE_BLOCK_SIZE)
                    device_vertex_dictionary->last_insert_edge_offset[current_vertex] = 0;

            }
            // printf("Checkpoint final thread#%lu\n", id);
        }

    }

    exit_insert:

    // __syncthreads();

    // if(id == 0) {
    //     printf("Checkpoint final\n");
    // }

}

// __device__ unsigned long device_binary_search_find(unsigned long *input_array, unsigned long key, unsigned long size) {

//     unsigned long start = 0;
//     unsigned long end = size;
//     unsigned long mid;

//     while (start <= end) {

//         mid = (start + end) / 2;

//         // Check if x is present at mid
//         if ((input_array[mid] == key))
//             return 1;

//         // If x is smaller, ignore right half
//         if ((input_array[mid] > key))
//             end = mid - 1;
//         // If x greater, ignore left half
//         else
//             start = mid + 1;


//     }

//     // If we reach here, then element was not present
//     return 0;

// }

__device__ unsigned long device_binary_search_find(unsigned long *input_array, unsigned long key, unsigned long size) {

    long start = 0;
    long end = size;
    long mid;

    while (start <= end) {

        mid = (start + end) / 2;
        // mid = start + (end - start) / 2;
        unsigned long item = input_array[mid];

        // Check if x is present at mid
        if (item == key)
            return 1;

        // If x greater, ignore left half
        if (item < key)
            start = mid + 1;

            // If x is smaller, ignore right half
        else
            end = mid - 1;
    }

    // If we reach here, then element was not present
    return 0;

}

__device__ unsigned long device_binary_search(unsigned long *input_array, unsigned long key, unsigned long size) {

    // unsigned long start = 0;
    // unsigned long end = size;
    // unsigned long mid;

    long start = 0;
    long end = size;
    long mid;

    while (start <= end) {

        mid = (start + end) / 2;

        // Check if x is present at mid
        if ((input_array[mid] == key) && ((mid + 1) <= end) && (input_array[mid + 1] > key))
            return mid;

        // If x is smaller, ignore right half
        if (input_array[mid] > key)
            end = mid - 1;
            // If x greater, ignore left half
        else
            start = mid + 1;


    }

    // If we reach here, then element was not present
    return start - 1;

}

__global__ void batched_edge_inserts_EC(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, unsigned long total_edge_blocks_count_batch, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks, struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long batch_number, unsigned long batch_size, unsigned long start_index_batch, unsigned long end_index_batch, unsigned long* d_csr_offset, unsigned long* d_csr_edges, unsigned long *d_source_degrees) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < batch_size) {

        // printf("ID=\n");

        unsigned long target_vertex = d_csr_edges[id];

        if(target_vertex != INFTY) {

            unsigned long source_vertex = device_binary_search(d_csr_offset, id, vertex_size);
            unsigned long active_edge_block_count = device_vertex_dictionary->edge_block_count[source_vertex];
            unsigned long new_edge_block_count = d_prefix_sum_edge_blocks[source_vertex + 1] - d_prefix_sum_edge_blocks[source_vertex];
            // unsigned long index_counter = id -  d_csr_offset[source_vertex];
            struct edge_block *root = NULL;
            struct edge_block *base = NULL;

            // if(source_vertex == 36871)
            //     printf("ID=%lu, source=%lu, target=%lu, new_edge_block_count=%lu\n", id, source_vertex, target_vertex, new_edge_block_count);

            // below if is taken if CBT is constructed for the first time for an adjacency
            if((device_vertex_dictionary->edge_block_address[source_vertex] == NULL) || (batch_number == 0)) {

                if(new_edge_block_count > 0) {

                    unsigned long index_counter = id -  d_csr_offset[source_vertex];
                    unsigned long current_edge_block_counter = (index_counter / EDGE_BLOCK_SIZE);
                    base = pop_edge_block_address(total_edge_blocks_count_batch, d_prefix_sum_edge_blocks, source_vertex);
                    root = base + current_edge_block_counter;
                    // struct edge_block *device_edge_block[1];
                    // parallel_pop_from_edge_preallocate_queue(device_edge_block, total_edge_blocks_count_batch, d_prefix_sum_edge_blocks, source_vertex, thread_blocks, 0, new_edge_block_count);
                    // base = device_edge_block[0];
                    // root = base + current_edge_block_counter;
                    // device_vertex_dictionary->edge_block_count[id] += new_edge_block_count;
                    // device_edge_block_base = device_edge_block[0];


                    // printf("ID is %lu, source is %lu, index_counter is %lu, edge_block_counter is %lu, edge_blocks_required is %lu, root is %p\n", id, source_vertex, index_counter, current_edge_block_counter, new_edge_block_count, root);
                    // printf("Checkpoint ID is %lu, device edge block base is %p\n", id, device_edge_block_base);

                    // this if is taken for the thread inserting the first entry in the edge block
                    // that particular thread is responsible for adding the edge block to the CBT
                    if(!(index_counter % EDGE_BLOCK_SIZE)) {

                        // printf("ID is %lu, source is %lu, index_counter is %lu, edge_block_counter is %lu, edge_blocks_required is %lu, root is %p\n", id, source_vertex, index_counter, current_edge_block_counter, new_edge_block_count, root);

                        // pop the current edge block from the edge queue

                        // if((index_counter * 2) < (active_edge_block_count + new_edge_block_count)) {
                        //     // now fix the lptr;
                        //     ;
                        //     if(((index_counter * 2) + 1) < (active_edge_block_count + new_edge_block_count))
                        //     // now fix the rptr;
                        //     ;
                        // }


                        if (new_edge_block_count > ((current_edge_block_counter * 2) + 2)) {

                            // two children, fix the lptr and rptr
                            root->lptr = base + (current_edge_block_counter * 2) + 1;
                            root->rptr = base + (current_edge_block_counter * 2) + 2;

                            // if(source_vertex == 36871)
                            //     printf("Hit first if, id=%lu\n", id);
                            // if(source_vertex == 38)
                            //     printf("ID=%lu, source=%lu, root=%p, root->lptr=%p, root->rptr=%p\n", id, source_vertex, root, root->lptr, root->rptr);

                        }
                        else if (new_edge_block_count > ((current_edge_block_counter * 2) + 1)) {

                            // one child, fix only lptr
                            root->lptr = base + (current_edge_block_counter * 2) + 1;
                            root->rptr = NULL;

                            // if(source_vertex == 36871)
                            //     printf("Hit second if, id=%lu, base=%p, lptr=%p, new_edge_block_count=%lu, counter=%lu, root=%p, lptr=%p, rptr=%p\n", id, base,  base + (current_edge_block_counter * 2) + 1, new_edge_block_count ,((current_edge_block_counter * 2) + 1), root, root->lptr, root->rptr);

                        }
                        else {

                            // leaf node, hence ptrs point to NULL
                            root->lptr = NULL;
                            root->rptr = NULL;

                            // if(source_vertex == 36871)
                            //     printf("Hit third if, id=%lu, base=%p, new_edge_block_count=%lu, counter=%lu, root=%p, lptr=%p, rptr=%p\n", id, base, new_edge_block_count ,((current_edge_block_counter * 2) + 1), root, root->lptr, root->rptr);

                        }



                        if(current_edge_block_counter) {
                            root->level_order_predecessor = root - 1;
                            // if(source_vertex == 36871)
                            //     printf("child guy, level_order_predecessor is %p\n", root->level_order_predecessor);
                        }
                        else if(!index_counter) {
                            // printf("ID is %lu, source is %lu, index_counter is %lu, edge_block_counter is %lu, edge_blocks_required is %lu, root is %p, edge_count is %lu\n", id, source_vertex, index_counter, current_edge_block_counter, new_edge_block_count, root, d_source_degrees[source_vertex]);
                            if(!batch_number)
                                device_vertex_dictionary->active_edge_count[source_vertex] = d_source_degrees[source_vertex];


                            device_vertex_dictionary->edge_block_address[source_vertex] = base;
                            device_vertex_dictionary->vertex_id[source_vertex] = source_vertex + 1;
                            device_vertex_dictionary->edge_block_count[source_vertex] = new_edge_block_count;
                            // device_vertex_dictionary->edge_block_count[source_vertex] = ceil(double(d_source_degrees[source_vertex]) / EDGE_BLOCK_SIZE);

                            device_vertex_dictionary->last_insert_edge_block[source_vertex] = base + new_edge_block_count - 1;
                            // device_vertex_dictionary->last_insert_edge_block[source_vertex] = base + device_vertex_dictionary->edge_block_count[source_vertex] - 1;
                            device_vertex_dictionary->last_insert_edge_offset[source_vertex] = d_source_degrees[source_vertex] % EDGE_BLOCK_SIZE;


                        }
                        // fix the lptr and rptr


                    }



                    unsigned long edge_entry_index = index_counter % EDGE_BLOCK_SIZE;
                    // return;

                    root->edge_block_entry[edge_entry_index].destination_vertex = target_vertex;
                    return;
                }

            }

                // below else is taken if it's the subsequent batch insert to an adjacency
            else if((device_vertex_dictionary->edge_block_address[source_vertex] != NULL) && (batch_number)) {
                // else if(0) {
                // printf("Else taken terror, source_vertex=%lu, root=%p, edge_count=%lu\n", source_vertex, device_vertex_dictionary->edge_block_address[source_vertex], device_vertex_dictionary->active_edge_count[source_vertex]);

                unsigned long last_insert_edge_offset = device_vertex_dictionary->last_insert_edge_offset[source_vertex];
                unsigned long space_remaining = 0;
                if(last_insert_edge_offset != 0)
                    space_remaining = EDGE_BLOCK_SIZE - last_insert_edge_offset;
                unsigned long index_counter = id - d_csr_offset[source_vertex];

                // if((id > 4990))
                // // if(space_remaining == 0)
                //     printf("Outside, ID=%lu, source=%lu, target=%lu, index_counter=%lu, space_remaining=%lu, total_edge_blocks=%lu\n", id, source_vertex, target_vertex, index_counter, space_remaining, new_edge_block_count + active_edge_block_count);


                // fill up newly allocated edge_blocks
                // if(((index_counter >= space_remaining) || (!space_remaining)) && (new_edge_block_count > 0)) {
                if(((index_counter >= space_remaining)) && (new_edge_block_count > 0)) {

                    // unsigned long last_insert_edge_offset = device_vertex_dictionary->last_insert_edge_offset[source_vertex];
                    // unsigned long space_remaining = EDGE_BLOCK_SIZE - last_insert_edge_offset;
                    // unsigned long index_counter = id - d_csr_offset[source_vertex];


                    // if((index_counter < space_remaining) && (space_remaining != EDGE_BLOCK_SIZE)) {

                    //     // printf("Space Remaining: ID=%lu, source_vertex=%lu, last_insert_edge_offset=%lu, space_remaining=%lu, index_counter=%lu, target_vertex=%lu, last_insert_edge_block=%p\n", id, source_vertex, last_insert_edge_offset, space_remaining, index_counter, target_vertex, device_vertex_dictionary->last_insert_edge_block[source_vertex]);

                    //     // traverse to last insert edge block
                    //     unsigned long edge_entry_index = index_counter + last_insert_edge_offset;
                    //     device_vertex_dictionary->last_insert_edge_block[source_vertex]->edge_block_entry[edge_entry_index].destination_vertex = target_vertex;

                    //     return;
                    // }

                    // if(space_remaining != EDGE_BLOCK_SIZE)
                    index_counter -= space_remaining;

                    // current_edge_block_counter value is 0 for the first new edge block
                    unsigned long current_edge_block_counter = (index_counter / EDGE_BLOCK_SIZE);
                    base = pop_edge_block_address(total_edge_blocks_count_batch, d_prefix_sum_edge_blocks, source_vertex);
                    root = base + current_edge_block_counter;

                    // printf("Inside, ID=%lu, source=%lu, target=%lu, index_counter=%lu, base=%p, root=%p\n", id, source_vertex, target_vertex, index_counter, base, root);

                    // parallel_pop_from_edge_preallocate_queue(new_edge_block_count, total_edge_blocks_count_batch, d_prefix_sum_edge_blocks, id, thread_blocks, 0, new_edge_block_count);
                    // device_vertex_dictionary->edge_block_count[id] += new_edge_block_count;
                    // device_edge_block_base = device_edge_block[0];

                    // printf("ID is %lu, source is %lu, index_counter is %lu, edge_block_counter is %lu, edge_blocks_required is %lu, root is %p\n", id, source_vertex, index_counter, current_edge_block_counter, new_edge_block_count, root);
                    // printf("Checkpoint ID is %lu, device edge block base is %p\n", id, device_edge_block_base);

                    // this if is taken for the thread inserting the first entry in the edge block
                    // that particular thread is responsible for adding the edge block to the CBT
                    if(!(index_counter % EDGE_BLOCK_SIZE)) {

                        // printf("Regular: ID is %lu, source is %lu, index_counter is %lu, edge_block_counter is %lu, edge_blocks_required is %lu, root is %p\n", id, source_vertex, index_counter, current_edge_block_counter, new_edge_block_count, root);

                        // pop the current edge block from the edge queue

                        // if((index_counter * 2) < (active_edge_block_count + new_edge_block_count)) {
                        //     // now fix the lptr;
                        //     ;
                        //     if(((index_counter * 2) + 1) < (active_edge_block_count + new_edge_block_count))
                        //     // now fix the rptr;
                        //     ;
                        // }

                        unsigned long global_index_counter = active_edge_block_count + current_edge_block_counter;
                        unsigned long length = 0;
                        // unsigned long bit_string = bit_string_lookup[active_edge_block_count + current_edge_block_counter];
                        unsigned long bit_string = bit_string_lookup[global_index_counter];

                        // insert_edge_block_to_CBT(device_vertex_dictionary->edge_block_address[source_vertex], bit_string, length, root, device_vertex_dictionary->last_insert_edge_block[source_vertex], current_edge_block_counter);
                        insert_edge_block_to_CBT_v2(device_vertex_dictionary->edge_block_address[source_vertex], bit_string, length, root, device_vertex_dictionary->last_insert_edge_block[source_vertex], current_edge_block_counter, global_index_counter, active_edge_block_count - 1, active_edge_block_count + new_edge_block_count - 1, source_vertex, id);


                        // if (new_edge_block_count > ((current_edge_block_counter * 2) + 2)) {

                        //     // two children, fix the lptr and rptr
                        //     root->lptr = base + (current_edge_block_counter * 2) + 1;
                        //     root->rptr = base + (current_edge_block_counter * 2) + 2;

                        //     // if(source_vertex == 38)
                        //     //     printf("ID=%lu, source=%lu, root=%p, root->lptr=%p, root->rptr=%p\n", id, source_vertex, root, root->lptr, root->rptr);

                        // }
                        // else if (new_edge_block_count > ((current_edge_block_counter * 2) + 1)) {

                        //     // one child, fix only lptr
                        //     root->lptr = base + (current_edge_block_counter * 2) + 1;

                        // }
                        // else {

                        //     // leaf node, hence ptrs point to NULL
                        //     root->lptr = NULL;
                        //     root->rptr = NULL;
                        // }

                        // if(current_edge_block_counter)
                        //     root->level_order_predecessor = root - 1;
                        // else if(!index_counter) {
                        //     // printf("ID is %lu, source is %lu, index_counter is %lu, edge_block_counter is %lu, edge_blocks_required is %lu, root is %p, edge_count is %lu\n", id, source_vertex, index_counter, current_edge_block_counter, new_edge_block_count, root, d_source_degrees[source_vertex]);

                        //     device_vertex_dictionary->edge_block_address[source_vertex] = base;
                        //     device_vertex_dictionary->vertex_id[source_vertex] = source_vertex + 1;
                        //     device_vertex_dictionary->active_edge_count[source_vertex] = d_source_degrees[source_vertex];
                        //     device_vertex_dictionary->edge_block_count[source_vertex] += new_edge_block_count;

                        // }
                        // fix the lptr and rptr


                    }

                    unsigned long edge_entry_index = index_counter % EDGE_BLOCK_SIZE;
                    root->edge_block_entry[edge_entry_index].destination_vertex = target_vertex;

                }

                    // fill up remaining space in last_insert_edge_block
                else {

                    if((index_counter < space_remaining) && (space_remaining != EDGE_BLOCK_SIZE)) {

                        // printf("Space Remaining: ID=%lu, source_vertex=%lu, last_insert_edge_offset=%lu, space_remaining=%lu, index_counter=%lu, target_vertex=%lu, last_insert_edge_block=%p\n", id, source_vertex, last_insert_edge_offset, space_remaining, index_counter, target_vertex, device_vertex_dictionary->last_insert_edge_block[source_vertex]);

                        // traverse to last insert edge block
                        unsigned long edge_entry_index = index_counter + last_insert_edge_offset;
                        device_vertex_dictionary->last_insert_edge_block[source_vertex]->edge_block_entry[edge_entry_index].destination_vertex = target_vertex;

                        return;
                    }

                }

            }

        }

    }

}

__global__ void batched_edge_inserts_EC_postprocessing(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees, unsigned long *d_prefix_sum_edge_blocks) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < batch_size) {

        unsigned long target_vertex = d_csr_edges[id];

        if(target_vertex != INFTY) {

            unsigned long source_vertex = device_binary_search(d_csr_offset, id, vertex_size - 1);
            unsigned long active_edge_block_count = device_vertex_dictionary->edge_block_count[source_vertex];
            unsigned long new_edge_block_count = d_prefix_sum_edge_blocks[source_vertex + 1] - d_prefix_sum_edge_blocks[source_vertex];

            unsigned long index_counter = id - d_csr_offset[source_vertex];

            if(!index_counter) {

                struct edge_block *base = pop_edge_block_address(d_prefix_sum_edge_blocks[vertex_size], d_prefix_sum_edge_blocks, source_vertex);

                if(device_vertex_dictionary->active_edge_count[source_vertex]) {

                    unsigned long last_insert_edge_offset = device_vertex_dictionary->last_insert_edge_offset[source_vertex];
                    unsigned long space_remaining = 0;
                    if(last_insert_edge_offset != 0)
                        space_remaining = EDGE_BLOCK_SIZE - last_insert_edge_offset;

                    // device_vertex_dictionary->edge_block_address[source_vertex] = base;
                    // device_vertex_dictionary->vertex_id[source_vertex] = source_vertex + 1;
                    // // device_vertex_dictionary->active_edge_count[source_vertex] = d_source_degrees[source_vertex];
                    device_vertex_dictionary->edge_block_count[source_vertex] += new_edge_block_count;
                    // // device_vertex_dictionary->edge_block_count[source_vertex] = ceil(double(d_source_degrees[source_vertex]) / EDGE_BLOCK_SIZE);

                    // device_vertex_dictionary->edge_block_count[source_vertex] += new_edge_block_count;
                    device_vertex_dictionary->active_edge_count[source_vertex] += d_source_degrees[source_vertex];

                    if(new_edge_block_count)
                        device_vertex_dictionary->last_insert_edge_block[source_vertex] = base + new_edge_block_count - 1;
                    // device_vertex_dictionary->last_insert_edge_block[source_vertex] = base + device_vertex_dictionary->edge_block_count[source_vertex] - 1;
                    device_vertex_dictionary->last_insert_edge_offset[source_vertex] = (d_source_degrees[source_vertex] - space_remaining) % EDGE_BLOCK_SIZE;

                }

                else {

                    device_vertex_dictionary->active_edge_count[source_vertex] += d_source_degrees[source_vertex];

                }
                // }
                // if(id > 4990) {

                //     printf("ID=%lu, source=%lu, before=%lu, new=%lu, after=%lu\n", id, source_vertex, active_edge_block_count, new_edge_block_count, device_vertex_dictionary->edge_block_count[source_vertex]);

                // }

            }

        }

    }

}

__global__ void copy_vertex_dictionary(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long *d_vertex_id, unsigned long *d_edge_block_count, unsigned int *d_active_edge_count, unsigned long *d_last_insert_edge_offset, struct edge_block **d_last_insert_edge_block, struct edge_block **d_edge_block_address, unsigned long vertex_size) {

    unsigned long id = blockDim.x * blockIdx.x + threadIdx.x;

    if(id < vertex_size) {

        d_vertex_id[id] = device_vertex_dictionary->vertex_id[id];
        d_edge_block_count[id] = device_vertex_dictionary->edge_block_count[id];
        d_active_edge_count[id] = device_vertex_dictionary->active_edge_count[id];
        d_last_insert_edge_offset[id] = device_vertex_dictionary->last_insert_edge_offset[id];
        if(device_vertex_dictionary->last_insert_edge_block[id] != NULL)
            d_last_insert_edge_block[id] = device_vertex_dictionary->last_insert_edge_block[id];
        if(device_vertex_dictionary->edge_block_address[id])
            d_edge_block_address[id] = device_vertex_dictionary->edge_block_address[id];

    }

}

__global__ void copy_vertex_dictionary_postprocessing(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long *d_vertex_id, unsigned long *d_edge_block_count, unsigned int *d_active_edge_count, unsigned long *d_last_insert_edge_offset, struct edge_block **d_last_insert_edge_block, struct edge_block **d_edge_block_address, unsigned long vertex_size) {

    device_vertex_dictionary->vertex_id = d_vertex_id;
    device_vertex_dictionary->edge_block_count = d_edge_block_count;
    device_vertex_dictionary->active_edge_count = d_active_edge_count;
    device_vertex_dictionary->last_insert_edge_offset = d_last_insert_edge_offset;
    device_vertex_dictionary->last_insert_edge_block = d_last_insert_edge_block;
    device_vertex_dictionary->edge_block_address = d_edge_block_address;

    // d_vertex_id[id] = device_vertex_dictionary->vertex_id[id];
    // d_edge_block_count[id] = device_vertex_dictionary->edge_block_count[id];
    // d_active_edge_count[id] = device_vertex_dictionary->active_edge_count[id];
    // d_last_insert_edge_offset[id] = device_vertex_dictionary->last_insert_edge_offset[id];
    // d_last_insert_edge_block[id] = device_vertex_dictionary->last_insert_edge_block[id];
    // d_edge_block_address[id] = device_vertex_dictionary->edge_block_address[id];


}

__global__ void vertex_insert_kernel(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long vertex_batch_size, unsigned long *d_vertex_batch) {

    unsigned long id = blockDim.x * blockIdx.x + threadIdx.x;

    if(id < vertex_batch_size) {

        device_vertex_dictionary->vertex_id[d_vertex_batch[id] - 1] = d_vertex_batch[id];

    }

}

__global__ void vertex_insert_kernel_postprocessing(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_batch_size) {

    device_vertex_dictionary->active_vertex_count += vertex_batch_size;

}

__device__ struct edge_block* traverse_bit_string(struct edge_block* root, unsigned long bit_string) {

    struct edge_block *curr = root;

    for( ; bit_string > 0  ; bit_string /= 10) {

        if(bit_string % 2)
            curr = curr->lptr;
        else
            curr = curr->rptr;

    }

    return curr;
}

__global__ void vertex_delete_kernel_directed(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long vertex_batch_size, unsigned long *d_vertex_batch) {

    unsigned long id = blockDim.x * blockIdx.x + threadIdx.x;

    // if(!id) {

    //     printf("\nVertex Batch: ");
    //     for(unsigned long i = 0 ; i < vertex_batch_size ; i++)
    //         printf("%lu ", d_vertex_batch[i]);
    //     printf("\n\n");

    // }

    if(id < vertex_size) {
        // if(id < 6) {

        unsigned long source_vertex = id;

        unsigned long edge_block_count = device_vertex_dictionary->edge_block_count[source_vertex];
        // printf("ID=%lu, source=%lu, edge_count=%lu\n", id, source_vertex, edge_block_count);

        for(unsigned long i = 0 ; i < edge_block_count ; i++) {


            unsigned long bit_string = bit_string_lookup[i];
            struct edge_block *root = device_vertex_dictionary->edge_block_address[source_vertex];
            root = traverse_bit_string(root, bit_string);

            if(root == NULL)
                printf("Null hit\n");

            for(unsigned long j = 0 ; j < EDGE_BLOCK_SIZE ; j++) {

                if(root->edge_block_entry[j].destination_vertex == 0)
                    break;
                else if(root->edge_block_entry[j].destination_vertex == INFTY)
                    continue;
                else {

                    // unsigned long vertex_located = 1;
                    // unsigned long target_vertex = root->edge_block_entry[j].destination_vertex;
                    // unsigned long vertex_located = 0;
                    // if(target_vertex != 1)
                    unsigned vertex_located = device_binary_search_find(d_vertex_batch, root->edge_block_entry[j].destination_vertex, vertex_batch_size - 1);
                    // if(id == 6) {
                    //     // vertex_located = device_binary_search_find(d_vertex_batch, target_vertex, vertex_batch_size - 1);
                    //     printf("ID=%lu, source=%lu, target=%lu, root=%p, vertex_located=%lu\n", id, source_vertex, root->edge_block_entry[j].destination_vertex, root, vertex_located);
                    // }
                    if(vertex_located)
                        root->edge_block_entry[j].destination_vertex = INFTY;

                }

            }

        }

    }

}

__global__ void vertex_delete_kernel_stage_1(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long vertex_batch_size, unsigned long *d_vertex_batch) {

    unsigned long id = blockDim.x * blockIdx.x + threadIdx.x;

    if(id < vertex_batch_size) {

        unsigned long source_vertex = d_vertex_batch[id] - 1;

        unsigned long edge_block_count = device_vertex_dictionary->edge_block_count[source_vertex];
        // printf("ID=%lu, source=%lu, edge_count=%lu\n", id, source_vertex, edge_block_count);

        for(unsigned long i = 0 ; i < edge_block_count ; i++) {


            unsigned long bit_string = bit_string_lookup[i];
            struct edge_block *root = device_vertex_dictionary->edge_block_address[source_vertex];
            root = traverse_bit_string(root, bit_string);


            for(unsigned long j = 0 ; j < EDGE_BLOCK_SIZE ; j++) {

                if(root->edge_block_entry[j].destination_vertex == 0)
                    break;
                else if(root->edge_block_entry[j].destination_vertex == INFTY)
                    continue;
                else {


                    unsigned long target_vertex = root->edge_block_entry[j].destination_vertex;
                    // if(target_vertex != INFTY) {

                    unsigned long edge_block_count_target = device_vertex_dictionary->edge_block_count[target_vertex - 1];

                    // printf("ID=%lu, source=%lu, root=%p, current_entry=%lu, edge_block_count_target=%lu\n", id, source_vertex + 1, root, target_vertex + 1, edge_block_count_target);

                    for(unsigned long k = 0 ; k < edge_block_count_target ; k++) {

                        unsigned long bit_string_target = bit_string_lookup[k];
                        struct edge_block *root_target = device_vertex_dictionary->edge_block_address[target_vertex - 1];
                        root_target = traverse_bit_string(root_target, bit_string_target);
                        // if(source_vertex == 0)
                        // printf("ID=%lu, source=%lu, root=%p, target=%lu, target_edge_blocks=%lu, root_target=%p inside\n", id, source_vertex, root, target_vertex, edge_block_count_target, root_target);

                        for(unsigned long l = 0 ; l < EDGE_BLOCK_SIZE ; l++) {

                            if(root_target->edge_block_entry[l].destination_vertex == 0)
                                break;
                            else if(root_target->edge_block_entry[l].destination_vertex == INFTY)
                                continue;
                            else {

                                if(root_target->edge_block_entry[l].destination_vertex == (source_vertex + 1)) {
                                    root_target->edge_block_entry[l].destination_vertex = INFTY;
                                    // device_vertex_dictionary->active_edge_count[target_vertex - 1]--;
                                    // not updating active edge count here for now
                                }


                            }

                        }

                    }

                    // }

                }

            }

        }

    }

}

__global__ void vertex_delete_kernel_stage_2_preprocessing(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long vertex_batch_size, unsigned long *d_vertex_batch, unsigned long *d_edge_block_count_free_vector) {

    unsigned long id = blockDim.x * blockIdx.x + threadIdx.x;

    if(id < vertex_batch_size) {

        if((id == (vertex_batch_size - 1)) || (d_vertex_batch[id] != d_vertex_batch[id + 1])) {

            // printf("ID=%lu inside vertex delete kernel\n", id);

            // unsigned long edge_block_count = device_vertex_dictionary->edge_block_count[d_vertex_batch[id] - 1];

            d_edge_block_count_free_vector[id] = device_vertex_dictionary->edge_block_count[d_vertex_batch[id] - 1];

        }

    }

}

__global__ void vertex_delete_kernel_stage_2(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long vertex_batch_size, unsigned long *d_vertex_batch, unsigned long *d_edge_count_free_vector) {

    unsigned long id = blockDim.x * blockIdx.x + threadIdx.x;

    if(id < vertex_batch_size) {

        if((id == (vertex_batch_size - 1)) || (d_vertex_batch[id] != d_vertex_batch[id + 1])) {

            // printf("ID=%lu inside vertex delete kernel\n", id);

            // iteratively push all edge blocks
            unsigned long edge_block_count = device_vertex_dictionary->edge_block_count[d_vertex_batch[id] - 1];

            // printf("ID=%lu, source=%lu, root=%p, current_entry=%lu, edge_block_count_target=%lu\n", id, source_vertex + 1, root, target_vertex + 1, edge_block_count_target);

            unsigned long start_index = d_edge_count_free_vector[id];

            for(unsigned long k = 0 ; k < edge_block_count ; k++) {

                unsigned long bit_string = bit_string_lookup[k];
                struct edge_block *root = device_vertex_dictionary->edge_block_address[d_vertex_batch[id] - 1];
                root = traverse_bit_string(root, bit_string);

                // push root to edge_queue
                // printf
                d_e_queue.edge_block_address[d_e_queue.rear + start_index + k] = root;

                // if(source_vertex == 0)
                // printf("ID=%lu, source=%lu, root=%p, target=%lu, target_edge_blocks=%lu, root_target=%p inside\n", id, source_vertex, root, target_vertex, edge_block_count_target, root_target);

                // for(unsigned long l = 0 ; l < EDGE_BLOCK_SIZE ; l++) {

                //     if(root->edge_block_entry[l].destination_vertex == 0)
                //         break;
                //     else if(root_target->edge_block_entry[l].destination_vertex == INFTY)
                //         continue;
                //     else {

                //         if(root_target->edge_block_entry[l].destination_vertex == (source_vertex + 1)) {
                //             root_target->edge_block_entry[l].destination_vertex = INFTY;
                //             // device_vertex_dictionary->active_edge_count[target_vertex - 1]--;
                //             // not updating active edge count here for now
                //         }


                //     }

                // }

            }

            device_vertex_dictionary->vertex_id[d_vertex_batch[id] - 1] = 0;
            device_vertex_dictionary->active_edge_count[d_vertex_batch[id] - 1] = 0;
            device_vertex_dictionary->edge_block_count[d_vertex_batch[id] - 1] = 0;
            device_vertex_dictionary->edge_block_address[d_vertex_batch[id] - 1] = NULL;

        }

    }

}

__global__ void vertex_delete_kernel_postprocessing(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_batch_size, unsigned long *d_edge_block_free_vector) {

    // device_vertex_dictionary->active_vertex_count -= vertex_batch_size;
    d_e_queue.rear = (d_e_queue.rear + (unsigned int)d_edge_block_free_vector[vertex_batch_size]) % EDGE_PREALLOCATE_LIST_SIZE;
    // printf("Queue front is %u(%p) and rear is %u\n", d_e_queue.front, d_e_queue.edge_block_address[d_e_queue.front], d_e_queue.rear);


}

__global__ void batched_edge_inserts_preprocessing_v6(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, unsigned long total_edge_blocks_count_batch, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks, struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long batch_number, unsigned long batch_size, unsigned long start_index_batch, unsigned long end_index_batch, unsigned long* d_csr_offset, unsigned long* d_csr_edges, unsigned long* d_source_vector) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        unsigned long start_index = d_prefix_sum_edge_blocks[id] + id;
        unsigned long end_index = d_prefix_sum_edge_blocks[id + 1] + id + 1;

        for(unsigned long i = start_index ; i < end_index ; i++)
            d_source_vector[i] = id;

    }

    // if(id == 0) {

    //     for(unsigned long i = 0 ; i < d_prefix_sum_edge_blocks[vertex_size] + vertex_size ; i++)
    //         printf("%lu ", d_source_vector[i]);
    //     printf("\n");

    // }

}

__global__ void batched_edge_inserts_v6(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, unsigned long total_edge_blocks_count_batch, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks, struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long batch_number, unsigned long batch_size, unsigned long start_index_batch, unsigned long end_index_batch, unsigned long* d_csr_offset, unsigned long* d_csr_edges, unsigned long* d_source_vector) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < (vertex_size + d_prefix_sum_edge_blocks[vertex_size])) {

        unsigned long source_vertex = d_source_vector[id];
        unsigned long index_counter = id - d_prefix_sum_edge_blocks[source_vertex] - source_vertex;
        struct edge_block *device_edge_block;
        // unsigned long edge_blocks_used_present = 0;
        unsigned long edge_blocks_required = d_prefix_sum_edge_blocks[source_vertex + 1] - d_prefix_sum_edge_blocks[source_vertex];
        unsigned long bit_string = bit_string_lookup[index_counter - 1];


        if(index_counter != 0) {
            device_edge_block = parallel_pop_from_edge_preallocate_queue_v1(total_edge_blocks_count_batch, d_prefix_sum_edge_blocks, source_vertex, index_counter);

            // fixing structure

            if((index_counter - 1) < ((edge_blocks_required / 2) - 1)) {

                device_edge_block->lptr = device_edge_block + (2 * (index_counter - 1)) + 1;
                device_edge_block->rptr = device_edge_block + (2 * (index_counter - 1)) + 2;

            }

            else if((index_counter - 1) == ((edge_blocks_required / 2) - 1)) {

                if(edge_blocks_required % 2) {

                    // curr->value = array [i];
                    device_edge_block->lptr = device_edge_block + (2 * (index_counter - 1)) + 1;
                    device_edge_block->rptr = device_edge_block + (2 * (index_counter - 1)) + 2;

                    // printf("Inserted internal node v1 %p for id %lu\n", curr, id);

                }
                else {
                    // curr->value = array [i];
                    device_edge_block->lptr = device_edge_block + (2 * (index_counter - 1)) + 1;

                    // printf("Inserted internal node v2 %p for id %lu, lptr is %p\n", curr, id, *device_edge_block + (2 * (i - 1)) + 1);

                }

            }

            else if((index_counter - 1) > (edge_blocks_required / 2)) {

                device_edge_block->lptr = NULL;
                device_edge_block->rptr = NULL;

            }

            unsigned long start_index = d_csr_offset[source_vertex] + ((index_counter - 1) * EDGE_BLOCK_SIZE);
            unsigned long end_index;
            if(d_csr_offset[source_vertex + 1] > (d_csr_offset[source_vertex] + (index_counter * EDGE_BLOCK_SIZE)))
                end_index = d_csr_offset[source_vertex] + (index_counter * EDGE_BLOCK_SIZE);
            else
                end_index = d_csr_offset[source_vertex + 1];

            unsigned long edge_block_entry_count = 0;
            for(unsigned long i = start_index ; i < end_index ; i++) {

                device_edge_block->edge_block_entry[edge_block_entry_count++].destination_vertex = d_csr_edges[i];
                device_edge_block->active_edge_count++;

            }

            // printf("Hello from id %lu and source %lu, counter %lu, edge_blocks_required %lu, prefix_sum_value %lu, bit_string %lu, address %p\n", id, source_vertex, index_counter, edge_blocks_required, d_prefix_sum_edge_blocks[source_vertex], bit_string_lookup[index_counter - 1], device_edge_block);



        }
        else {

            device_vertex_dictionary->vertex_adjacency[source_vertex]->edge_block_address = d_e_queue.edge_block_address[d_e_queue.front + d_prefix_sum_edge_blocks[source_vertex]];

            // printf("Hello from id %lu and source %lu, counter %lu, edge_blocks_required %lu, prefix_sum_value %lu, bit_string %lu, address %p\n", id, source_vertex, index_counter, edge_blocks_required, d_prefix_sum_edge_blocks[source_vertex], bit_string_lookup[index_counter - 1], device_edge_block);

        }




    }

}

__global__ void update_edge_queue(unsigned long pop_count) {


    d_e_queue.count -= (unsigned int)pop_count;

    // printf("Edge Queue before, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);

    if((d_e_queue.front + (unsigned int)pop_count - 1) % EDGE_PREALLOCATE_LIST_SIZE == d_e_queue.rear) {

        d_e_queue.front = -1;
        d_e_queue.rear = -1;

    }
    else
        d_e_queue.front = (d_e_queue.front + (unsigned int)pop_count) % EDGE_PREALLOCATE_LIST_SIZE;

    // printf("Queue front is %u(%p) and rear is %u\n", d_e_queue.front, d_e_queue.edge_block_address[d_e_queue.front], d_e_queue.rear);

    // printf("Edge Queue before, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);


}

__global__ void search_edge_kernel(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long total_search_threads, unsigned long search_source, unsigned long search_destination, unsigned long *d_search_flag) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < total_search_threads) {

        unsigned long edge_block = id / EDGE_BLOCK_SIZE;
        unsigned long edge_block_entry = id % EDGE_BLOCK_SIZE;

        // printf("Search kernel launch test for source %lu and destination %lu, edge_block %lu and entry %lu\n", search_source, search_destination, edge_block, edge_block_entry);

        struct adjacency_sentinel_new *edge_sentinel = device_vertex_dictionary->vertex_adjacency[search_source - 1];
        unsigned long extracted_value;

        struct edge_block *itr = edge_sentinel->edge_block_address + (edge_block);
        extracted_value = itr->edge_block_entry[edge_block_entry].destination_vertex;

        // extracted_value = edge_sentinel->edge_block_address[edge_block]->edge_block_entry[edge_block_entry].destination_vertex;

        if(extracted_value == search_destination) {
            *d_search_flag = 1;
            // printf("Edge exists\n");
        }

    }

}

__device__ struct edge_block *search_blocks[SEARCH_BLOCKS_COUNT];
__device__ unsigned long search_index = 0;

__device__ void inorderTraversal_search(struct edge_block *root) {

    if(root == NULL)
        return;
    else {

        inorderTraversal_search(root->lptr);
        search_blocks[search_index++] = root;
        inorderTraversal_search(root->rptr);

    }

}

__global__ void search_pre_processing(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long curr_source) {

    struct edge_block *root = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address;

    for(unsigned long i = 0 ; i < device_vertex_dictionary->edge_block_count[curr_source - 1] ; i++)
        search_blocks[i] = NULL;
    search_index = 0;

    inorderTraversal_search(root);

}

__global__ void search_edge_kernel_v1(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long total_search_threads, unsigned long search_source, unsigned long search_destination, unsigned long *d_search_flag) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < total_search_threads) {

        __syncthreads();

        *d_search_flag = 0;

        unsigned long edge_block = id / EDGE_BLOCK_SIZE;
        unsigned long edge_block_entry = id % EDGE_BLOCK_SIZE;

        // printf("Search kernel launch test for source %lu and destination %lu, edge_block %lu and entry %lu\n", search_source, search_destination, edge_block, edge_block_entry);

        // struct adjacency_sentinel_new *edge_sentinel = device_vertex_dictionary->vertex_adjacency[search_source - 1];
        unsigned long extracted_value;

        // struct edge_block *itr = edge_sentinel->edge_block_address + (edge_block);
        // extracted_value = itr->edge_block_entry[edge_block_entry].destination_vertex;
        extracted_value = search_blocks[edge_block]->edge_block_entry[edge_block_entry].destination_vertex;

        // extracted_value = edge_sentinel->edge_block_address[edge_block]->edge_block_entry[edge_block_entry].destination_vertex;

        if(extracted_value == search_destination) {
            *d_search_flag = 1;
            // printf("Edge exists\n");
        }

    }

}



__global__ void delete_edge_kernel(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long total_search_threads, unsigned long search_source, unsigned long search_destination, unsigned long *d_search_flag) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < total_search_threads) {

        __syncthreads();

        unsigned long edge_block = id / EDGE_BLOCK_SIZE;
        unsigned long edge_block_entry = id % EDGE_BLOCK_SIZE;

        // printf("Search kernel launch test for source %lu and destination %lu, edge_block %lu and entry %lu\n", search_source, search_destination, edge_block, edge_block_entry);

        // struct adjacency_sentinel_new *edge_sentinel = device_vertex_dictionary->vertex_adjacency[search_source - 1];
        unsigned long extracted_value;

        // struct edge_block *itr = edge_sentinel->edge_block_address + (edge_block);
        // extracted_value = itr->edge_block_entry[edge_block_entry].destination_vertex;
        extracted_value = search_blocks[edge_block]->edge_block_entry[edge_block_entry].destination_vertex;

        // extracted_value = edge_sentinel->edge_block_address[edge_block]->edge_block_entry[edge_block_entry].destination_vertex;

        if(extracted_value == search_destination) {
            *d_search_flag = 1;
            search_blocks[edge_block]->edge_block_entry[edge_block_entry].destination_vertex = 0;
            // printf("Edge exists\n");
        }

    }

}

// __device__ struct edge_block *delete_blocks[SEARCH_BLOCKS_COUNT];
// __device__ unsigned long delete_index[] = 0;

__device__ struct edge_block *stack_traversal[1];
__device__ unsigned long stack_top = 0;

__device__ void preorderTraversal_batch_delete(struct edge_block *root, unsigned long start_index, unsigned long end_index, unsigned long *d_csr_edges) {

    if(root == NULL)
        return;
    else {

        struct edge_block *temp = root;
        // struct stack *s_temp = NULL;
        int flag = 1;
        while (flag) {			//Loop run untill temp is null and stack is empty

            if (temp) {

                // printf ("%p ", temp);

                for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

                    for(unsigned long j = start_index ; j < end_index ; j++) {

                        if(temp->edge_block_entry[i].destination_vertex == d_csr_edges[j])
                            temp->edge_block_entry[i].destination_vertex = 0;
                        // if(j >= BATCH_SIZE)
                        //     printf("Hit dispute\n");

                    }
                }

                stack_traversal[stack_top++] = root;
                // push (&s_temp, temp);
                temp = temp->lptr;

            }
            else {

                if (stack_top) {

                    // temp = pop (&s_temp);
                    temp = stack_traversal[--stack_top] = root;
                    temp = temp->rptr;

                }
                else
                    flag = 0;
            }
        }

        // preorderTraversal_batch_delete(root->lptr, start_index, end_index, d_csr_edges);

        // // search_blocks[search_index++] = root;

        // for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

        //     for(unsigned long j = start_index ; j < end_index ; j++) {

        //         // if(root->edge_block_entry[i].destination_vertex == d_csr_edges[j])
        //             root->edge_block_entry[i].destination_vertex = 0;
        //             if(j == BATCH_SIZE)
        //                 printf("Hit dispute\n");

        //     }
        // }


        // preorderTraversal_batch_delete(root->rptr, start_index, end_index, d_csr_edges);

    }

}

__device__ void inorderTraversal_batch_delete(struct edge_block *root, unsigned long start_index, unsigned long end_index, unsigned long *d_csr_edges) {

    if(root == NULL)
        return;
    else {

        inorderTraversal_batch_delete(root->lptr, start_index, end_index, d_csr_edges);

        // search_blocks[search_index++] = root;

        for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

            for(unsigned long j = start_index ; j < end_index ; j++) {

                // if(root->edge_block_entry[i].destination_vertex == d_csr_edges[j])
                root->edge_block_entry[i].destination_vertex = 0;
                if(j == BATCH_SIZE)
                    printf("Hit dispute\n");

            }
        }


        inorderTraversal_batch_delete(root->rptr, start_index, end_index, d_csr_edges);

    }

}


// __device__ struct edge_block *delete_blocks[VERTEX_BLOCK_SIZE][2];
// __device__ struct edge_block *delete_blocks_v1[EDGE_PREALLOCATE_LIST_SIZE];
__device__ struct edge_block *delete_blocks_v1[1];




// __device__ unsigned long delete_index[VERTEX_BLOCK_SIZE];
__device__ unsigned long delete_index[1];
// __device__ unsigned long delete_index_blocks[VERTEX_BLOCK_SIZE];
// __device__ unsigned long delete_source[EDGE_PREALLOCATE_LIST_SIZE];
__device__ unsigned long delete_source[1];
// __device__ unsigned long delete_source_counter[EDGE_PREALLOCATE_LIST_SIZE];

__device__ void inorderTraversal_batch_delete_v1(struct edge_block *root, unsigned long id, unsigned long offset, unsigned long *d_prefix_sum_edge_blocks) {

    if(root == NULL)
        return;
    else {

        inorderTraversal_batch_delete_v1(root->lptr, id, offset, d_prefix_sum_edge_blocks);

        // search_blocks[search_index++] = root;

        // delete_blocks[id][delete_index[id]] = root;

        // if(id != 0)
        // delete_blocks_v1[d_prefix_sum_edge_blocks[id] + delete_index[id]] = root;
        // else
        delete_blocks_v1[delete_index[id] + offset] = root;
        delete_source[delete_index[id]++ + offset] = id;
        // delete_source_counter[delete_index[id]] = delete_index[id]++;


        // printf("id is %lu, delete_blocks is %p and delete_source is %lu and delete_source_counter is %lu\n", id, delete_blocks[id][delete_index[id] - 1], delete_source[delete_index[id] - 1 + offset], delete_source_counter[delete_index[id] - 1]);

        // delete_source[id] = id;

        // for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

        //     for(unsigned long j = start_index ; j < end_index ; j++) {

        //         // if(root->edge_block_entry[i].destination_vertex == d_csr_edges[j])
        //             root->edge_block_entry[i].destination_vertex = 0;
        //             if(j == BATCH_SIZE)
        //                 printf("Hit dispute\n");

        //     }
        // }


        inorderTraversal_batch_delete_v1(root->rptr, id, offset, d_prefix_sum_edge_blocks);

    }

}

__global__ void batched_delete_preprocessing(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_degrees) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if((id < vertex_size) && (device_vertex_dictionary->vertex_adjacency[id]->edge_block_address != NULL)) {

        struct edge_block *root = device_vertex_dictionary->vertex_adjacency[id]->edge_block_address;

        // for(unsigned long i = 0 ; i < device_vertex_dictionary->edge_block_count[id] ; i++)
        //     search_blocks[i] = NULL;
        delete_index[id] = 0;
        // unsigned long offset = d_csr_offset[id];
        unsigned long offset = 0;
        if(id != 0)
            //     offset = 0;
            // else
            offset = d_prefix_sum_edge_blocks[id - 1];

        // unsigned long start_index = d_csr_offset[id];
        // unsigned long end_index = d_csr_offset[id + 1];

        // for(unsigned long i = start_index; i < end_index ; i++)
        // delete_source[delete_index[id]++] = id;

        inorderTraversal_batch_delete_v1(root, id, offset, d_prefix_sum_edge_blocks);



        // printf("ID is %lu, source is %lu, and address is %p\n", id, id, delete_blocks_v1[offset]);

    }

}

__global__ void batched_delete_kernel_v1(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long total_search_threads, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id < vertex_size) {
    if(id < total_search_threads) {

        // __syncthreads();

        // *d_search_flag = 0;


        // unsigned long edge_block_entry = edge_block_count % EDGE_BLOCK_SIZE;




        // struct adjacency_sentinel_new *edge_sentinel = device_vertex_dictionary->vertex_adjacency[search_source - 1];
        unsigned long extracted_value;
        unsigned long curr_source = delete_source[id];
        unsigned long start_index = d_csr_offset[curr_source];
        unsigned long end_index = d_csr_offset[curr_source + 1];

        // if(id == 0) {
        //     for(unsigned long i = 0 ; i < 10 ; i++) {
        //         printf("%lu\n", delete_source[i]);
        //     }
        //     printf("\n");
        //     for(unsigned long i = 0 ; i < 10 ; i++) {

        //         if(delete_blocks[i][0] != NULL)
        //             printf("%p\n", delete_blocks[i][0]);

        //     }
        //     printf("\n");
        // }


        unsigned long offset = 0;
        unsigned long edge_block_count = d_prefix_sum_edge_blocks[curr_source];
        if(curr_source != 0) {
            //     edge_block_count = d_prefix_sum_edge_blocks[curr_source];
            //     offset = 0;
            // }
            // else {
            edge_block_count = d_prefix_sum_edge_blocks[curr_source] - d_prefix_sum_edge_blocks[curr_source - 1];
            offset = d_prefix_sum_edge_blocks[curr_source - 1];
        }
        // unsigned long edge_block = delete_source_counter[curr_source];

        unsigned long edge_block = id - offset;

        // unsigned long edge_block = d_prefix_sum_edge_blocks;
        // if(curr_source != 0)
        //     edge_block = d_prefix_sum_edge_blocks[curr_source] - d_prefix_sum_edge_blocks[curr_source - 1];
        // if(curr_source == 0)
        // edge_block = id;
        // else
        // edge_block = id - d_prefix_sum_edge_blocks[curr_source - 1];

        // unsigned long tester = 0;
        // if(curr_source != 0)
        //     tester = id - offset;

        // printf("ID is %lu, source is %lu, edge_block is %lu, test is %lu and address is %p\n", id, curr_source, edge_block, id - offset, delete_blocks_v1[offset + edge_block]);

        // printf("Search kernel launch test for id %lu, edge_block_address %p, edge_block_count %lu, source %lu, edge_block %lu, start_index %lu and end_index %lu\n", id, delete_blocks[curr_source][edge_block], edge_block_count, curr_source, edge_block, start_index, end_index);


        // printf("ID is %lu, start_index is %lu, end_index is %lu\n", id, start_index, end_index);

        // struct edge_block *root = device_vertex_dictionary->vertex_adjacency[curr_source]->edge_block_address;

        // struct edge_block *itr = edge_sentinel->edge_block_address + (edge_block);
        // extracted_value = itr->edge_block_entry[edge_block_entry].destination_vertex;

        for(unsigned long j = 0 ; j < EDGE_BLOCK_SIZE ; j++) {



            for(unsigned long i = start_index ; i < end_index ; i++) {


                // delete_blocks_v1[d_prefix_sum_edge_blocks[curr_source] + edge_block]
                // if((delete_blocks[curr_source][edge_block] != NULL) && (delete_blocks[curr_source][edge_block]->edge_block_entry[j].destination_vertex == d_csr_edges[i]))
                //     delete_blocks[curr_source][edge_block]->edge_block_entry[j].destination_vertex = 0;

                if((delete_blocks_v1[offset + edge_block] != NULL) && (delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex == d_csr_edges[i]))
                    delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex = 0;


            }

        }

    }

    // if(id == 0)
    //     printf("Checkpoint final\n");

}

__device__ void preorderTraversal_batch_delete_edge_centric(struct edge_block *root, unsigned long id, unsigned long offset, unsigned long *d_csr_offset) {

    if(root == NULL)
        return;
    else {
        delete_blocks_v1[delete_index[id]++ + offset] = root;

        preorderTraversal_batch_delete_edge_centric(root->lptr, id, offset, d_csr_offset);

        // search_blocks[search_index++] = root;

        // delete_blocks[id][delete_index[id]] = root;

        // if(id != 0)
        // delete_blocks_v1[d_prefix_sum_edge_blocks[id] + delete_index[id]] = root;
        // else
        // delete_source[delete_index[id]++ + offset] = id;
        // delete_source_counter[delete_index[id]] = delete_index[id]++;


        // printf("id is %lu, delete_blocks is %p and delete_source is %lu and delete_source_counter is %lu\n", id, delete_blocks[id][delete_index[id] - 1], delete_source[delete_index[id] - 1 + offset], delete_source_counter[delete_index[id] - 1]);

        // delete_source[id] = id;

        // for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

        //     for(unsigned long j = start_index ; j < end_index ; j++) {

        //         // if(root->edge_block_entry[i].destination_vertex == d_csr_edges[j])
        //             root->edge_block_entry[i].destination_vertex = 0;
        //             if(j == BATCH_SIZE)
        //                 printf("Hit dispute\n");

        //     }
        // }


        preorderTraversal_batch_delete_edge_centric(root->rptr, id, offset, d_csr_offset);

    }

}

__global__ void batched_delete_preprocessing_edge_centric(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_degrees) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if((id < vertex_size) && (device_vertex_dictionary->edge_block_address[id] != NULL)) {

        struct edge_block *root = device_vertex_dictionary->edge_block_address[id];

        // for(unsigned long i = 0 ; i < device_vertex_dictionary->edge_block_count[id] ; i++)
        //     search_blocks[i] = NULL;
        delete_index[id] = 0;
        // unsigned long offset = d_csr_offset[id];
        unsigned long offset = 0;
        if(id != 0)
            //     offset = 0;
            // else
            offset = d_prefix_sum_edge_blocks[id - 1];

        // unsigned long start_index = d_csr_offset[id];
        // unsigned long end_index = d_csr_offset[id + 1];

        // for(unsigned long i = start_index; i < end_index ; i++)
        // delete_source[delete_index[id]++] = id;

        preorderTraversal_batch_delete_edge_centric(root, id, offset, d_csr_offset);

        // printf("ID is %lu, source is %lu, and address is %p\n", id, id, delete_blocks_v1[offset]);

    }

    if(id == 0)
        printf("Checkpoint final preprocessing\n");

}

__global__ void batched_delete_kernel_edge_centric(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long total_search_threads, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source, unsigned long *d_destination) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;



    // if(id < vertex_size) {
    if(id < total_search_threads) {

        // __syncthreads();

        // *d_search_flag = 0;


        // unsigned long edge_block_entry = edge_block_count % EDGE_BLOCK_SIZE;

        // if(id == 0) {

        //     for(unsigned long i = 0 ; i < BATCH_SIZE ; i++) {

        //         printf("Source is %lu and Destination is %lu\n", d_source[i], d_destination[i]);

        //     }

        // }


        // struct adjacency_sentinel_new *edge_sentinel = device_vertex_dictionary->vertex_adjacency[search_source - 1];
        // unsigned long extracted_value;
        unsigned long curr_source = d_source[id] - 1;
        // unsigned long start_index = d_csr_offset[curr_source];
        // unsigned long end_index = d_csr_offset[curr_source + 1];

        // if(id == 0) {
        //     for(unsigned long i = 0 ; i < 10 ; i++) {
        //         printf("%lu\n", delete_source[i]);
        //     }
        //     printf("\n");
        //     for(unsigned long i = 0 ; i < 10 ; i++) {

        //         if(delete_blocks[i][0] != NULL)
        //             printf("%p\n", delete_blocks[i][0]);

        //     }
        //     printf("\n");
        // }


        unsigned long offset = 0;
        unsigned long edge_block_count = device_vertex_dictionary->edge_block_count[curr_source];
        // unsigned long edge_block_count = d_prefix_sum_edge_blocks[curr_source];
        if(curr_source != 0) {
            //     edge_block_count = d_prefix_sum_edge_blocks[curr_source];
            //     offset = 0;
            // }
            // else {
            // edge_block_count = d_prefix_sum_edge_blocks[curr_source] - d_prefix_sum_edge_blocks[curr_source - 1];
            offset = d_prefix_sum_edge_blocks[curr_source - 1];
        }
        // unsigned long edge_block = delete_source_counter[curr_source];



        // unsigned long edge_block = d_prefix_sum_edge_blocks;
        // if(curr_source != 0)
        //     edge_block = d_prefix_sum_edge_blocks[curr_source] - d_prefix_sum_edge_blocks[curr_source - 1];
        // if(curr_source == 0)
        // edge_block = id;
        // else
        // edge_block = id - d_prefix_sum_edge_blocks[curr_source - 1];

        // unsigned long tester = 0;
        // if(curr_source != 0)
        //     tester = id - offset;

        // unsigned long delete_edge = d_csr_edges[id];

        // printf("ID is %lu, source is %lu, edge_block_count is %lu, edge_block is %lu, delete_edge is %lu and address is %p\n", id, d_source[id], edge_block_count, edge_block, d_destination[id], delete_blocks_v1[offset]);

        // printf("Search kernel launch test for id %lu, edge_block_count %lu, source %lu, edge_block %lu, start_index %lu and end_index %lu\n", id, edge_block_count, curr_source, edge_block, start_index, end_index);


        // printf("ID is %lu, start_index is %lu, end_index is %lu\n", id, start_index, end_index);

        // struct edge_block *root = device_vertex_dictionary->vertex_adjacency[curr_source]->edge_block_address;

        // struct edge_block *itr = edge_sentinel->edge_block_address + (edge_block);
        // extracted_value = itr->edge_block_entry[edge_block_entry].destination_vertex;

        // unsigned long breakFlag = 0;

        // unsigned long *destination_entry;

        for(unsigned long edge_block = 0 ; edge_block < edge_block_count ; edge_block++) {

            // printf("ID is %lu, source is %lu, edge_block_count is %lu, edge_block is %lu, delete_edge is %lu and address is %p\n", id, d_source[id], edge_block_count, edge_block, d_destination[id], delete_blocks_v1[offset + edge_block]);


            if(delete_blocks_v1[offset + edge_block] != NULL) {

                // unsigned long edge_block_entry_count = delete_blocks_v1[offset + edge_block]->active_edge_count;

                // for(unsigned long j = 0 ; j < delete_blocks_v1[offset + edge_block]->active_edge_count ; j++) {
                for(unsigned long j = 0 ; j < EDGE_BLOCK_SIZE ; j++) {



                    // for(unsigned long i = start_index ; i < end_index ; i++) {


                    // delete_blocks_v1[d_prefix_sum_edge_blocks[curr_source] + edge_block]
                    // if((delete_blocks[curr_source][edge_block] != NULL) && (delete_blocks[curr_source][edge_block]->edge_block_entry[j].destination_vertex == d_csr_edges[i]))
                    //     delete_blocks[curr_source][edge_block]->edge_block_entry[j].destination_vertex = 0;

                    // if(delete_blocks_v1[offset + edge_block] != NULL) {

                    // destination_entry = &(delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex);

                    // if(*destination_entry == d_destination[id])
                    //     *destination_entry = INFTY;
                    // else if(*destination_entry == 0)
                    //     goto exit_delete;


                    if(delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex == d_destination[id])
                        delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex = INFTY;

                    else if(delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex == 0)
                        goto exit_delete;


                    // }

                    // if((delete_blocks_v1[offset + edge_block] != NULL) && (delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex == d_destination[id]))
                    //     delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex = INFTY;

                    // else if((delete_blocks_v1[offset + edge_block] != NULL) && (delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex == 0)) {


                    //     goto exit_delete;
                    //     // breakFlag = 1;
                    //     // break;

                    // }



                    // }

                }

            }

            // if(breakFlag)
            // break;

            // printf("\n");

        }

        exit_delete:

    }

    // if(id == 0)
    //     printf("Checkpoint final\n");

}

__global__ void batched_delete_kernel_edge_centric_parallelized(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long total_search_threads, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source, unsigned long *d_destination) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id < vertex_size) {
    if(id < total_search_threads) {

        // __syncthreads();

        // unsigned long curr_source = d_source[id] - 1;
        // flagged
        unsigned long curr_source = d_source[id / EDGE_BLOCK_SIZE] - 1;

        unsigned long edge_block_entry = id % EDGE_BLOCK_SIZE;

        // goto exit_delete_v1;


        unsigned long offset = 0;
        unsigned long edge_block_count = device_vertex_dictionary->vertex_adjacency[curr_source]->edge_block_count;
        // unsigned long edge_block_count = d_prefix_sum_edge_blocks[curr_source];
        // unsigned long edge_block = 0;


        if(curr_source != 0) {
            //     edge_block_count = d_prefix_sum_edge_blocks[curr_source];
            //     offset = 0;
            // }
            // else {

            // edge_block_count = d_prefix_sum_edge_blocks[curr_source] - d_prefix_sum_edge_blocks[curr_source - 1];
            offset = d_prefix_sum_edge_blocks[curr_source - 1];
        }


        // if(id == 32)


        for(unsigned long edge_block = 0; edge_block < edge_block_count ; edge_block++) {

            // printf("ID is %lu, source is %lu, edge_block_count is %lu, edge_block is %lu, delete_edge is %lu and address is %p\n", id, curr_source, edge_block_count, edge_block, d_destination[id], delete_blocks_v1[offset + edge_block]);
            // if(curr_source == 4)
            //     printf("ID is %lu, source is %lu, destination is %lu, entry is %lu\n", id, curr_source + 1, d_destination[id / EDGE_BLOCK_SIZE], delete_blocks_v1[offset + edge_block]->edge_block_entry[edge_block_entry].destination_vertex);


            if(delete_blocks_v1[offset + edge_block] != NULL) {

                // for(unsigned long j = 0 ; j < EDGE_BLOCK_SIZE ; j++) {

                if(delete_blocks_v1[offset + edge_block]->edge_block_entry[edge_block_entry].destination_vertex == d_destination[id / EDGE_BLOCK_SIZE])
                    delete_blocks_v1[offset + edge_block]->edge_block_entry[edge_block_entry].destination_vertex = INFTY;

                // else if(delete_blocks_v1[offset + edge_block]->edge_block_entry[edge_block_entry].destination_vertex == 0)
                //     goto exit_delete;

                // }

            }

        }

        exit_delete_v1:

    }

}

__global__ void batched_delete_kernel(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long total_search_threads, unsigned long *d_csr_offset, unsigned long *d_csr_edges) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        // __syncthreads();

        // unsigned long search_source;
        // for(unsigned long i = 0 ; i < vertex_size + 1 ; i++) {

        //     if(id < d_csr_offset[i]) {
        //         search_source = i - 1;
        //         break;
        //     }

        // }
        // unsigned long index = id - d_csr_offset[search_source];
        // unsigned long search_destination = d_csr_edges[id];
        unsigned long start_index = d_csr_offset[id];
        unsigned long end_index = d_csr_offset[id + 1];


        // printf("ID is %lu, start_index is %lu, end_index is %lu\n", id, start_index, end_index);

        struct edge_block *root = device_vertex_dictionary->vertex_adjacency[id]->edge_block_address;
        if((root != NULL) && (root->active_edge_count > 0) && (start_index < end_index))
            inorderTraversal_batch_delete(root, start_index, end_index, d_csr_edges);
        // preorderTraversal_batch_delete(root, start_index, end_index, d_csr_edges);

        // unsigned long edge_block = id / EDGE_BLOCK_SIZE;
        // unsigned long edge_block_entry = id % EDGE_BLOCK_SIZE;

        // printf("Search kernel launch test for source %lu and destination %lu, edge_block %lu and entry %lu\n", search_source, search_destination, edge_block, edge_block_entry);

        // struct adjacency_sentinel_new *edge_sentinel = device_vertex_dictionary->vertex_adjacency[search_source - 1];
        // unsigned long extracted_value;

        // // struct edge_block *itr = edge_sentinel->edge_block_address + (edge_block);
        // // extracted_value = itr->edge_block_entry[edge_block_entry].destination_vertex;
        // extracted_value = search_blocks[edge_block]->edge_block_entry[edge_block_entry].destination_vertex;

        // // extracted_value = edge_sentinel->edge_block_address[edge_block]->edge_block_entry[edge_block_entry].destination_vertex;

        // if(extracted_value == search_destination) {
        //     *d_search_flag = 1;
        //     search_blocks[edge_block]->edge_block_entry[edge_block_entry].destination_vertex = 0;
        //     // printf("Edge exists\n");
        // }

    }

    // if(id == 0)
    //     printf("Checkpoint final\n");

}

// __device__ struct edge_block* traverse_bit_string(struct edge_block* root, unsigned long bit_string) {

//     struct edge_block *curr = root;

//     for( ; bit_string > 0  ; bit_string /= 10) {

//         if(bit_string % 2)
//             curr = curr->lptr;
//         else
//             curr = curr->rptr;

//     }

//     return curr;
// }

__global__ void device_prefix_sum_calculation_preprocessing(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_prefix_sum_edge_blocks) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        d_prefix_sum_edge_blocks[id] = device_vertex_dictionary->edge_block_count[id];

        // if(id == 15981872)
        //     printf("ID=%lu, prefix_sum is %lu\n", id, d_prefix_sum_edge_blocks[id]);

        // unsigned long start_index = d_prefix_sum_edge_blocks[id];
        // unsigned long end_index = d_prefix_sum_edge_blocks[id + 1];

        // // source vector values start from 0
        // for(unsigned long i = start_index ; i < end_index ; i++) {

        //     d_source_vector[i] = id;

        // }

    }

    // if (id == 0) {

    //     printf("last index is %lu\n", d_prefix_sum_edge_blocks[vertex_size]);
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_prefix_sum_edge_blocks[i]);

    //     printf("\n");

    //     // printf("Size is %lu\n", d_prefix_sum_edge_blocks[vertex_size]);

    //     for(unsigned long i = 0 ; i < d_prefix_sum_edge_blocks[vertex_size] ; i++)
    //         printf("%lu ", d_source_vector[i]);

    // }

}

__global__ void device_prefix_sum_calculation(unsigned long *d_edge_blocks_count, unsigned long vertex_size, unsigned long *d_prefix_sum_edge_blocks_new) {

    for(unsigned long i = 0 ; i < vertex_size ; i++)
        d_prefix_sum_edge_blocks_new[i + 1] = d_prefix_sum_edge_blocks_new[i] + d_edge_blocks_count[i];

}

__global__ void batched_delete_preprocessing_edge_block_centric(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_degrees, unsigned long *d_source_vector) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        unsigned long start_index = d_prefix_sum_edge_blocks[id];
        unsigned long end_index = d_prefix_sum_edge_blocks[id + 1];

        // source vector values start from 0
        for(unsigned long i = start_index ; i < end_index ; i++) {

            d_source_vector[i] = id;

        }

    }

    // if (id == 0) {

    //     printf("last index is %lu\n", d_prefix_sum_edge_blocks[vertex_size]);
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_prefix_sum_edge_blocks[i]);

    //     printf("\n");

    //     // printf("Size is %lu\n", d_prefix_sum_edge_blocks[vertex_size]);

    //     for(unsigned long i = 0 ; i < d_prefix_sum_edge_blocks[vertex_size] ; i++)
    //         printf("%lu ", d_source_vector[i]);

    // }

}

__global__ void batched_delete_preprocessing_edge_block_centric_v4(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_degrees, unsigned long *d_source_vector) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        unsigned long vertex_edge_blocks = d_prefix_sum_edge_blocks[id + 1] - d_prefix_sum_edge_blocks[id];
        unsigned long batch_degree = d_csr_offset[id + 1] - d_csr_offset[id];

        d_source_vector[id] = vertex_edge_blocks * batch_degree;


        // unsigned long start_index = d_prefix_sum_edge_blocks[id];
        // unsigned long end_index = d_prefix_sum_edge_blocks[id + 1];

        // // source vector values start from 0
        // for(unsigned long i = start_index ; i < end_index ; i++) {

        //     d_source_vector[i] = id;

        // }

    }

    // if (id == 0) {

    //     // printf("last index is %lu\n", d_prefix_sum_edge_blocks[vertex_size]);
    //     // for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //     //     printf("%lu ", d_prefix_sum_edge_blocks[i]);

    //     // printf("\n");

    //     // // printf("Size is %lu\n", d_prefix_sum_edge_blocks[vertex_size]);

    //     for(unsigned long i = 0 ; i < vertex_size ; i++)
    //         printf("%lu ", d_source_vector[i]);

    // }

}

__global__ void batched_delete_preprocessing_EC_LD(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_degrees, unsigned long *d_source_vector) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        unsigned long vertex_edge_blocks = d_prefix_sum_edge_blocks[id + 1] - d_prefix_sum_edge_blocks[id];
        unsigned long batch_degree = d_csr_offset[id + 1] - d_csr_offset[id];

        d_source_vector[id] = vertex_edge_blocks * batch_degree;

        // printf("ID=%lu, edge_blocks=%lu, batch_degree=%lu\n", id, vertex_edge_blocks, batch_degree);

        // unsigned long start_index = d_prefix_sum_edge_blocks[id];
        // unsigned long end_index = d_prefix_sum_edge_blocks[id + 1];

        // // source vector values start from 0
        // for(unsigned long i = start_index ; i < end_index ; i++) {

        //     d_source_vector[i] = id;

        // }

    }

    // if (id == 0) {

    //     // printf("last index is %lu\n", d_prefix_sum_edge_blocks[vertex_size]);
    //     // for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //     //     printf("%lu ", d_prefix_sum_edge_blocks[i]);

    //     // printf("\n");

    //     // // printf("Size is %lu\n", d_prefix_sum_edge_blocks[vertex_size]);

    //     printf("Source vector in preprocessing\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++)
    //         printf("%lu ", d_source_vector[i]);
    //     printf("\n");

    // }

}

__global__ void batched_delete_preprocessing_LB(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_degrees, unsigned long *d_source_vector) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        unsigned long vertex_degree = (d_prefix_sum_edge_blocks[id + 1] - d_prefix_sum_edge_blocks[id]) * EDGE_BLOCK_SIZE;
        // unsigned long vertex_degree = device_vertex_dictionary->active_edge_count[id];
        unsigned long batch_degree = d_csr_offset[id + 1] - d_csr_offset[id];

        d_source_vector[id] = vertex_degree * batch_degree;

        // printf("ID=%lu, vertex_degree=%lu, batch_degree=%lu\n", id, vertex_degree, batch_degree);

        // unsigned long start_index = d_prefix_sum_edge_blocks[id];
        // unsigned long end_index = d_prefix_sum_edge_blocks[id + 1];

        // // source vector values start from 0
        // for(unsigned long i = start_index ; i < end_index ; i++) {

        //     d_source_vector[i] = id;

        // }

    }

    // if (id == 0) {

    //     printf("Prefix sum for VC_LB\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++) {

    //         printf("%lu ", d_source_vector[i]);

    //     }
    //     printf("\n");

    // //     // printf("last index is %lu\n", d_prefix_sum_edge_blocks[vertex_size]);
    // //     // for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    // //     //     printf("%lu ", d_prefix_sum_edge_blocks[i]);

    // //     // printf("\n");

    // //     // // printf("Size is %lu\n", d_prefix_sum_edge_blocks[vertex_size]);

    // //     for(unsigned long i = 0 ; i < vertex_size ; i++)
    // //         printf("%lu ", d_source_vector[i]);

    // }

}

__global__ void batched_delete_kernel_edge_block_centric(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if((id < d_prefix_sum_edge_blocks[vertex_size])) {

        unsigned long source_vertex = d_source_vector[id];
        unsigned long start_index = d_csr_offset[source_vertex];
        unsigned long end_index = d_csr_offset[source_vertex + 1];

        if(start_index < end_index) {


            unsigned long index_counter = id - d_prefix_sum_edge_blocks[source_vertex];
            unsigned long bit_string = bit_string_lookup[index_counter];
            struct edge_block *root = device_vertex_dictionary->edge_block_address[source_vertex];

            root = traverse_bit_string(root, bit_string);



            // printf("id=%lu, source=%lu, counter=%lu, bit_string=%lu, edge_block=%p, start_index=%lu, end_index=%lu\n", id, d_source_vector[id], index_counter, bit_string, root, start_index, end_index);

            for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

                if(root->edge_block_entry[i].destination_vertex == 0)
                    break;
                else {

                    for(unsigned long j = start_index ; j < end_index ; j++) {

                        if(root->edge_block_entry[i].destination_vertex == d_csr_edges[j]) {
                            root->edge_block_entry[i].destination_vertex = INFTY;
                            device_vertex_dictionary->active_edge_count[source_vertex]--;
                            break;
                        }

                    }

                }

            }

            // for(unsigned long i = start_index ; i < end_index ; i++) {

            //     for(unsigned long j = 0 ; j < EDGE_BLOCK_SIZE ; j++) {

            //         if(root->edge_block_entry[j].destination_vertex == d_csr_edges[i]) {
            //             root->edge_block_entry[j].destination_vertex = INFTY;
            //             continue;
            //         }
            //         else if(root->edge_block_entry[j].destination_vertex == 0)
            //             break;

            //     }

            // }

            // exit_point_1:

            // for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

            //     if(root->edge_block_entry[i].destination_vertex == 0)
            //         break;
            //     else {

            //         for(unsigned long j = start_index ; j < end_index ; j++) {

            //             if(root->edge_block_entry[i].destination_vertex == d_csr_edges[j]) {
            //                 root->edge_block_entry[i].destination_vertex = INFTY;
            //                 break;
            //             }

            //         }

            //     }

            // }

            // if(id == 0) {

            //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
            //         printf("%lu ", d_csr_offset[i]);
            //     printf("\n");
            //     for(unsigned long i = 0 ; i < batch_size ; i++)
            //         printf("%lu ", d_csr_edges[i]);
            //     printf("\n");

            // }

        }

    }

}

__global__ void batched_delete_kernel_VC(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id == 0) {

    //     printf("CSR offset\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_csr_offset[i]);
    //     printf("\n");
    //     printf("CSR edges\n");
    //     for(unsigned long i = 0 ; i < batch_size + 1 ; i++)
    //         printf("%lu ", d_csr_edges[i]);
    //     printf("New source vector\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_source_vector[i]);
    //     printf("\n\n");

    // }

    if((id < vertex_size)) {

        // unsigned long source_vertex = device_binary_search(d_source_vector, id, vertex_size + 1);
        unsigned long source_vertex = id;

        // unsigned long index_counter = id - d_source_vector[source_vertex];


        unsigned long start_index = d_csr_offset[source_vertex];
        unsigned long end_index = d_csr_offset[source_vertex + 1];
        unsigned long source_degree_batch = end_index - start_index;

        // if(start_index < end_index) {


        //     unsigned long index_counter = id - d_prefix_sum_edge_blocks[source_vertex];
        if(source_degree_batch > 0) {

            unsigned long source_degree_edge_block_count = d_prefix_sum_edge_blocks[source_vertex + 1] - d_prefix_sum_edge_blocks[source_vertex];

            // unsigned long target_vertex = d_csr_edges[d_csr_offset[source_vertex] + (index_counter % source_degree_batch)];


            for(unsigned long z = 0 ; z < source_degree_edge_block_count ; z++) {

                unsigned long bit_string = bit_string_lookup[z];
                struct edge_block *root = device_vertex_dictionary->edge_block_address[source_vertex];

                root = traverse_bit_string(root, bit_string);

                // if(root == NULL)
                //     printf("null hit at id=%lu\n", id);
                // else
                // printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_degree_edge_block_count, target_vertex, (index_counter % source_degree_batch), bit_string, root);


                //     // printf("id=%lu, source=%lu, counter=%lu, bit_string=%lu, edge_block=%p, start_index=%lu, end_index=%lu\n", id, d_source_vector[id], index_counter, bit_string, root, start_index, end_index);

                for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

                    if(root->edge_block_entry[i].destination_vertex == 0)
                        break;
                    else {

                        for(unsigned long j = start_index ; j < end_index ; j++) {
                            // printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, current_vertex=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_degree_edge_block_count, root->edge_block_entry[i].destination_vertex, target_vertex, (index_counter % source_degree_batch), bit_string, root);

                            if((root->edge_block_entry[i].destination_vertex == d_csr_edges[j]) && (d_csr_edges[j] != INFTY)) {
                                root->edge_block_entry[i].destination_vertex = INFTY;
                                // device_vertex_dictionary->active_edge_count[source_vertex]--;
                                atomicDec(&(device_vertex_dictionary->active_edge_count[source_vertex]), INFTY);
                                // root->active_edge_count--;
                                break;
                            }

                        }

                    }

                }

            }
        }

        //     // if(id == 0) {

        //     //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
        //     //         printf("%lu ", d_csr_offset[i]);
        //     //     printf("\n");
        //     //     for(unsigned long i = 0 ; i < batch_size ; i++)
        //     //         printf("%lu ", d_csr_edges[i]);
        //     //     printf("\n");

        //     // }

        // }

    }

}

__global__ void batched_delete_kernel_VC_LB(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long vertex_LB_factor) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id == 0) {

    //     printf("CSR offset\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_csr_offset[i]);
    //     printf("\n");
    //     printf("CSR edges\n");
    //     for(unsigned long i = 0 ; i < batch_size + 1 ; i++)
    //         printf("%lu ", d_csr_edges[i]);
    //     printf("New source vector\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_source_vector[i]);
    //     printf("\n\n");

    // }

    if((id < ceil((double)vertex_size / vertex_LB_factor))) {

        unsigned long vertex_start_index = id * vertex_LB_factor;
        unsigned long vertex_end_index = (id + 1) * vertex_LB_factor;

        if(vertex_end_index > vertex_size)
            vertex_end_index = vertex_size;

        for(unsigned long t = vertex_start_index ; t < vertex_end_index ; t++) {

            // unsigned long source_vertex = device_binary_search(d_source_vector, id, vertex_size + 1);
            unsigned long source_vertex = t;

            // unsigned long index_counter = id - d_source_vector[source_vertex];


            unsigned long start_index = d_csr_offset[source_vertex];
            unsigned long end_index = d_csr_offset[source_vertex + 1];
            unsigned long source_degree_batch = end_index - start_index;

            // if(start_index < end_index) {


            //     unsigned long index_counter = id - d_prefix_sum_edge_blocks[source_vertex];
            if(source_degree_batch > 0) {

                unsigned long source_degree_edge_block_count = d_prefix_sum_edge_blocks[source_vertex + 1] - d_prefix_sum_edge_blocks[source_vertex];

                // unsigned long target_vertex = d_csr_edges[d_csr_offset[source_vertex] + (index_counter % source_degree_batch)];


                for(unsigned long z = 0 ; z < source_degree_edge_block_count ; z++) {

                    unsigned long bit_string = bit_string_lookup[z];
                    struct edge_block *root = device_vertex_dictionary->edge_block_address[source_vertex];

                    root = traverse_bit_string(root, bit_string);

                    // if(root == NULL)
                    //     printf("null hit at id=%lu\n", id);
                    // else
                    // printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_degree_edge_block_count, target_vertex, (index_counter % source_degree_batch), bit_string, root);


                    //     // printf("id=%lu, source=%lu, counter=%lu, bit_string=%lu, edge_block=%p, start_index=%lu, end_index=%lu\n", id, d_source_vector[id], index_counter, bit_string, root, start_index, end_index);

                    for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

                        if(root->edge_block_entry[i].destination_vertex == 0)
                            break;
                        else {

                            for(unsigned long j = start_index ; j < end_index ; j++) {
                                // printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, current_vertex=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_degree_edge_block_count, root->edge_block_entry[i].destination_vertex, target_vertex, (index_counter % source_degree_batch), bit_string, root);

                                if((root->edge_block_entry[i].destination_vertex == d_csr_edges[j]) && (d_csr_edges[j] != INFTY)) {
                                    root->edge_block_entry[i].destination_vertex = INFTY;
                                    // device_vertex_dictionary->active_edge_count[source_vertex]--;
                                    atomicDec(&(device_vertex_dictionary->active_edge_count[source_vertex]), INFTY);
                                    // root->active_edge_count--;
                                    break;
                                }

                            }

                        }

                    }

                }
            }

            //     // if(id == 0) {

            //     //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
            //     //         printf("%lu ", d_csr_offset[i]);
            //     //     printf("\n");
            //     //     for(unsigned long i = 0 ; i < batch_size ; i++)
            //     //         printf("%lu ", d_csr_edges[i]);
            //     //     printf("\n");

            //     // }

            // }

        }

    }

}

// Edge-centric, High-Diameter
__global__ void batched_delete_kernel_EC_HD(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id == 0) {

    //     printf("CSR offset\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_csr_offset[i]);
    //     printf("\n");
    //     printf("CSR edges\n");
    //     for(unsigned long i = 0 ; i < batch_size + 1 ; i++)
    //         printf("%lu ", d_csr_edges[i]);
    //     printf("New source vector\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_source_vector[i]);
    //     printf("\n\n");

    // }

    if((id < ceil((double)batch_size))) {

        // unsigned long vertex_start_index = id * vertex_LB_factor;
        // unsigned long vertex_end_index = (id + 1) * vertex_LB_factor;

        // if(vertex_end_index > vertex_size)
        //     vertex_end_index = vertex_size;

        // for(unsigned long t = vertex_start_index ; t < vertex_end_index ; t++) {

        // unsigned long source_vertex = device_binary_search(d_source_vector, id, vertex_size + 1);
        unsigned long source_vertex = device_binary_search(d_csr_offset, id, vertex_size + 1);

        // unsigned long index_counter = id - d_source_vector[source_vertex];


        // unsigned long start_index = d_csr_offset[source_vertex];
        // unsigned long end_index = d_csr_offset[source_vertex + 1];
        // unsigned long source_degree_batch = end_index - start_index;
        unsigned long target_vertex = d_csr_edges[id];

        // if(start_index < end_index) {


        //     unsigned long index_counter = id - d_prefix_sum_edge_blocks[source_vertex];
        if(target_vertex != INFTY) {

            // probable issue
            // unsigned long source_degree_edge_block_count = d_prefix_sum_edge_blocks[source_vertex + 1] - d_prefix_sum_edge_blocks[source_vertex];
            unsigned long source_degree_edge_block_count = device_vertex_dictionary->edge_block_count[source_vertex];

            // unsigned long target_vertex = d_csr_edges[d_csr_offset[source_vertex] + (index_counter % source_degree_batch)];


            for(unsigned long z = 0 ; z < source_degree_edge_block_count ; z++) {

                unsigned long bit_string = bit_string_lookup[z];
                struct edge_block *root = device_vertex_dictionary->edge_block_address[source_vertex];

                root = traverse_bit_string(root, bit_string);

                if((root == NULL)) {
                    printf("null hit at id=%lu, source_vertex=%lu, target_vertex=%lu, edge_block_count=%lu, counter=%lu, at GV=%lu\n", id, source_vertex, target_vertex, source_degree_edge_block_count, z, device_vertex_dictionary->edge_block_count[source_vertex]);
                    // return;
                }
                // else
                // printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_degree_edge_block_count, target_vertex, (index_counter % source_degree_batch), bit_string, root);


                //     // printf("id=%lu, source=%lu, counter=%lu, bit_string=%lu, edge_block=%p, start_index=%lu, end_index=%lu\n", id, d_source_vector[id], index_counter, bit_string, root, start_index, end_index);

                for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

                    if(root->edge_block_entry[i].destination_vertex == 0)
                        break;
                    else {

                        // for(unsigned long j = start_index ; j < end_index ; j++) {
                        // printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, current_vertex=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_degree_edge_block_count, root->edge_block_entry[i].destination_vertex, target_vertex, (index_counter % source_degree_batch), bit_string, root);

                        if((root->edge_block_entry[i].destination_vertex == target_vertex)) {
                            root->edge_block_entry[i].destination_vertex = INFTY;
                            // device_vertex_dictionary->active_edge_count[source_vertex]--;
                            atomicDec(&(device_vertex_dictionary->active_edge_count[source_vertex]), INFTY);
                            // root->active_edge_count--;
                            break;
                        }

                        // }

                    }

                }

            }
        }

        //     // if(id == 0) {

        //     //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
        //     //         printf("%lu ", d_csr_offset[i]);
        //     //     printf("\n");
        //     //     for(unsigned long i = 0 ; i < batch_size ; i++)
        //     //         printf("%lu ", d_csr_edges[i]);
        //     //     printf("\n");

        //     // }

        // }

        // }

    }

}

// Edge-centric, High-Diameter, used during vertex deletes
__global__ void batched_delete_kernel_EC_HD_vertex(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long vertex_batch_size) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id == 0) {

    //     printf("CSR offset\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_csr_offset[i]);
    //     printf("\n");
    //     printf("CSR edges\n");
    //     for(unsigned long i = 0 ; i < batch_size + 1 ; i++)
    //         printf("%lu ", d_csr_edges[i]);
    //     printf("New source vector\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_source_vector[i]);
    //     printf("\n\n");

    // }

    if((id < ceil((double)vertex_batch_size * vertex_size))) {

        // unsigned long vertex_start_index = id * vertex_LB_factor;
        // unsigned long vertex_end_index = (id + 1) * vertex_LB_factor;

        // if(vertex_end_index > vertex_size)
        //     vertex_end_index = vertex_size;

        // for(unsigned long t = vertex_start_index ; t < vertex_end_index ; t++) {

        // unsigned long source_vertex = device_binary_search(d_source_vector, id, vertex_size + 1);
        // unsigned long source_vertex = device_binary_search(d_csr_offset, id, vertex_size + 1);
        // unsigned long source_vertex = id / vertex_size;
        unsigned long source_vertex = id / vertex_batch_size;

        // unsigned long index_counter = id - d_source_vector[source_vertex];


        // unsigned long start_index = d_csr_offset[source_vertex];
        // unsigned long end_index = d_csr_offset[source_vertex + 1];
        // unsigned long source_degree_batch = end_index - start_index;

        // if((id > (source_vertex * vertex_size)) && ((id - (source_vertex * vertex_size)) >  vertex_batch_size))
        //     printf("ID=%lu, out of bounds\n");
        // else if ((id > (source_vertex * vertex_size)))
        //     printf("ID=%lu, under bounds\n");

        // unsigned long target_vertex = d_csr_edges[id - (source_vertex * vertex_size)];
        unsigned long target_vertex = d_csr_edges[id % vertex_batch_size];

        // printf("ID=%lu, source=%lu, target=%lu\n", id, source_vertex, target_vertex);
        // if(start_index < end_index) {


        //     unsigned long index_counter = id - d_prefix_sum_edge_blocks[source_vertex];
        if(target_vertex != INFTY) {

            // probable issue
            // unsigned long source_degree_edge_block_count = d_prefix_sum_edge_blocks[source_vertex + 1] - d_prefix_sum_edge_blocks[source_vertex];
            unsigned long source_degree_edge_block_count = device_vertex_dictionary->edge_block_count[source_vertex];

            // unsigned long target_vertex = d_csr_edges[d_csr_offset[source_vertex] + (index_counter % source_degree_batch)];


            for(unsigned long z = 0 ; z < source_degree_edge_block_count ; z++) {

                unsigned long bit_string = bit_string_lookup[z];
                struct edge_block *root = device_vertex_dictionary->edge_block_address[source_vertex];

                root = traverse_bit_string(root, bit_string);

                if((root == NULL)) {
                    printf("null hit at id=%lu, source_vertex=%lu, target_vertex=%lu, edge_block_count=%lu, counter=%lu, at GV=%lu\n", id, source_vertex, target_vertex, source_degree_edge_block_count, z, device_vertex_dictionary->edge_block_count[source_vertex]);
                    // return;
                }
                // else
                // printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_degree_edge_block_count, target_vertex, (index_counter % source_degree_batch), bit_string, root);


                //     // printf("id=%lu, source=%lu, counter=%lu, bit_string=%lu, edge_block=%p, start_index=%lu, end_index=%lu\n", id, d_source_vector[id], index_counter, bit_string, root, start_index, end_index);

                for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

                    if(root->edge_block_entry[i].destination_vertex == 0)
                        break;
                    else {

                        // for(unsigned long j = start_index ; j < end_index ; j++) {
                        // printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, current_vertex=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_degree_edge_block_count, root->edge_block_entry[i].destination_vertex, target_vertex, (index_counter % source_degree_batch), bit_string, root);

                        if((root->edge_block_entry[i].destination_vertex == target_vertex)) {
                            root->edge_block_entry[i].destination_vertex = INFTY;
                            // device_vertex_dictionary->active_edge_count[source_vertex]--;
                            // atomicDec could lead to incorrect results if there are batch duplicates
                            atomicDec(&(device_vertex_dictionary->active_edge_count[source_vertex]), INFTY);
                            // root->active_edge_count--;
                            break;
                        }

                        // }

                    }

                }

            }
        }

        //     // if(id == 0) {

        //     //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
        //     //         printf("%lu ", d_csr_offset[i]);
        //     //     printf("\n");
        //     //     for(unsigned long i = 0 ; i < batch_size ; i++)
        //     //         printf("%lu ", d_csr_edges[i]);
        //     //     printf("\n");

        //     // }

        // }

        // }

    }

}

// Edge-centric, Low-Diameter
__global__ void batched_delete_kernel_EC_LD(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id == 0) {

    //     printf("At actual kernel\n");
    //     printf("CSR offset\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_csr_offset[i]);
    //     printf("\n");
    //     printf("CSR edges\n");
    //     for(unsigned long i = 0 ; i < batch_size + 1 ; i++)
    //         printf("%lu ", d_csr_edges[i]);
    //     printf("\nNew source vector\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_source_vector[i]);
    //     printf("\n\n");

    // }

    if((id < ceil((double)d_source_vector[vertex_size]))) {

        // unsigned long vertex_start_index = id * vertex_LB_factor;
        // unsigned long vertex_end_index = (id + 1) * vertex_LB_factor;

        // if(vertex_end_index > vertex_size)
        //     vertex_end_index = vertex_size;

        // for(unsigned long t = vertex_start_index ; t < vertex_end_index ; t++) {

        unsigned long source_vertex = device_binary_search(d_source_vector, id, vertex_size + 1);
        // unsigned long source_vertex = device_binary_search(d_csr_offset, id, vertex_size + 1);

        // unsigned long index_counter = id - d_source_vector[source_vertex];


        // unsigned long start_index = d_csr_offset[source_vertex];
        // unsigned long end_index = d_csr_offset[source_vertex + 1];
        // unsigned long source_degree_batch = end_index - start_index;

        unsigned long input_batch_degree = d_csr_offset[source_vertex + 1] - d_csr_offset[source_vertex];
        unsigned long index_counter = id - d_source_vector[source_vertex];
        // unsigned long thread_count_source = d_source_vector[source_vertex + 1] - d_source_vector[source_vertex];
        unsigned long edge_block_index = index_counter / input_batch_degree;
        unsigned long target_vertex = d_csr_edges[d_csr_offset[source_vertex] + (index_counter % input_batch_degree)];

        // if(start_index < end_index) {

        // printf("ID=%lu, source_vertex=%lu, index_counter=%lu, edge_block_index=%lu, target_vertex=%lu\n", id, source_vertex, index_counter, edge_block_index, target_vertex);

        //     unsigned long index_counter = id - d_prefix_sum_edge_blocks[source_vertex];
        if(target_vertex != INFTY) {

            // unsigned long source_degree_edge_block_count = d_prefix_sum_edge_blocks[source_vertex + 1] - d_prefix_sum_edge_blocks[source_vertex];

            // unsigned long target_vertex = d_csr_edges[d_csr_offset[source_vertex] + (index_counter % source_degree_batch)];


            // for(unsigned long z = 0 ; z < source_degree_edge_block_count ; z++) {

            unsigned long bit_string = bit_string_lookup[edge_block_index];
            struct edge_block *root = device_vertex_dictionary->edge_block_address[source_vertex];

            root = traverse_bit_string(root, bit_string);

            // if(root == NULL)
            //     printf("null hit at id=%lu\n", id);
            // else
            // printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_degree_edge_block_count, target_vertex, (index_counter % source_degree_batch), bit_string, root);


            //     // printf("id=%lu, source=%lu, counter=%lu, bit_string=%lu, edge_block=%p, start_index=%lu, end_index=%lu\n", id, d_source_vector[id], index_counter, bit_string, root, start_index, end_index);

            for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

                if(root->edge_block_entry[i].destination_vertex == 0)
                    break;
                else {

                    // for(unsigned long j = start_index ; j < end_index ; j++) {
                    // printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, current_vertex=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_degree_edge_block_count, root->edge_block_entry[i].destination_vertex, target_vertex, (index_counter % source_degree_batch), bit_string, root);

                    if((root->edge_block_entry[i].destination_vertex == target_vertex)) {
                        root->edge_block_entry[i].destination_vertex = INFTY;
                        // device_vertex_dictionary->active_edge_count[source_vertex]--;
                        atomicDec(&(device_vertex_dictionary->active_edge_count[source_vertex]), INFTY);
                        // root->active_edge_count--;
                        break;
                    }

                    // }

                }

            }

            // }
        }

        //     // if(id == 0) {

        //     //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
        //     //         printf("%lu ", d_csr_offset[i]);
        //     //     printf("\n");
        //     //     for(unsigned long i = 0 ; i < batch_size ; i++)
        //     //         printf("%lu ", d_csr_edges[i]);
        //     //     printf("\n");

        //     // }

        // }

        // }

    }

}

// Edge-centric, Low-Diameter, Load-Balanced
__global__ void batched_delete_kernel_EC_LD_LB(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long thread_multiplier) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id == 0) {

    //     printf("At actual kernel\n");
    //     printf("CSR offset\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_csr_offset[i]);
    //     printf("\n");
    //     printf("CSR edges\n");
    //     for(unsigned long i = 0 ; i < batch_size + 1 ; i++)
    //         printf("%lu ", d_csr_edges[i]);
    //     printf("\nNew source vector\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_source_vector[i]);
    //     printf("\n\n");

    // }

    if((id < thread_multiplier * ((double)d_source_vector[vertex_size]))) {

        // unsigned long vertex_start_index = id * vertex_LB_factor;
        // unsigned long vertex_end_index = (id + 1) * vertex_LB_factor;

        // if(vertex_end_index > vertex_size)
        //     vertex_end_index = vertex_size;

        // for(unsigned long t = vertex_start_index ; t < vertex_end_index ; t++) {

        unsigned long source_vertex = device_binary_search(d_source_vector, id / thread_multiplier, vertex_size + 1);
        // unsigned long source_vertex = device_binary_search(d_csr_offset, id, vertex_size + 1);

        // unsigned long index_counter = id - d_source_vector[source_vertex];


        // unsigned long start_index = d_csr_offset[source_vertex];
        // unsigned long end_index = d_csr_offset[source_vertex + 1];
        // unsigned long source_degree_batch = end_index - start_index;

        unsigned long input_batch_degree = d_csr_offset[source_vertex + 1] - d_csr_offset[source_vertex];
        unsigned long index_counter = (id / thread_multiplier) - d_source_vector[source_vertex];
        // unsigned long thread_count_source = d_source_vector[source_vertex + 1] - d_source_vector[source_vertex];
        unsigned long edge_block_index = index_counter / input_batch_degree;
        unsigned long target_vertex = d_csr_edges[d_csr_offset[source_vertex] + (index_counter % input_batch_degree)];

        unsigned long edge_entry_index = ((id) % thread_multiplier);
        unsigned long start_index = edge_entry_index * 10;
        unsigned long end_index = start_index + 10;

        // if(start_index < end_index) {

        // printf("ID=%lu, source_vertex=%lu, index_counter=%lu, edge_block_index=%lu, edge_entry_index=%lu, target_vertex=%lu, start_index=%lu, end_index=%lu\n", id, source_vertex, index_counter, edge_block_index, edge_entry_index, target_vertex, start_index, end_index);

        //     unsigned long index_counter = id - d_prefix_sum_edge_blocks[source_vertex];
        if(target_vertex != INFTY) {

            // unsigned long source_degree_edge_block_count = d_prefix_sum_edge_blocks[source_vertex + 1] - d_prefix_sum_edge_blocks[source_vertex];

            // unsigned long target_vertex = d_csr_edges[d_csr_offset[source_vertex] + (index_counter % source_degree_batch)];


            // for(unsigned long z = 0 ; z < source_degree_edge_block_count ; z++) {

            unsigned long bit_string = bit_string_lookup[edge_block_index];
            struct edge_block *root = device_vertex_dictionary->edge_block_address[source_vertex];

            root = traverse_bit_string(root, bit_string);

            // if(root == NULL)
            //     printf("null hit at id=%lu\n", id);
            // else
            // printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_degree_edge_block_count, target_vertex, (index_counter % source_degree_batch), bit_string, root);


            //     // printf("id=%lu, source=%lu, counter=%lu, bit_string=%lu, edge_block=%p, start_index=%lu, end_index=%lu\n", id, d_source_vector[id], index_counter, bit_string, root, start_index, end_index);

            // for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {
            for(unsigned long i = start_index ; i < end_index ; i++) {

                if(root->edge_block_entry[i].destination_vertex == 0)
                    break;
                else {

                    // for(unsigned long j = start_index ; j < end_index ; j++) {
                    // printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, current_vertex=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_degree_edge_block_count, root->edge_block_entry[i].destination_vertex, target_vertex, (index_counter % source_degree_batch), bit_string, root);

                    if((root->edge_block_entry[i].destination_vertex == target_vertex)) {
                        root->edge_block_entry[i].destination_vertex = INFTY;
                        // device_vertex_dictionary->active_edge_count[source_vertex]--;
                        atomicDec(&(device_vertex_dictionary->active_edge_count[source_vertex]), INFTY);
                        // root->active_edge_count--;
                        break;
                    }

                    // }

                }

            }

            // }
        }

        //     // if(id == 0) {

        //     //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
        //     //         printf("%lu ", d_csr_offset[i]);
        //     //     printf("\n");
        //     //     for(unsigned long i = 0 ; i < batch_size ; i++)
        //     //         printf("%lu ", d_csr_edges[i]);
        //     //     printf("\n");

        //     // }

        // }

        // }

    }

}

__global__ void batched_delete_kernel_LB(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long thread_divider) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id == 0) {

    //     printf("CSR offset\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_csr_offset[i]);
    //     printf("\n");
    //     printf("CSR edges\n");
    //     for(unsigned long i = 0 ; i < batch_size ; i++)
    //         printf("%lu ", d_csr_edges[i]);
    //     printf("\nNew source vector\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_source_vector[i]);
    //     printf("\n\n");

    // }

    if(id < ceil(double((d_source_vector[vertex_size])) / thread_divider)) {

        // unsigned long source_vertex = device_binary_search(d_source_vector, id, vertex_size + 1);
        // unsigned long source_vertex = id;

        // unsigned long index_counter = id - d_source_vector[source_vertex];

        unsigned long thread_start_index = id * thread_divider;
        unsigned long thread_end_index = (id + 1) * thread_divider;
        if(thread_end_index > d_source_vector[vertex_size])
            thread_end_index = d_source_vector[vertex_size];
        // unsigned long work_done = 0;
        // if(id == ceil(double((d_source_vector[vertex_size])) / thread_divider)) {


        // }

        unsigned long i = thread_start_index;
        while(i < thread_end_index) {

            unsigned long source_vertex = device_binary_search(d_source_vector, i, vertex_size + 1);
            unsigned long source_end_index = d_source_vector[source_vertex + 1];

            unsigned long start_index = d_csr_offset[source_vertex];
            unsigned long end_index = d_csr_offset[source_vertex + 1];
            unsigned long source_degree_batch = end_index - start_index;

            // printf("ID=%lu, thread_start_index=%lu, thread_end_index=%lu, source_vertex=%lu, start_index=%lu, end_index=%lu, i=%lu\n", id, thread_start_index, thread_end_index, source_vertex, start_index, end_index, i);
            // printf("ID=%lu, source_vertex=%lu, thread_start_index=%lu, thread_end_index=%lu\n", id, source_vertex, thread_start_index, thread_end_index);
            // break;

            if(source_degree_batch) {

                unsigned long source_edge_block_count = d_prefix_sum_edge_blocks[source_vertex + 1] - d_prefix_sum_edge_blocks[source_vertex];
                unsigned long index_counter = i - d_source_vector[source_vertex];
                unsigned long source_edge_block_counter = (index_counter / source_degree_batch) / EDGE_BLOCK_SIZE;
                unsigned long source_edge_entry_counter = (index_counter / source_degree_batch) % EDGE_BLOCK_SIZE;
                // unsigned long source_batch_entry_counter = ((unsigned long)(ceil(((double)index_counter) / EDGE_BLOCK_SIZE))) % source_degree_batch;
                unsigned long source_batch_entry_counter = (index_counter - (source_edge_block_counter * EDGE_BLOCK_SIZE * source_degree_batch)) % source_degree_batch;
                // unsigned long source_batch_entry_counter = index_counter % source_degree_batch;

                if(index_counter)
                    start_index += source_batch_entry_counter;

                // printf("ID=%lu, thread_start_index=%lu, thread_end_index=%lu, source_vertex=%lu, start_index=%lu, end_index=%lu, index_counter=%lu, source_edge_block_counter=%lu, source_edge_entry_counter=%lu, source_batch_entry_counter=%lu\n", id, thread_start_index, thread_end_index, source_vertex, start_index, end_index, index_counter, source_edge_block_counter, source_edge_entry_counter, source_batch_entry_counter);

                // unsigned long temp = ((d_source_vector[source_vertex + 1] - d_source_vector[source_vertex]) - index_counter);
                // unsigned long temp1 = temp;

                // if(temp1 / 10) {
                //     i += 10;
                //     printf("ID=%lu, incremented by %lu\n", id, 10);
                // }
                // else {
                //     i += temp % 10;
                //     printf("ID=%lu, incremented by %lu\n", id, temp % 10);
                // }



                // if(temp % 10) {
                //     i += temp % 10;
                //     printf("ID=%lu, incremented by %lu\n", id, temp % 10);
                // }
                // else {
                //     i += 10;
                //     printf("ID=%lu, incremented by %lu\n", id, 10);
                // }
                for(unsigned long j = source_edge_block_counter ; (j < source_edge_block_count) && (i < thread_end_index) ; j++) {

                    unsigned long bit_string = bit_string_lookup[j];
                    struct edge_block *root = device_vertex_dictionary->edge_block_address[source_vertex];

                    root = traverse_bit_string(root, bit_string);

                    // printf("ID=%lu, hit\n");


                    // if(root == NULL)
                    //     printf("null hit at id=%lu\n", id);
                    // else
                    // printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_degree_edge_block_count, target_vertex, (index_counter % source_degree_batch), bit_string, root);


                    // printf("id=%lu, source=%lu, counter=%lu, bit_string=%lu, edge_block=%p, start_index=%lu, end_index=%lu\n", id, d_source_vector[id], index_counter, bit_string, root, start_index, end_index);

                    // printf("ID=%lu, thread_start_index=%lu, thread_end_index=%lu, source_vertex=%lu, start_index=%lu, end_index=%lu, index_counter=%lu, root=%p, i=%lu\n", id, thread_start_index, thread_end_index, source_vertex, start_index, end_index, index_counter, root, i);

                    // __syncthreads();

                    for(unsigned long k = source_edge_entry_counter ; (k < EDGE_BLOCK_SIZE) && (i < thread_end_index) ; k++) {


                        if( (root->edge_block_entry[k].destination_vertex == 0)) {
                            i += source_degree_batch;
                            break;
                            // goto thread_end_VC_LB;
                        }
                        else if (root->edge_block_entry[k].destination_vertex == INFTY) {
                            i += source_degree_batch;
                            continue;
                        }
                        else {

                            for(unsigned long l = start_index ; (l < end_index) && (i < thread_end_index) ; l++) {

                                i++;
                                // if(id == 10)
                                // printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, current_vertex=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_edge_block_count, root->edge_block_entry[i].destination_vertex, root->edge_block_entry[i].destination_vertex, (index_counter % source_degree_batch), bit_string, root);

                                if((root->edge_block_entry[k].destination_vertex == d_csr_edges[l])) {

                                    // if((source_vertex == 3))
                                    //     printf("Hit Problem\n");

                                    root->edge_block_entry[k].destination_vertex = INFTY;
                                    // device_vertex_dictionary->active_edge_count[source_vertex]--;
                                    atomicDec(&(device_vertex_dictionary->active_edge_count[source_vertex]), INFTY);
                                    // root->active_edge_count--;
                                    // i += end_index - l;
                                    // break;
                                }

                            }

                            start_index = d_csr_offset[source_vertex];

                        }

                        // if(i > thread_divider)
                        //     goto thread_end_VC_LB;


                    }

                }

                // problem here
                // unsigned long temp = ((d_source_vector[source_vertex + 1] - d_source_vector[source_vertex]) - index_counter);
                // unsigned long temp = ((d_source_vector[source_vertex + 1] - d_source_vector[source_vertex]) - index_counter);
                // unsigned long temp1 = temp;

                // if(temp1 / 10) {
                //     i += 10;
                //     printf("ID=%lu, incremented by %lu\n", id, 10);
                // }
                // else {
                //     i += temp % 10;
                //     printf("ID=%lu, incremented by %lu\n", id, temp % 10);
                // }

            }

            // i++;

        }


        // for(unsigned long i = thread_start_index ; i < thread_end_index ; i++) {

        //     unsigned long source_vertex = device_binary_search(d_source_vector, i, vertex_size + 1);
        //     unsigned long source_end_index = d_source_vector[source_vertex + 1];

        //     unsigned long start_index = d_csr_offset[source_vertex];
        //     unsigned long end_index = d_csr_offset[source_vertex + 1];
        //     unsigned long source_degree_batch = end_index - start_index;


        //     if(source_degree_batch > 0) {

        //         // if(end_index > thread_end_index)
        //         //     end_index = thread_end_index;

        //         unsigned long source_degree_edge_block_count = d_prefix_sum_edge_blocks[source_vertex + 1] - d_prefix_sum_edge_blocks[source_vertex];
        //         unsigned long index_counter = thread_start_index - d_source_vector[source_vertex];
        //         unsigned long source_edge_block_counter = index_counter / EDGE_BLOCK_SIZE;
        //         unsigned long source_edge_entry_counter = index_counter % EDGE_BLOCK_SIZE;
        //         // unsigned long target_vertex = d_csr_edges[d_csr_offset[source_vertex] + (index_counter % source_degree_batch)];

        //         // printf("ID=%lu hit, source_vertex=%lu, thread_start_index=%lu, thread_end_index=%lu\n", id, source_vertex, thread_start_index, thread_end_index);
        //         printf("ID=%lu, thread_start_index=%lu, thread_end_index=%lu, source_vertex=%lu, start_index=%lu, end_index=%lu, index_counter=%lu\n", id, thread_start_index, thread_end_index, source_vertex, start_index, end_index, index_counter);


        //         for(unsigned long z = source_edge_block_counter ; z < source_degree_edge_block_count ; z++) {

        //             unsigned long bit_string = bit_string_lookup[z];
        //             struct edge_block *root = device_vertex_dictionary->edge_block_address[source_vertex];

        //             root = traverse_bit_string(root, bit_string);

        //         // if(root == NULL)
        //         //     printf("null hit at id=%lu\n", id);
        //         // else
        //             // printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_degree_edge_block_count, target_vertex, (index_counter % source_degree_batch), bit_string, root);


        //             // printf("id=%lu, source=%lu, counter=%lu, bit_string=%lu, edge_block=%p, start_index=%lu, end_index=%lu\n", id, d_source_vector[id], index_counter, bit_string, root, start_index, end_index);

        //             // printf("ID=%lu, thread_start_index=%lu, thread_end_index=%lu, source_vertex=%lu, start_index=%lu, end_index=%lu, index_counter=%lu, root=%p\n", id, thread_start_index, thread_end_index, source_vertex, start_index, end_index, index_counter, root);

        //             for(unsigned long k = source_edge_entry_counter ; k < EDGE_BLOCK_SIZE, i < thread_end_index ; k++) {



        //                 if( (root->edge_block_entry[i].destination_vertex == 0))
        //                     break;
        //                 else {

        //                     for(unsigned long j = start_index ; j < end_index, i < thread_end_index ; j++, i++) {

        //                         // if(!id)
        //                         //     printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, current_vertex=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_degree_edge_block_count, root->edge_block_entry[i].destination_vertex, root->edge_block_entry[i].destination_vertex, (index_counter % source_degree_batch), bit_string, root);

        //                         if(root->edge_block_entry[i].destination_vertex == d_csr_edges[j]) {
        //                             root->edge_block_entry[i].destination_vertex = INFTY;
        //                             // device_vertex_dictionary->active_edge_count[source_vertex]--;
        //                             atomicDec(&(device_vertex_dictionary->active_edge_count[source_vertex]), INFTY);
        //                             // root->active_edge_count--;
        //                             break;
        //                         }

        //                     }

        //                 }

        //                 // if(i > thread_divider)
        //                 //     goto thread_end_VC_LB;


        //             }

        //         }
        //     }

        thread_end_VC_LB:

        // }

        //     // if(id == 0) {

        //     //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
        //     //         printf("%lu ", d_csr_offset[i]);
        //     //     printf("\n");
        //     //     for(unsigned long i = 0 ; i < batch_size ; i++)
        //     //         printf("%lu ", d_csr_edges[i]);
        //     //     printf("\n");

        //     // }

        // }

    }

}

__global__ void batched_delete_kernel_edge_block_centric_v4(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id == 0) {

    //     printf("CSR offset\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_csr_offset[i]);
    //     printf("\n");
    //     printf("CSR edges\n");
    //     for(unsigned long i = 0 ; i < batch_size + 1 ; i++)
    //         printf("%lu ", d_csr_edges[i]);
    //     printf("New source vector\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_source_vector[i]);
    //     printf("\n\n");

    // }

    if((id < d_source_vector[vertex_size])) {

        unsigned long source_vertex = device_binary_search(d_source_vector, id, vertex_size + 1);

        unsigned long index_counter = id - d_source_vector[source_vertex];

        unsigned long source_degree_batch = d_csr_offset[source_vertex + 1] - d_csr_offset[source_vertex];

        // unsigned long start_index = d_csr_offset[source_vertex];
        // unsigned long end_index = d_csr_offset[source_vertex + 1];

        // if(start_index < end_index) {


        //     unsigned long index_counter = id - d_prefix_sum_edge_blocks[source_vertex];
        if(source_degree_batch > 0) {

            unsigned long source_degree_edge_block_count = d_prefix_sum_edge_blocks[source_vertex + 1] - d_prefix_sum_edge_blocks[source_vertex];

            unsigned long target_vertex = d_csr_edges[d_csr_offset[source_vertex] + (index_counter % source_degree_batch)];


            unsigned long bit_string = bit_string_lookup[index_counter / source_degree_batch];
            struct edge_block *root = device_vertex_dictionary->edge_block_address[source_vertex];

            root = traverse_bit_string(root, bit_string);

            // if(root == NULL)
            //     printf("null hit at id=%lu\n", id);
            // else
            // printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_degree_edge_block_count, target_vertex, (index_counter % source_degree_batch), bit_string, root);


            //     // printf("id=%lu, source=%lu, counter=%lu, bit_string=%lu, edge_block=%p, start_index=%lu, end_index=%lu\n", id, d_source_vector[id], index_counter, bit_string, root, start_index, end_index);

            for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

                if(root->edge_block_entry[i].destination_vertex == 0)
                    break;
                else {

                    // for(unsigned long j = start_index ; j < end_index ; j++) {
                    // printf("ID is %lu, source_vertex=%lu, index_counter=%lu, source_degree_batch=%lu, source_degree_edge_block_count=%lu, current_vertex=%lu, target_vertex=%lu, target_vertex_index=%lu, bit_string=%lu, root=%p\n", id, source_vertex, index_counter, source_degree_batch, source_degree_edge_block_count, root->edge_block_entry[i].destination_vertex, target_vertex, (index_counter % source_degree_batch), bit_string, root);

                    if(root->edge_block_entry[i].destination_vertex == target_vertex) {
                        root->edge_block_entry[i].destination_vertex = INFTY;
                        // device_vertex_dictionary->active_edge_count[source_vertex]--;
                        atomicDec(&(device_vertex_dictionary->active_edge_count[source_vertex]), INFTY);
                        // root->active_edge_count--;
                        break;
                    }

                    // }

                }

            }
        }

        //     // if(id == 0) {

        //     //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
        //     //         printf("%lu ", d_csr_offset[i]);
        //     //     printf("\n");
        //     //     for(unsigned long i = 0 ; i < batch_size ; i++)
        //     //         printf("%lu ", d_csr_edges[i]);
        //     //     printf("\n");

        //     // }

        // }

    }

}

__global__ void batched_delete_preprocessing_edge_block_centric_v2(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_degrees, unsigned long *d_source_vector, unsigned long *d_non_zero_vertices, unsigned long *d_index_counter, unsigned long d_non_zero_vertices_count) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < d_non_zero_vertices_count) {

        unsigned long current_source = d_non_zero_vertices[id];
        unsigned long start_index = d_prefix_sum_edge_blocks[id];
        unsigned long end_index = d_prefix_sum_edge_blocks[id + 1];
        unsigned long index = 0;

        // source vector values start from 0
        for(unsigned long i = start_index ; i < end_index ; i++) {

            d_source_vector[i] = current_source;
            d_index_counter[i] = index++;

        }

    }

    // if (id == 0) {

    //     // printf("last index is %lu\n", d_prefix_sum_edge_blocks[vertex_size]);
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //         printf("%lu ", d_prefix_sum_edge_blocks[i]);

    //     printf("\n");

    //     // printf("Size is %lu\n", d_prefix_sum_edge_blocks[vertex_size]);
    //     printf("Source vector is\n");
    //     for(unsigned long i = 0 ; i < d_prefix_sum_edge_blocks[vertex_size] ; i++)
    //         printf("%lu ", d_source_vector[i]);
    //     printf("\nIndex counter is\n");
    //     for(unsigned long i = 0 ; i < d_prefix_sum_edge_blocks[vertex_size] ; i++)
    //         printf("%lu ", d_index_counter[i]);
    // }

}

__global__ void batched_delete_preprocessing_v3_1(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_thread_count_vector) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        d_thread_count_vector[id + 1] = device_vertex_dictionary-> vertex_adjacency[id]->edge_block_count * (d_csr_offset[id + 1] - d_csr_offset[id]);

        // printf("Thread %lu requires %lu threads\n", id, d_thread_count_vector[id + 1]);

    }

}

__global__ void batched_delete_preprocessing_v3_prefix_sum(unsigned long *d_thread_count_vector, unsigned long vertex_size) {

    // printf("%lu %lu ", d_thread_count_vector[0], d_thread_count_vector[1]);

    for(unsigned long i = 2 ; i < vertex_size + 1 ; i++) {
        d_thread_count_vector[i] = d_thread_count_vector[i - 1] + d_thread_count_vector[i];
        // printf("%lu ", d_thread_count_vector[i]);
    }
    // printf("\n");

}

__global__ void batched_delete_preprocessing_v3_2(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_degrees, unsigned long *d_source_vector, unsigned long *d_thread_count_vector) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        unsigned long start_index = d_thread_count_vector[id];
        unsigned long end_index = d_thread_count_vector[id + 1];

        for(unsigned long i = start_index ; i < end_index ; i++)
            d_source_vector[i] = id;

        // printf("thread %lu, start_index %lu, end_index %lu\n", id, start_index, end_index);

    }

    // if(id == 0) {

    //     for(unsigned long i = 0 ; i < d_thread_count_vector[vertex_size] ; i++)
    //         printf("Index %lu Source is %lu\n", i, d_source_vector[i]);

    // }

}

__global__ void batched_delete_kernel_edge_block_centric_v3(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long *d_thread_count_vector) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if((id < d_thread_count_vector[vertex_size])) {

        unsigned long source_vertex = d_source_vector[id];
        unsigned long start_index = d_csr_offset[source_vertex];
        unsigned long end_index = d_csr_offset[source_vertex + 1];

        if(start_index < end_index) {


            unsigned long index_counter = id - d_thread_count_vector[source_vertex];

            unsigned long destination_vertex_index = index_counter % (end_index - start_index);
            unsigned long destination_vertex = d_csr_edges[start_index + destination_vertex_index];
            unsigned long edge_block = index_counter / (end_index - start_index);

            unsigned long bit_string = bit_string_lookup[edge_block];
            struct edge_block *root = device_vertex_dictionary->vertex_adjacency[source_vertex]->edge_block_address;

            root = traverse_bit_string(root, bit_string);

            // printf("id %lu, source %lu, index_counter %lu, edge_block %lu, edge_block_address %p, destination_vertex_index %lu, destination_vertex %lu\n", id, source_vertex, index_counter, edge_block, root, destination_vertex_index, destination_vertex);

            for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

                if(root->edge_block_entry[i].destination_vertex == 0)
                    break;
                else {

                    if(root->edge_block_entry[i].destination_vertex == destination_vertex) {
                        root->edge_block_entry[i].destination_vertex = INFTY;

                        //         for(unsigned long j = start_index ; j < end_index ; j++) {

                        //             if(root->edge_block_entry[i].destination_vertex == d_csr_edges[j]) {
                        //                 root->edge_block_entry[i].destination_vertex = INFTY;
                        //                 break;
                        //             }

                    }

                }

            }

        }

    }

}

__global__ void correctness_check_kernel(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_correctness_flag) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        // if(id == 0) {
        //     printf("Device csr offset\n");
        //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
        //         printf("%lu ", d_csr_offset[i]);
        //     printf("\n");
        // }

        unsigned long existsFlag = 0;

        // for(unsigned long i = 0 ; i < edge_size ; i++)
        //     printf("%lu and %lu\n", d_c_source[i], d_c_destination[i]);

        // printf("*-------------*\n");

        unsigned long start_index = d_csr_offset[id];
        unsigned long end_index = d_csr_offset[id + 1];
        unsigned long curr_source = id + 1;

        // printf("Source=%lu, Start index=%lu, End index=%lu\n", curr_source, start_index, end_index);

        for(unsigned long i = start_index ; i < end_index ; i++) {

            // unsigned long curr_source = d_c_source[i];
            unsigned long curr_destination = d_csr_edges[i];

            // printf("%lu and %lu\n", curr_source, curr_destination);

            existsFlag = 0;

            // here checking at curr_source - 1, since source vertex 10 would be stored at index 9
            if((device_vertex_dictionary->vertex_adjacency[curr_source - 1] != NULL) && (device_vertex_dictionary->vertex_id[curr_source - 1] != 0)) {
                // printf("%lu -> , edge blocks = %lu, edge sentinel = %p, active edge count = %lu, destination vertices -> ", device_vertex_dictionary->vertex_id[i], device_vertex_dictionary->edge_block_count[i], device_vertex_dictionary->vertex_adjacency[i], device_vertex_dictionary->vertex_adjacency[i]->active_edge_count);

                unsigned long edge_block_counter = 0;
                unsigned long edge_block_entry_count = 0;
                // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address[edge_block_counter];
                struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address + (edge_block_counter);

                for(unsigned long j = 0 ; j < device_vertex_dictionary->vertex_adjacency[curr_source - 1]->active_edge_count ; j++) {

                    // printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);

                    // edge_block_entry_count++;

                    if(itr->edge_block_entry[edge_block_entry_count].destination_vertex == curr_destination) {
                        existsFlag = 1;
                        // printf("Found %lu and %lu\n", curr_source, curr_destination);
                        break;
                    }

                    if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
                        // itr = itr->next;
                        // itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address[++edge_block_counter];
                        itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address + (++edge_block_counter);
                        edge_block_entry_count = 0;
                    }
                }

                if(!existsFlag) {
                    // printf("Issue at id=%lu destination=%lu\n", id, curr_destination);
                    break;
                }

                // printf("\n");

            }

        }

        if(!existsFlag)
            *d_correctness_flag = 1;
        else
            *d_correctness_flag = 0;

        // printf("*---------------*\n\n");

    }



}

__device__ unsigned long find_edge(struct edge_block *root, unsigned long curr_source, unsigned long curr_destination) {

    unsigned long existsFlag = 0;
    unsigned long edge_block_counter = 0;
    unsigned long edge_block_entry_count = 0;
    // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address[edge_block_counter];
    // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address + (edge_block_counter);

    for(unsigned long j = 0 ; j < root->active_edge_count ; j++) {

        // printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);

        // edge_block_entry_count++;

        if(root->edge_block_entry[edge_block_entry_count++].destination_vertex == curr_destination) {
            existsFlag = 1;
            // printf("%lu found at %lu in source %lu\n", curr_destination, edge_block_entry_count, curr_source);
            // printf("Found %lu and %lu\n", curr_source, curr_destination);
            break;
        }

        // if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
        //     // itr = itr->next;
        //     // itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address[++edge_block_counter];
        //     itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address + (++edge_block_counter);
        //     edge_block_entry_count = 0;
        // }
    }

    unsigned long existsFlag_lptr = 0;
    unsigned long existsFlag_rptr = 0;

    if(root->lptr != NULL)
        existsFlag_lptr = find_edge(root->lptr, curr_source, curr_destination);
    if(root->rptr != NULL)
        existsFlag_rptr = find_edge(root->rptr, curr_source, curr_destination);

    return (existsFlag || existsFlag_lptr || existsFlag_rptr);
}

__global__ void correctness_check_kernel_v1(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_correctness_flag) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        // if(id == 0) {
        //     printf("Device csr offset\n");
        //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
        //         printf("%lu ", d_csr_offset[i]);
        //     printf("\n");
        // }

        unsigned long existsFlag = 0;

        // for(unsigned long i = 0 ; i < edge_size ; i++)
        //     printf("%lu and %lu\n", d_c_source[i], d_c_destination[i]);

        // printf("*-------------*\n");

        unsigned long start_index = d_csr_offset[id];
        unsigned long end_index = d_csr_offset[id + 1];
        unsigned long curr_source = id + 1;

        // printf("Source=%lu, Start index=%lu, End index=%lu\n", curr_source, start_index, end_index);

        for(unsigned long i = start_index ; i < end_index ; i++) {

            // unsigned long curr_source = d_c_source[i];
            unsigned long curr_destination = d_csr_edges[i];

            // printf("%lu and %lu\n", curr_source, curr_destination);

            existsFlag = 0;

            // here checking at curr_source - 1, since source vertex 10 would be stored at index 9
            if((device_vertex_dictionary->vertex_adjacency[curr_source - 1] != NULL) && (device_vertex_dictionary->vertex_id[curr_source - 1] != 0)) {
                // printf("%lu -> , edge blocks = %lu, edge sentinel = %p, active edge count = %lu, destination vertices -> ", device_vertex_dictionary->vertex_id[i], device_vertex_dictionary->edge_block_count[i], device_vertex_dictionary->vertex_adjacency[i], device_vertex_dictionary->vertex_adjacency[i]->active_edge_count);

                existsFlag = find_edge(device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address, curr_source, curr_destination);
                // printf("%lu found in source %lu\n", curr_destination, curr_source);

                // if(!existsFlag)
                //     printf("Issue at id=%lu destination=%lu\n", id, curr_destination);

                // unsigned long edge_block_counter = 0;
                // unsigned long edge_block_entry_count = 0;
                // // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address[edge_block_counter];
                // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address + (edge_block_counter);

                // for(unsigned long j = 0 ; j < device_vertex_dictionary->vertex_adjacency[curr_source - 1]->active_edge_count ; j++) {

                //     // printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);

                //     // edge_block_entry_count++;

                //     if(itr->edge_block_entry[edge_block_entry_count].destination_vertex == curr_destination) {
                //         existsFlag = 1;
                //         // printf("Found %lu and %lu\n", curr_source, curr_destination);
                //         break;
                //     }

                //     if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
                //         // itr = itr->next;
                //         // itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address[++edge_block_counter];
                //         itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address + (++edge_block_counter);
                //         edge_block_entry_count = 0;
                //     }
                // }

                if(!existsFlag) {
                    printf("Issue at id=%lu, source=%lu and destination=%lu\n", id, curr_source, curr_destination);
                    break;
                }

                // printf("\n");

            }
            // else if(device_vertex_dictionary->vertex_adjacency[curr_source - 1]->active_edge_count == 0)
            //     existsFlag = 1;

        }

        // this means degree of that source vertex is 0
        if(start_index == end_index)
            existsFlag = 1;

        if(!existsFlag)
            *d_correctness_flag = 1;
        else
            *d_correctness_flag = 0;

        // printf("*---------------*\n\n");

    }



}

__global__ void printKernelmodded_v1(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long size) {

    printf("Printing Linked List\n");

    // struct vertex_block *ptr = d_v_d_sentinel.next;

    unsigned long vertex_block = 0;

    for(unsigned long i = 0 ; i < device_vertex_dictionary->active_vertex_count ; i++) {

        // printf("Checkpoint\n");

        if((device_vertex_dictionary->vertex_adjacency[i] != NULL) && (device_vertex_dictionary->vertex_id[i] != 0)) {
            printf("%lu -> , edge blocks = %lu, edge sentinel = %p, active edge count = %lu, destination vertices -> ", device_vertex_dictionary->vertex_id[i], device_vertex_dictionary->edge_block_count[i], device_vertex_dictionary->vertex_adjacency[i], device_vertex_dictionary->vertex_adjacency[i]->active_edge_count);

            unsigned long edge_block_counter = 0;
            unsigned long edge_block_entry_count = 0;
            // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[edge_block_counter];
            struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address;

            for(unsigned long j = 0 ; j < device_vertex_dictionary->vertex_adjacency[i]->active_edge_count ; j++) {

                printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);

                // edge_block_entry_count++;

                if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
                    // itr = itr->next;
                    // itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[++edge_block_counter];
                    itr = itr + 1;
                    edge_block_entry_count = 0;
                }
            }
            printf("\n");

        }



    }


    printf("VDS = %lu\n", d_v_d_sentinel.vertex_count);
    printf("K2 counter = %u\n", k2counter);

}

__device__ void preorderTraversal(struct edge_block *root) {

    if(root == NULL)
        return;

    else {

        printf("\nedge block edge count = %lu, %p, level_order_predecessor = %p, ", root->active_edge_count, root, root->level_order_predecessor);

        // for(unsigned long j = 0 ; j < root->active_edge_count ; j++) {
        for(unsigned long j = 0 ; j < EDGE_BLOCK_SIZE ; j++) {

            printf("%lu ", root->edge_block_entry[j].destination_vertex);

            // edge_block_entry_count++;

            // if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
            //     // itr = itr->next;
            //     // itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[++edge_block_counter];
            //     itr = itr + 1;
            //     edge_block_entry_count = 0;
            // }
        }

        preorderTraversal(root->lptr);



        // printf("\n");

        preorderTraversal(root->rptr);

    }

    // unsigned long edge_block_counter = 0;
    // unsigned long edge_block_entry_count = 0;
    // // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[edge_block_counter];
    // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address;


    // for(unsigned long j = 0 ; j < device_vertex_dictionary->vertex_adjacency[i]->active_edge_count ; j++) {

    //     printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);

    //     // edge_block_entry_count++;

    //     if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
    //         // itr = itr->next;
    //         // itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[++edge_block_counter];
    //         itr = itr + 1;
    //         edge_block_entry_count = 0;
    //     }
    // }

}

__global__ void printKernelmodded_v2(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long size) {

    printf("Printing Linked List\n");
    printf("Active Vertex Count = %lu\n", device_vertex_dictionary->active_vertex_count);

    // struct vertex_block *ptr = d_v_d_sentinel.next;

    unsigned long vertex_block = 0;

    // printf("%lu hidden value\n", device_vertex_dictionary->vertex_adjacency[4]->edge_block_address->active_edge_count);

    for(unsigned long i = 0 ; i < device_vertex_dictionary->active_vertex_count ; i++) {

        // printf("Checkpoint\n");
        printf("%lu -> , edge blocks = %lu, root = %p, active edge count = %u, last insert edge block = %p, last insert edge offset = %lu, destination vertices -> ", device_vertex_dictionary->vertex_id[i], device_vertex_dictionary->edge_block_count[i], device_vertex_dictionary->edge_block_address[i], device_vertex_dictionary->active_edge_count[i], device_vertex_dictionary->last_insert_edge_block[i], device_vertex_dictionary->last_insert_edge_offset[i]);

        if((device_vertex_dictionary->edge_block_address[i] != NULL) && (device_vertex_dictionary->vertex_id[i] != 0)) {
            // printf("%lu -> , edge blocks = %lu, root = %p, active edge count = %u, last insert edge block = %p, last insert edge offset = %lu, destination vertices -> ", device_vertex_dictionary->vertex_id[i], device_vertex_dictionary->edge_block_count[i], device_vertex_dictionary->edge_block_address[i], device_vertex_dictionary->active_edge_count[i], device_vertex_dictionary->last_insert_edge_block[i], device_vertex_dictionary->last_insert_edge_offset[i]);

            preorderTraversal(device_vertex_dictionary->edge_block_address[i]);

            // unsigned long edge_block_counter = 0;
            // unsigned long edge_block_entry_count = 0;
            // // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[edge_block_counter];
            // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address;


            // for(unsigned long j = 0 ; j < device_vertex_dictionary->vertex_adjacency[i]->active_edge_count ; j++) {

            //     printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);

            //     // edge_block_entry_count++;

            //     if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
            //         // itr = itr->next;
            //         // itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[++edge_block_counter];
            //         itr = itr + 1;
            //         edge_block_entry_count = 0;
            //     }
            // }
            // printf("\n");

        }
        printf("\n");

        // else {

        //     printf("Hit\n");

        // }



    }


    printf("VDS = %lu\n", d_v_d_sentinel.vertex_count);
    printf("K2 counter = %u\n", k2counter);

}

__global__ void cbt_stats(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long size) {

    printf("Printing Bucket Stats\n");
    unsigned long tree_height[100];
    unsigned long max_height = 0;

    for(unsigned long i = 0 ; i < device_vertex_dictionary->active_vertex_count ; i++) {

        if((device_vertex_dictionary->vertex_adjacency[i] != NULL) && (device_vertex_dictionary->vertex_id[i] != 0)) {
            // printf("%lu -> , edge blocks = %lu, edge sentinel = %p, root = %p, active edge count = %lu, destination vertices -> ", device_vertex_dictionary->vertex_id[i], device_vertex_dictionary->edge_block_count[i], device_vertex_dictionary->vertex_adjacency[i], device_vertex_dictionary->vertex_adjacency[i]->edge_block_address, device_vertex_dictionary->vertex_adjacency[i]->active_edge_count);

            unsigned long height = floor(log2(ceil(double(device_vertex_dictionary->vertex_adjacency[i]->active_edge_count) / EDGE_BLOCK_SIZE)));
            tree_height[height]++;
            if(height > max_height)
                max_height = height;
            // inorderTraversal(device_vertex_dictionary->vertex_adjacency[i]->edge_block_address);
            // printf("\n");

        }

    }

    for(unsigned long i = 0 ; i <= max_height ; i++) {
        printf("Height %lu has %lu vertices\n", i, tree_height[i]);
        tree_height[i] = 0;
    }


}

__device__ void preorderTraversal_PageRank(struct vertex_dictionary_structure *device_vertex_dictionary, struct edge_block *root, float *d_pageRankVector_1, float *d_pageRankVector_2, unsigned long id, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, float page_factor) {

    if(root == NULL)
        return;

    else {

        // printf("\nedge block edge count = %lu, %p, ", root->active_edge_count, root);

        // unsigned long page_factor = d_pageRankVector_1[root->edge_block_entry[0].destination_vertex - 1] / device_vertex_dictionary->vertex_adjacency[root->edge_block_entry[0].destination_vertex - 1]->active_edge_count;

        // for(unsigned long i = 0 ; i < root->active_edge_count ; i++) {

        //     // printf("%lu ", root->edge_block_entry[i].destination_vertex);



        //     // d_pageRankVector_2[id] += d_pageRankVector_1[root->edge_block_entry[i].destination_vertex - 1] / device_vertex_dictionary->vertex_adjacency[root->edge_block_entry[i].destination_vertex - 1]->active_edge_count;
        //     // d_pageRankVector_2[id] += d_pageRankVector_1[root->edge_block_entry[i].destination_vertex - 1] / d_source_degrees[root->edge_block_entry[i].destination_vertex - 1];
        //     // d_pageRankVector_2[id] += page_factor;

        //     atomicAdd(&d_pageRankVector_2[root->edge_block_entry[i].destination_vertex - 1], page_factor);

        //     // d_pageRankVector_2[root->edge_block_entry[i].destination_vertex - 1] += page_factor;

        // }

        for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {
            // for(unsigned long i = 0 ; i < root->active_edge_count ; i++) {

            if(root->edge_block_entry[i].destination_vertex == 0)
                break;
            else {

                // for(unsigned long j = start_index ; j < end_index ; j++) {

                // if((root == NULL) || ((root->edge_block_entry[i].destination_vertex - 1) >= (vertex_size)) || ((root->edge_block_entry[i].destination_vertex - 1) < 0))
                //     printf("Hit error at %lu\n", root->edge_block_entry[i].destination_vertex - 1);

                // page_factor = page_factor * 2;

                // float kochappi = 2;

                // d_pageRankVector_2[root->edge_block_entry[i].destination_vertex - 1] += page_factor;

                atomicAdd(&d_pageRankVector_2[root->edge_block_entry[i].destination_vertex - 1], page_factor);

                // if(root->edge_block_entry[i].destination_vertex == d_csr_edges[j]) {
                //     root->edge_block_entry[i].destination_vertex = INFTY;
                //     break;
                // }

            }

            // }

        }


        preorderTraversal_PageRank(device_vertex_dictionary, root->lptr, d_pageRankVector_1, d_pageRankVector_2, id, d_source_degrees, d_csr_offset, d_csr_edges, page_factor);



        // printf("\n");

        preorderTraversal_PageRank(device_vertex_dictionary, root->rptr, d_pageRankVector_1, d_pageRankVector_2, id, d_source_degrees, d_csr_offset, d_csr_edges, page_factor);

    }

}


__global__ void pageRankInitialization(float *d_pageRankVector_1, unsigned long vertex_size) {

    for(unsigned long i = 0 ; i < vertex_size ; i++) {

        d_pageRankVector_1[i] = 0.25;

    }

}

__global__ void pageRankKernel(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, float *d_pageRankVector_1, float *d_pageRankVector_2, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {


        // d_pageRankVector_2[id] = 0;

        // for(unsigned long i = 0 ; i < vertex_size ; i++) {

        //     if((device_vertex_dictionary->vertex_adjacency[i]->active_edge_count != 0))
        //         d_pageRankVector_2[id] += d_pageRankVector_1[i] / device_vertex_dictionary->vertex_adjacency[i]->active_edge_count;

        //         // d_pageRankVector_2[id] += d_pageRankVector_1[i] / 4;


        // }

        // d_pageRankVector_2[id] -= d_pageRankVector_1[id] / device_vertex_dictionary->vertex_adjacency[id]->active_edge_count;

        float page_factor = d_pageRankVector_1[id] / d_source_degrees[id];

        preorderTraversal_PageRank(device_vertex_dictionary, device_vertex_dictionary->vertex_adjacency[id]->edge_block_address, d_pageRankVector_1, d_pageRankVector_2, id, d_source_degrees, d_csr_offset, d_csr_edges, page_factor);

        // metadata code start

        // unsigned long start_index = d_csr_offset[id];
        // unsigned long end_index = d_csr_offset[id + 1];

        // for(unsigned long i = start_index ; i < end_index ; i++) {

        //     atomicAdd(&d_pageRankVector_2[d_csr_edges[i] - 1], page_factor);



        // }

        // printf("\n");

        // metadata code end



        // // for(unsigned long i = 0 ; i < device_vertex_dictionary->vertex_adjacency[id]->active_edge_count ; i++) {
        // for(unsigned long i = 0 ; i < d_source_degrees[id] ; i++) {

        //     // d_pageRankVector_2[id] += d_pageRankVector_1[i] / d_source_degrees[i];
        //     d_pageRankVector_2[id] += page_factor;

        //     // if((device_vertex_dictionary->vertex_adjacency[i]->active_edge_count != 0))
        //     //     d_pageRankVector_2[id] += d_pageRankVector_1[i] / device_vertex_dictionary->vertex_adjacency[i]->active_edge_count;

        //         // d_pageRankVector_2[id] += d_pageRankVector_1[i] / 4;


        // }


        // printf("PageRank kernel at id = %lu, neighbours = %lu, PageRankVector_1  is %f, PageRankVector_2 is %f\n", id, device_vertex_dictionary->vertex_adjacency[id]->active_edge_count, d_pageRankVector_1[id], d_pageRankVector_2[id]);


    }

}

__global__ void pageRank_kernel_preprocessing(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        unsigned long start_index = d_prefix_sum_edge_blocks[id];
        unsigned long end_index = d_prefix_sum_edge_blocks[id + 1];

        // source vector values start from 0
        for(unsigned long i = start_index ; i < end_index ; i++) {

            d_source_vector[i] = id;

        }

    }

}

__global__ void pageRank_kernel(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, float *d_pageRankVector_1, float *d_pageRankVector_2, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if((id < d_prefix_sum_edge_blocks[vertex_size])) {

        unsigned long source_vertex = d_source_vector[id];
        unsigned long start_index = d_csr_offset[source_vertex];
        unsigned long end_index = d_csr_offset[source_vertex + 1];

        if(start_index < end_index) {


            unsigned long index_counter = id - d_prefix_sum_edge_blocks[source_vertex];
            unsigned long bit_string = bit_string_lookup[index_counter];
            struct edge_block *root = device_vertex_dictionary->edge_block_address[source_vertex];

            // if(root != NULL) {

            root = traverse_bit_string(root, bit_string);

            // if((root != NULL) && (d_source_degrees[source_vertex] > 0)) {

            float page_factor = d_pageRankVector_1[source_vertex] / d_source_degrees[source_vertex];


            // printf("id=%lu, source=%lu, counter=%lu, bit_string=%lu, edge_block=%p, start_index=%lu, end_index=%lu\n", id, d_source_vector[id], index_counter, bit_string, root, start_index, end_index);

            for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {
                // for(unsigned long i = 0 ; i < root->active_edge_count ; i++) {

                if(root->edge_block_entry[i].destination_vertex == 0)
                    break;
                else {

                    // for(unsigned long j = start_index ; j < end_index ; j++) {

                    // if((root == NULL) || ((root->edge_block_entry[i].destination_vertex - 1) >= (vertex_size)) || ((root->edge_block_entry[i].destination_vertex - 1) < 0))
                    //     printf("Hit error at %lu\n", root->edge_block_entry[i].destination_vertex - 1);

                    // page_factor = page_factor * 2;

                    // float kochappi = 2;

                    // d_pageRankVector_2[root->edge_block_entry[i].destination_vertex - 1] += page_factor;

                    atomicAdd(&d_pageRankVector_2[root->edge_block_entry[i].destination_vertex - 1], page_factor);

                    // if(root->edge_block_entry[i].destination_vertex == d_csr_edges[j]) {
                    //     root->edge_block_entry[i].destination_vertex = INFTY;
                    //     break;
                    // }

                }

                // }

            }

            // }

            // }

        }

    }

}

__device__ float total_triangles = 0.0f;

__global__ void triangleCountingKernel(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, float *d_triangleCount, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        // float page_factor = d_pageRankVector_1[id] / d_source_degrees[id];

        // preorderTraversal_PageRank(device_vertex_dictionary, device_vertex_dictionary->vertex_adjacency[id]->edge_block_address, d_pageRankVector_1, d_pageRankVector_2, id, d_source_degrees, d_csr_offset, d_csr_edges, page_factor);

        // metadata code start

        unsigned long start_index = d_csr_offset[id];
        unsigned long end_index = d_csr_offset[id + 1];

        // printf("ID=%lu, start_index=%lu, end_index=%lu\n", id, start_index, end_index);


        for(unsigned long i = start_index ; i < end_index ; i++) {


            if((d_csr_edges[i] - 1) > id) {

                // printf("ID=%lu, u=%lu\n", id, d_csr_edges[i]);

                for(unsigned long j = i + 1 ; j < end_index ; j++) {

                    if((d_csr_edges[j] - 1) > (d_csr_edges[i] - 1)) {

                        // printf("ID=%lu, u=%lu and i=%lu, w=%lu and j=%lu\n", id, d_csr_edges[i] - 1, i, d_csr_edges[j], j);

                        // check if edge between i and j
                        // for(unsigned long k = d_csr_offset[d_csr_edges[j] - 1] ; k < d_csr_offset[d_csr_edges[j]] ; k++) {

                        //     printf("Check, ID=%lu, k=%lu, d_csr_edges[k]=%lu, d_csr_edges[i]=%lu\n", id, k, d_csr_edges[k], d_csr_edges[i]);

                        //     if(d_csr_edges[k] == (d_csr_edges[i] - 1)) {
                        //         atomicAdd(&d_triangleCount[id], 1);
                        //         atomicAdd(&d_triangleCount[d_csr_edges[i] - 1], 1);
                        //         atomicAdd(&d_triangleCount[d_csr_edges[k]], 1);
                        //         break;
                        //     }
                        // }
                        for(unsigned long k = d_csr_offset[d_csr_edges[i] - 1] ; k < d_csr_offset[d_csr_edges[i]] ; k++) {

                            // printf("Check, ID=%lu, k=%lu, d_csr_edges[i]=%lu, d_csr_edges[k]=%lu, d_csr_edges[j]=%lu\n", id, k, d_csr_edges[i], d_csr_edges[k], d_csr_edges[j]);

                            if((d_csr_edges[k] - 1) == (d_csr_edges[j] - 1)) {
                                atomicAdd(&total_triangles, 1);
                                atomicAdd(&d_triangleCount[id], 1);
                                atomicAdd(&d_triangleCount[d_csr_edges[i] - 1], 1);
                                atomicAdd(&d_triangleCount[d_csr_edges[j] - 1], 1);
                                break;
                            }
                        }

                    }

                }

            }


            // atomicAdd(&d_pageRankVector_2[d_csr_edges[i] - 1], page_factor);

            // if(id == 3)
            //     printf("%lu ", d_csr_edges[i]);

        }

        // printf("ID=%lu, Triangle count is %f\n", id, d_triangleCount[id]);

        // __syncthreads();
        // if(id == 0)
        //     printf("Total triangles is %f", total_triangles);

    }

}

__global__ void triangle_counting_kernel_1(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long *d_TC_edge_vector, unsigned long *d_source_vector_1) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if((id < d_prefix_sum_edge_blocks[vertex_size])) {

        unsigned long source_vertex = d_source_vector[id];
        unsigned long index_counter = id - d_prefix_sum_edge_blocks[source_vertex];
        unsigned long start_index = d_csr_offset[source_vertex] + (index_counter * EDGE_BLOCK_SIZE);
        unsigned long end_index = start_index + EDGE_BLOCK_SIZE;

        if(end_index > d_csr_offset[source_vertex + 1])
            end_index = d_csr_offset[source_vertex + 1];

        // printf("id=%lu, source=%lu, index_counter=%lu, start_index=%lu, end_index=%lu\n", id, source_vertex, index_counter, start_index, end_index);

        if(start_index < end_index) {

            unsigned long bit_string = bit_string_lookup[index_counter];
            struct edge_block *root = device_vertex_dictionary->vertex_adjacency[source_vertex]->edge_block_address;

            if(root != NULL) {

                root = traverse_bit_string(root, bit_string);

                unsigned long k = 0;
                for(unsigned long i = start_index ; i < end_index ; i++) {

                    d_TC_edge_vector[i] = root->edge_block_entry[k++].destination_vertex - 1;
                    d_source_vector_1[i] = source_vertex;

                    // if(d_TC_edge_vector[i] != (d_csr_edges[i] - 1))
                    //     printf("Hit csr\n");

                    // if(source_vertex == 0)
                    //     printf("%lu ", d_TC_edge_vector[i]);

                    // if(source_vertex == 540485)
                    //     printf("CSR offset of 540K is %lu\n", d_csr_offset[source_vertex]);

                }
            }

        }

    }

    // if(id == 0) {

    //     printf("TC source vector\n");
    //     for(unsigned long i = 0 ; i < batch_size ; i++)
    //         printf("%lu ", d_source_vector_1[i]);
    //     printf("\n");
    //     printf("TC edge vector\n");
    //     for(unsigned long i = 0 ; i < batch_size ; i++)
    //         printf("%lu ", d_TC_edge_vector[i]);
    //     printf("\n");
    // }

}

__global__ void triangle_counting_kernel_2(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long *d_TC_edge_vector, float *d_triangleCount, unsigned long *d_source_vector_1) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if((id < batch_size)) {

        unsigned long source_vertex = d_source_vector_1[id];
        unsigned long index_counter = id - d_csr_offset[source_vertex];
        unsigned long start_index = d_csr_offset[source_vertex];
        unsigned long end_index = d_csr_offset[source_vertex + 1];

        // if(end_index > d_csr_offset[source_vertex + 1])
        //     end_index = d_csr_offset[source_vertex + 1];

        // printf("id=%lu, source=%lu, index_counter=%lu, start_index=%lu, end_index=%lu\n", id, source_vertex, index_counter, start_index, end_index);

        //   forall(u in g.neighbors(v).filter(u < v))
        //      {
        //        forall(w in g.neighbors(v).filter(w > v))
        //          {
        //              if(g.is_an_edge(u,w))
        //                {
        //                 triangle_count+=1;
        //                }
        //          }

        //      }
        unsigned long temp, temp1, temp2;
        temp = d_TC_edge_vector[source_vertex];

        for(unsigned long i = start_index ; i < end_index ; i++) {

            temp1 = d_TC_edge_vector[i];

            if((temp1 < temp)) {
                // if((temp > temp1)) {

                for(unsigned long j = i ; j < end_index ; j++) {

                    temp2 = d_TC_edge_vector[j];

                    if((temp2 > temp)) {
                        // ;
                        atomicAdd(&d_triangleCount[source_vertex], 1);
                        atomicAdd(&d_triangleCount[temp1], 1);
                        atomicAdd(&d_triangleCount[temp2], 1);
                        // d_triangleCount[source_vertex] += 1;
                        // d_triangleCount[temp1] += 1;
                        // d_triangleCount[temp2] += 1;

                    }

                }

            }

        }

    }

    // if(id == 0) {

    //     printf("Triangle count\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++)
    //         printf("%f ", d_triangleCount[i]);
    //     printf("\n");

    // }

}

__device__ unsigned long tc_device_binary_search(unsigned long *input_array, unsigned long key, unsigned long size) {

    long start = 0;
    long end = (long)size;
    long mid;

    // printf("ID is %lu, mid is %lu, start is %lu, end is %lu, size is %lu\n", key, mid, start, end, size);

    while (start <= end) {

        mid = (start + end) / 2;
        // printf("ID is %lu, mid is %lu, start is %lu, end is %lu, size is %lu\n", key, mid, start, end, size);

        unsigned long item = input_array[mid] - 1;

        // Check if x is present at mid
        if (item == key) {
            // printf("ID is %lu, mid is %lu, start is %lu, end is %lu\n", key, mid, start, end, size);
            return mid + 1;
        }

        // If x greater, ignore left half
        if (item < key)
            start = mid + 1;

            // If x is smaller, ignore right half
        else
            end = mid - 1;
    }

    // If we reach here, then element was not present
    // printf("ID is %lu, mid is %lu, start is %lu, end is %lu\n", key, mid, start, end);
    // if(key)
    //     return start + 1;
    // else
    return start;
}

__device__ unsigned long long tc_final_device_binary_search(unsigned long *input_array, unsigned long long key, unsigned long long size) {

    unsigned long start = 0;
    unsigned long end = size;
    unsigned long mid;

    while (start <= end) {

        mid = (start + end) / 2;

        unsigned long item = input_array[mid] - 1;

        // Check if x is present at mid
        if (item == key)
            return 1;

        // If x greater, ignore left half
        if (item < key)
            start = mid + 1;

            // If x is smaller, ignore right half
        else
            end = mid - 1;
    }

    // If we reach here, then element was not present
    return 0;

}

__global__ void triangle_counting_kernel_VC(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, float *d_triangleCount, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        // float page_factor = d_pageRankVector_1[id] / d_source_degrees[id];

        // preorderTraversal_PageRank(device_vertex_dictionary, device_vertex_dictionary->vertex_adjacency[id]->edge_block_address, d_pageRankVector_1, d_pageRankVector_2, id, d_source_degrees, d_csr_offset, d_csr_edges, page_factor);

        // metadata code start9,223,372,036,854,775,807

        unsigned long start_index = d_csr_offset[id];
        unsigned long end_index = d_csr_offset[id + 1];

        // printf("ID=%lu, start_index=%lu, end_index=%lu, degree=%lu\n", id, start_index, end_index, end_index - start_index);

        unsigned long start_index_second_vertex = (tc_device_binary_search(d_csr_edges + start_index, id, end_index - start_index));

        // printf("ID=%lu, start_index=%lu, end_index=%lu, start_index_second_vertex=%lu\n", id, start_index, end_index, start_index_second_vertex);

        for(unsigned long i = start_index_second_vertex + d_csr_offset[id] ; i < end_index ; i++) {

            unsigned long second_vertex = d_csr_edges[i] - 1;

            unsigned long first_adjacency_size = end_index - start_index;
            unsigned long second_adjacency_size = d_csr_offset[second_vertex + 1] - d_csr_offset[second_vertex];

            // printf("ID=%lu, start_index_s=%lu, end_index_s=%lu, start_index_second_vertex=%lu, second_vertex=%lu\n", id, d_csr_offset[second_vertex], d_csr_offset[second_vertex + 1], start_index_second_vertex, second_vertex);

            if(first_adjacency_size <= second_adjacency_size) {

                unsigned long start_index_third_vertex = tc_device_binary_search(d_csr_edges + start_index, second_vertex, end_index - start_index);

                // printf("FA, ID=%lu, start_index=%lu, end_index=%lu, start_index_second_vertex=%lu, second_vertex=%lu, start_index_third_vertex=%lu\n", id, start_index, end_index, start_index_second_vertex, second_vertex, start_index_third_vertex);

                for(unsigned long j = start_index_third_vertex + d_csr_offset[id] ; j < end_index ; j++) {

                    unsigned long third_vertex = d_csr_edges[j] - 1;

                    unsigned long target_vertex_index;

                    // if(second_vertex != third_vertex) {
                    target_vertex_index = tc_final_device_binary_search(d_csr_edges + (unsigned long)d_csr_offset[second_vertex], third_vertex, d_csr_offset[second_vertex + 1] - d_csr_offset[second_vertex]);
                    // printf("FA, ID=%lu, , first_vertex=%lu, second_vertex=%lu, third_vertex=%lu, hit=%lu\n", id, id, second_vertex, third_vertex, target_vertex_index);
                    // printf("FA, ID=%lu, second_vertex=%lu, third_vertex=%lu, hit=%lu\n", id, second_vertex, third_vertex, target_vertex_index);
                    // printf("third_vertex_index=%lu\n", j);
                    // printf("yoyo\n");
                    if(target_vertex_index) {
                        atomicAdd(&total_triangles, 1);
                        atomicAdd(&d_triangleCount[id], 1);
                        atomicAdd(&d_triangleCount[second_vertex], 1);
                        atomicAdd(&d_triangleCount[third_vertex], 1);
                        // break;
                    }
                    // }

                }

            }

            else {

                unsigned long start_index_third_vertex = tc_device_binary_search(d_csr_edges + (unsigned long)d_csr_offset[second_vertex], second_vertex, d_csr_offset[second_vertex + 1] - d_csr_offset[second_vertex]);

                // printf("SA, ID=%lu, start_index=%lu, end_index=%lu, start_index_second_vertex=%lu, second_vertex=%lu, start_index_third_vertex=%lu\n", id, d_csr_offset[second_vertex], d_csr_offset[second_vertex + 1], start_index_second_vertex, second_vertex, start_index_third_vertex);

                for(unsigned long j = start_index_third_vertex + d_csr_offset[second_vertex] ; j < d_csr_offset[second_vertex + 1] ; j++) {

                    unsigned long third_vertex = d_csr_edges[j] - 1;
                    unsigned long target_vertex_index;

                    // if(second_vertex != third_vertex) {
                    target_vertex_index = tc_final_device_binary_search(d_csr_edges + start_index, third_vertex, end_index - start_index);
                    // printf("SA, ID=%lu, , first_vertex=%lu, second_vertex=%lu, third_vertex=%lu, hit=%lu\n", id, id, second_vertex, third_vertex, target_vertex_index);

                    // printf("SA, ID=%lu, second_vertex=%lu, third_vertex=%lu\n", id, second_vertex, third_vertex);
                    // printf("SA, ID=%lu, second_vertex=%lu, third_vertex=%lu, hit=%lu, j=%lu\n", id, second_vertex, third_vertex, target_vertex_index, j);
                    // printf("third_vertex_index=%lu\n", j);

                    if(target_vertex_index) {
                        atomicAdd(&total_triangles, 1);
                        atomicAdd(&d_triangleCount[id], 1);
                        atomicAdd(&d_triangleCount[second_vertex], 1);
                        atomicAdd(&d_triangleCount[third_vertex], 1);
                        // break;
                    }

                    // }

                }

            }

            // if((d_csr_edges[i]) > id) {

            //     // printf("ID=%lu, u=%lu\n", id, d_csr_edges[i]);

            //     for(unsigned long j = i + 1 ; j < end_index ; j++) {

            //         if((d_csr_edges[j]) > (d_csr_edges[i])) {

            //             for(unsigned long k = d_csr_offset[d_csr_edges[i]] ; k < d_csr_offset[d_csr_edges[i] + 1] ; k++) {

            //                 // printf("Check, ID=%lu, k=%lu, d_csr_edges[i]=%lu, d_csr_edges[k]=%lu, d_csr_edges[j]=%lu\n", id, k, d_csr_edges[i], d_csr_edges[k], d_csr_edges[j]);

            //                 if((d_csr_edges[k]) == (d_csr_edges[j])) {
            //                     atomicAdd(&total_triangles, 1);
            //                     atomicAdd(&d_triangleCount[id], 1);
            //                     atomicAdd(&d_triangleCount[d_csr_edges[i]], 1);
            //                     atomicAdd(&d_triangleCount[d_csr_edges[j]], 1);
            //                     break;
            //                 }
            //             }

            //         }

            //     }

            // }

        }

    }

}

__device__ unsigned long long tc_offset_device_binary_search(unsigned long *input_array, unsigned long key, unsigned long size) {

    unsigned long start = 0;
    unsigned long end = size;
    unsigned long mid;

    while (start <= end) {

        mid = (start + end) / 2;

        // Check if x is present at mid
        if (input_array[mid] == key)
            return mid;

        // If x greater, ignore left half
        if (input_array[mid] < key)
            start = mid + 1;

            // If x is smaller, ignore right half
        else
            end = mid - 1;
    }

    // If we reach here, then element was not present
    return start - 1;

}

__global__ void triangle_counting_kernel_EC_1(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, float *d_triangleCount, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_second_vertex_degrees) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        // float page_factor = d_pageRankVector_1[id] / d_source_degrees[id];

        // preorderTraversal_PageRank(device_vertex_dictionary, device_vertex_dictionary->vertex_adjacency[id]->edge_block_address, d_pageRankVector_1, d_pageRankVector_2, id, d_source_degrees, d_csr_offset, d_csr_edges, page_factor);

        // metadata code start9,223,372,036,854,775,807

        unsigned long start_index = d_csr_offset[id];
        unsigned long end_index = d_csr_offset[id + 1];

        // printf("ID=%lu, start_index=%lu, end_index=%lu, degree=%lu\n", id, start_index, end_index, end_index - start_index);

        unsigned long start_index_second_vertex = (tc_device_binary_search(d_csr_edges + start_index, id, end_index - start_index));

        // printf("ID=%lu, start_index=%lu, end_index=%lu, start_index_second_vertex=%lu\n", id, start_index, end_index, start_index_second_vertex);

        for(unsigned long i = start_index_second_vertex + d_csr_offset[id] ; i < end_index ; i++) {

            unsigned long second_vertex = d_csr_edges[i] - 1;

            unsigned long first_adjacency_size = end_index - start_index;
            unsigned long second_adjacency_size = d_csr_offset[second_vertex + 1] - d_csr_offset[second_vertex];

            // printf("ID=%lu, start_index_s=%lu, end_index_s=%lu, start_index_second_vertex=%lu, second_vertex=%lu\n", id, d_csr_offset[second_vertex], d_csr_offset[second_vertex + 1], start_index_second_vertex, second_vertex);

            d_second_vertex_degrees[id] = d_csr_offset[id + 1] - (start_index_second_vertex + d_csr_offset[id]);

            // if(first_adjacency_size <= second_adjacency_size) {

            //     unsigned long start_index_third_vertex = tc_device_binary_search(d_csr_edges + start_index, second_vertex, end_index - start_index);

            //     // printf("FA, ID=%lu, start_index=%lu, end_index=%lu, start_index_second_vertex=%lu, second_vertex=%lu, start_index_third_vertex=%lu\n", id, start_index, end_index, start_index_second_vertex, second_vertex, start_index_third_vertex);

            //     for(unsigned long j = start_index_third_vertex + d_csr_offset[id] ; j < end_index ; j++) {

            //         unsigned long third_vertex = d_csr_edges[j] - 1;

            //         unsigned long target_vertex_index;

            //         // if(second_vertex != third_vertex) {
            //             target_vertex_index = tc_final_device_binary_search(d_csr_edges + (unsigned long)d_csr_offset[second_vertex], third_vertex, d_csr_offset[second_vertex + 1] - d_csr_offset[second_vertex]);
            //             // printf("FA, ID=%lu, second_vertex=%lu, third_vertex=%lu, hit=%lu\n", id, second_vertex, third_vertex, target_vertex_index);
            //             // printf("third_vertex_index=%lu\n", j);
            //             // printf("yoyo\n");
            //             if(target_vertex_index) {
            //                 atomicAdd(&total_triangles, 1);
            //                 atomicAdd(&d_triangleCount[id], 1);
            //                 atomicAdd(&d_triangleCount[second_vertex], 1);
            //                 atomicAdd(&d_triangleCount[third_vertex], 1);
            //                 // break;
            //             }
            //         // }

            //     }

            // }

            // else {

            //     unsigned long start_index_third_vertex = tc_device_binary_search(d_csr_edges + (unsigned long)d_csr_offset[second_vertex], second_vertex, d_csr_offset[second_vertex + 1] - d_csr_offset[second_vertex]);

            //     // printf("SA, ID=%lu, start_index=%lu, end_index=%lu, start_index_second_vertex=%lu, second_vertex=%lu, start_index_third_vertex=%lu\n", id, d_csr_offset[second_vertex], d_csr_offset[second_vertex + 1], start_index_second_vertex, second_vertex, start_index_third_vertex);

            //     for(unsigned long j = start_index_third_vertex + d_csr_offset[second_vertex] ; j < d_csr_offset[second_vertex + 1] ; j++) {

            //         unsigned long third_vertex = d_csr_edges[j] - 1;
            //         unsigned long target_vertex_index;

            //         // if(second_vertex != third_vertex) {
            //             target_vertex_index = tc_final_device_binary_search(d_csr_edges + start_index, third_vertex, end_index - start_index);

            //             // printf("SA, ID=%lu, second_vertex=%lu, third_vertex=%lu\n", id, second_vertex, third_vertex);
            //             // printf("SA, ID=%lu, second_vertex=%lu, third_vertex=%lu, hit=%lu, j=%lu\n", id, second_vertex, third_vertex, target_vertex_index, j);
            //             // printf("third_vertex_index=%lu\n", j);

            //             if(target_vertex_index) {
            //                 atomicAdd(&total_triangles, 1);
            //                 atomicAdd(&d_triangleCount[id], 1);
            //                 atomicAdd(&d_triangleCount[second_vertex], 1);
            //                 atomicAdd(&d_triangleCount[third_vertex], 1);
            //                 // break;
            //             }

            //         // }

            //     }

            // }

        }

    }

    // if(!id) {

    //     printf("Second vertex offsets\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++)
    //         printf("%lu ", d_second_vertex_degrees[i]);
    //     printf("\n\n");

    // }

}

__global__ void triangle_counting_kernel_EC_2(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, float *d_triangleCount, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_second_vertex_degrees) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < d_second_vertex_degrees[vertex_size]) {



        // printf("ID=%lu, start_index=%lu, end_index=%lu, degree=%lu\n", id, start_index, end_index, end_index - start_index);

        // unsigned long start_index_second_vertex = (tc_device_binary_search(d_csr_edges + start_index, id, end_index - start_index));

        // printf("ID=%lu, start_index=%lu, end_index=%lu, start_index_second_vertex=%lu\n", id, start_index, end_index, start_index_second_vertex);

        // for(unsigned long i = start_index_second_vertex + d_csr_offset[id] ; i < end_index ; i++) {

        // unsigned long second_vertex = d_csr_edges[i] - 1;

        unsigned long first_vertex = tc_offset_device_binary_search(d_second_vertex_degrees, id, vertex_size);
        unsigned long start_index = d_csr_offset[first_vertex];
        unsigned long end_index = d_csr_offset[first_vertex + 1];

        unsigned long start_index_second_vertex = (id - d_second_vertex_degrees[first_vertex]) + (end_index - start_index) - (d_second_vertex_degrees[first_vertex + 1] - d_second_vertex_degrees[first_vertex]);
        start_index_second_vertex = d_csr_offset[first_vertex] + start_index_second_vertex;
        unsigned long second_vertex = d_csr_edges[start_index_second_vertex] - 1;


        printf("ID is %lu, first_vertex is %lu, second_vertex is %lu\n", id, first_vertex, second_vertex);
        // printf("ID is %lu, first_vertex is %lu\n", id, first_vertex);



        unsigned long first_adjacency_size = end_index - start_index;
        unsigned long second_adjacency_size = d_csr_offset[second_vertex + 1] - d_csr_offset[second_vertex];

        // printf("ID=%lu, start_index_s=%lu, end_index_s=%lu, start_index_second_vertex=%lu, second_vertex=%lu\n", id, d_csr_offset[second_vertex], d_csr_offset[second_vertex + 1], start_index_second_vertex, second_vertex);

        // d_second_vertex_degrees[id] = d_csr_offset[id + 1] - (start_index_second_vertex + d_csr_offset[id]);

        if(first_adjacency_size <= second_adjacency_size) {

            unsigned long start_index_third_vertex = tc_device_binary_search(d_csr_edges + start_index, second_vertex, end_index - start_index);

            // printf("FA, ID=%lu, start_index=%lu, end_index=%lu, start_index_second_vertex=%lu, second_vertex=%lu, start_index_third_vertex=%lu\n", id, start_index, end_index, start_index_second_vertex, second_vertex, start_index_third_vertex);

            for(unsigned long j = start_index_third_vertex + d_csr_offset[first_vertex] ; j < end_index ; j++) {

                unsigned long third_vertex = d_csr_edges[j] - 1;

                unsigned long target_vertex_index;

                // if(second_vertex != third_vertex) {
                target_vertex_index = tc_final_device_binary_search(d_csr_edges + (unsigned long)d_csr_offset[second_vertex], third_vertex, d_csr_offset[second_vertex + 1] - d_csr_offset[second_vertex]);
                // printf("FA, ID=%lu, , first_vertex=%lu, second_vertex=%lu, third_vertex=%lu, hit=%lu\n", id, first_vertex, second_vertex, third_vertex, target_vertex_index);
                // printf("third_vertex_index=%lu\n", j);
                // printf("yoyo\n");
                if(target_vertex_index) {
                    // printf("Hello\n");
                    atomicAdd(&total_triangles, 1);
                    atomicAdd(&d_triangleCount[first_vertex], 1);
                    atomicAdd(&d_triangleCount[second_vertex], 1);
                    atomicAdd(&d_triangleCount[third_vertex], 1);
                    // break;
                }
                // }

            }

        }

        else {

            unsigned long start_index_third_vertex = tc_device_binary_search(d_csr_edges + (unsigned long)d_csr_offset[second_vertex], second_vertex, d_csr_offset[second_vertex + 1] - d_csr_offset[second_vertex]);

            // printf("SA, ID=%lu, start_index=%lu, end_index=%lu, start_index_second_vertex=%lu, second_vertex=%lu, start_index_third_vertex=%lu\n", id, d_csr_offset[second_vertex], d_csr_offset[second_vertex + 1], start_index_second_vertex, second_vertex, start_index_third_vertex);

            for(unsigned long j = start_index_third_vertex + d_csr_offset[second_vertex] ; j < d_csr_offset[second_vertex + 1] ; j++) {

                unsigned long third_vertex = d_csr_edges[j] - 1;
                unsigned long target_vertex_index;

                // if(second_vertex != third_vertex) {
                target_vertex_index = tc_final_device_binary_search(d_csr_edges + start_index, third_vertex, end_index - start_index);
                // printf("SA, ID=%lu, , first_vertex=%lu, second_vertex=%lu, third_vertex=%lu, hit=%lu\n", id, first_vertex, second_vertex, third_vertex, target_vertex_index);

                // printf("SA, ID=%lu, second_vertex=%lu, third_vertex=%lu\n", id, second_vertex, third_vertex);
                // printf("SA, ID=%lu, second_vertex=%lu, third_vertex=%lu, hit=%lu, j=%lu\n", id, second_vertex, third_vertex, target_vertex_index, j);
                // printf("third_vertex_index=%lu\n", j);

                if(target_vertex_index) {
                    atomicAdd(&total_triangles, 1);
                    atomicAdd(&d_triangleCount[first_vertex], 1);
                    atomicAdd(&d_triangleCount[second_vertex], 1);
                    atomicAdd(&d_triangleCount[third_vertex], 1);
                    // break;
                }

                // }

            }

        }

        // }

    }

    // if(!id) {

    //     printf("Second vertex offsets\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++)
    //         printf("%lu ", d_second_vertex_degrees[i]);
    //     printf("\n\n");

    // }

}

__global__ void tc_second_vertex_offset_calculation (unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long vertex_size, unsigned long *d_second_vertex_offset, unsigned long *d_tc_thread_count) {

    // thrust::exclusive_scan(d_source_degrees_new.begin(), d_source_degrees_new.begin() + h_dp_thread_count, d_sssp_output_frontier_offset.begin());

    // printf("Current output frontier offset\n");
    // for(unsigned long i = 0 ; i < d_prev_thread_count + 1 ; i++)
    //     printf("%lu ", d_sssp_output_frontier_offset[i]);
    // printf("\n\n");

    // if(!type)
    //     for(unsigned long i = 1 ; i < d_prev_thread_count + 1 ; i++)
    //         d_sssp_output_frontier_offset[i] = d_source_degrees[d_sssp_queue_1[i - 1]] + d_sssp_output_frontier_offset[i - 1];
    // else
    //     for(unsigned long i = 1 ; i < d_prev_thread_count + 1 ; i++)
    //         d_sssp_output_frontier_offset[i] = d_source_degrees[d_sssp_queue_2[i - 1]] + d_sssp_output_frontier_offset[i - 1];

    // printf("Current input frontier\n");
    // if(!type)
    //     for(unsigned long i = 0 ; i < d_prev_thread_count ; i++)
    //         printf("%lu ", d_sssp_queue_1[i]);
    // else
    //     for(unsigned long i = 0 ; i < d_prev_thread_count ; i++)
    //         printf("%lu ", d_sssp_queue_2[i]);
    // printf("\n");
    // printf("Current output frontier offset\n");
    // for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //     printf("%lu ", d_second_vertex_offset[i]);
    // printf("\n\n");
    // printf("Threads needed %lu, thread_count is %lu\n", d_sssp_output_frontier_offset[d_prev_thread_count], d_prev_thread_count);

    *d_tc_thread_count = d_second_vertex_offset[vertex_size];

}

__device__ clock_t tester, temp;

__global__ void sssp_kernel_preprocessing(unsigned int *d_shortest_path, unsigned long *d_search_flag, unsigned long *d_mutex, unsigned long long *d_sssp_queue_1) {

    // d_search_flag = 0;
    d_shortest_path[0] = 0;
    d_sssp_queue_1[0] = 0;
    d_mutex[0] = 1;
    // tester = 0;

}

__global__ void sssp_kernel(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned int *d_shortest_path, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long *d_mutex, unsigned long *d_search_flag) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id == 0) {

    //     d_shortest_path[0] = 0;

    // //     printf("Printing Shortest Distances\n");
    // //     for(unsigned long i = 0 ; i < vertex_size ; i++)
    // //         printf("%u ", d_shortest_path[i]);
    // //     printf("\n");

    // }

    if((id < d_prefix_sum_edge_blocks[vertex_size])) {

        unsigned long source_vertex = d_source_vector[id];

        if((d_shortest_path[source_vertex] != UINT_MAX) && (d_csr_offset[source_vertex] < d_csr_offset[source_vertex + 1])) {

            temp = clock();

            unsigned long index_counter = id - d_prefix_sum_edge_blocks[source_vertex];
            unsigned long bit_string = bit_string_lookup[index_counter];
            struct edge_block *root = device_vertex_dictionary->vertex_adjacency[source_vertex]->edge_block_address;


            root = traverse_bit_string(root, bit_string);

            temp = clock() - temp;
            tester += temp;

            unsigned int new_distance = d_shortest_path[source_vertex] + 1;
            unsigned long destination_vertex;


            for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

                destination_vertex = root->edge_block_entry[i].destination_vertex;

                if((destination_vertex == 0))
                    break;

                else if(new_distance < d_shortest_path[destination_vertex - 1]) {

                    atomicMin(&(d_shortest_path[destination_vertex - 1]), new_distance);

                    // d_shortest_path[destination_vertex - 1] = new_distance;

                    *d_search_flag = 1;

                }

            }

        }

    }

    // if(id == 0) {

    //     printf("Printing Shortest Distances\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++)
    //         printf("%u ", d_shortest_path[i]);
    //     printf("\n");

    // }

}

// __device__ unsigned short d_search_flag;

__global__ void sssp_kernel_child(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned int *d_shortest_path, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long *d_TC_edge_vector, unsigned long *d_source_vector_1, unsigned long *d_mutex) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id == 0) {

    //     d_shortest_path[0] = 0;

    // //     printf("Printing Shortest Distances\n");
    // //     for(unsigned long i = 0 ; i < vertex_size ; i++)
    // //         printf("%u ", d_shortest_path[i]);
    // //     printf("\n");

    // }

    // if((id < batch_size) && (d_shortest_path[d_source_vector_1[id]] != UINT_MAX) && (!d_mutex[d_TC_edge_vector[id]])) {
    // if((id < batch_size) && (d_shortest_path[d_source_vector_1[id]] != UINT_MAX)) {
    if((id < batch_size) && (d_shortest_path[d_source_vector_1[id]] != UINT_MAX)) {

        // unsigned long source_vertex = d_source_vector_1[id];

        // if((d_shortest_path[d_source_vector_1[id]] != UINT_MAX)) {

        // temp = clock();

        // unsigned long index_counter = id - d_prefix_sum_edge_blocks[source_vertex];
        // unsigned long bit_string = bit_string_lookup[index_counter];
        // struct edge_block *root = device_vertex_dictionary->vertex_adjacency[source_vertex]->edge_block_address;


        // root = traverse_bit_string(root, bit_string);

        // temp = clock() - temp;
        // tester += temp;

        // unsigned int new_distance = d_shortest_path[d_source_vector_1[id]] + 1;
        // unsigned long destination_vertex = d_TC_edge_vector[id];

        // printf("ID=%lu, source=%lu, destination=%lu\n", id, source_vertex, destination_vertex);

        // for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

        // unsigned int *src = &d_shortest_path[d_source_vector_1[id]];
        // unsigned int *dst = &d_shortest_path[d_TC_edge_vector[id]];


        // if(*src + 1 < *dst) {

        //     atomicMin(dst, *src + 1);

        //     // // d_shortest_path[destination_vertex - 1] = new_distance;
        //     if(*d_search_flag != 1)
        //         *d_search_flag = 1;
        //     // *u_search_flag = 1;

        // }

        // if(!d_mutex[d_TC_edge_vector[id]]) {
        // unsigned long long empty = 0;
        // unsigned long long populated = 10000;
        // if(!atomicCAS(&(d_mutex[d_TC_edge_vector[id]]), 0, 1)) {

        //     // atomicAdd(&(d_shortest_path[d_TC_edge_vector[id]]), d_shortest_path[d_source_vector_1[id]] + 1);

        //     atomicMin(&(d_shortest_path[d_TC_edge_vector[id]]), d_shortest_path[d_source_vector_1[id]] + 1);

        //     // d_shortest_path[d_TC_edge_vector[id]] = d_shortest_path[d_source_vector_1[id]] + 1;
        //     // d_mutex[d_TC_edge_vector[id]] = 1;
        //     if(*d_search_flag != 1)
        //         *d_search_flag = 1;

        // }
        // printf("atomicMin hit\n");

        if(d_shortest_path[d_source_vector_1[id]] + 1 < d_shortest_path[d_TC_edge_vector[id]]) {

            atomicMin(&(d_shortest_path[d_TC_edge_vector[id]]), d_shortest_path[d_source_vector_1[id]] + 1);

            // d_shortest_path[d_TC_edge_vector[id]] = d_shortest_path[d_source_vector_1[id]] + 1;
            // if(*d_search_flag != 1)
            // *d_search_flag = 1;
            // *u_search_flag = 1;

        }



        // }

    }

    // if(id == 0) {

    //     printf("Printing Shortest Distances\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++)
    //         printf("%u ", d_shortest_path[i]);
    //     printf("\n");

    // }

}

__device__ unsigned long d_search_flag;
// __device__ float d_dp_thread_count = 1.00f;
// __device__ unsigned long long d_dp_thread_count = 1;
__device__ unsigned long type = 0;
// __device__ float d_prev_thread_count = 1.00f;
__device__ unsigned long long d_prev_thread_count = 1;
__device__ unsigned long long d_prev_thread_count_16to31 = 0;

__global__ void sssp_output_frontier_preprocessing(unsigned long *d_source_degrees, unsigned long long *d_sssp_queue_1, unsigned long long *d_sssp_queue_2, unsigned long long *d_sssp_output_frontier_offset, unsigned long *d_csr_offset) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < d_prev_thread_count) {

        unsigned long source = d_sssp_queue_1[id];

        // if(!type)
        // d_sssp_output_frontier_offset[id] = d_source_degrees[d_sssp_queue_1[id]];
        d_sssp_output_frontier_offset[id] = d_csr_offset[source + 1] - d_csr_offset[source];
        // else
        // d_sssp_output_frontier_offset[id] = d_source_degrees[d_sssp_queue_2[id]];

    }

    // if(id == 0) {
    //     printf("Preprocessing\n");
    //     for(unsigned long i = 0 ; i < d_prev_thread_count; i++)
    //         printf("%lu ", d_sssp_output_frontier_offset[i]);
    //     printf("\n\n");
    // }


}

__global__ void sssp_output_frontier_preprocessing_EBC(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long *d_source_degrees, unsigned long long *d_sssp_queue_1, unsigned long long *d_sssp_queue_2, unsigned long long *d_sssp_output_frontier_offset, unsigned long *d_csr_offset, unsigned long *d_prefix_sum_edge_blocks) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < d_prev_thread_count) {

        unsigned long source = d_sssp_queue_1[id];

        // if(!type)
        // d_sssp_output_frontier_offset[id] = d_source_degrees[d_sssp_queue_1[id]];
        // d_sssp_output_frontier_offset[id] = d_csr_offset[source + 1] - d_csr_offset[source];
        // d_sssp_output_frontier_offset[id] = device_vertex_dictionary->vertex_adjacency[source]->edge_block_count;
        d_sssp_output_frontier_offset[id] = device_vertex_dictionary->edge_block_count[source];
        // d_sssp_output_frontier_offset[id] = d_prefix_sum_edge_blocks[source + 1] - d_prefix_sum_edge_blocks[source];
        // else
        // d_sssp_output_frontier_offset[id] = d_source_degrees[d_sssp_queue_2[id]];

    }

    // if(id == 0) {
    //     printf("Preprocessing\n");
    //     for(unsigned long i = 0 ; i < d_prev_thread_count; i++)
    //         printf("%lu ", d_sssp_output_frontier_offset[i]);
    //     printf("\n\n");
    // }


}

__global__ void sssp_output_frontier_offset_calculation (unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long long *d_sssp_queue_1, unsigned long long *d_sssp_queue_2, unsigned long long *d_sssp_output_frontier_offset, unsigned long long *d_sssp_e_threads) {

    // thrust::exclusive_scan(d_source_degrees_new.begin(), d_source_degrees_new.begin() + h_dp_thread_count, d_sssp_output_frontier_offset.begin());

    // printf("Current output frontier offset\n");
    // for(unsigned long i = 0 ; i < d_prev_thread_count + 1 ; i++)
    //     printf("%lu ", d_sssp_output_frontier_offset[i]);
    // printf("\n\n");

    // if(!type)
    //     for(unsigned long i = 1 ; i < d_prev_thread_count + 1 ; i++)
    //         d_sssp_output_frontier_offset[i] = d_source_degrees[d_sssp_queue_1[i - 1]] + d_sssp_output_frontier_offset[i - 1];
    // else
    //     for(unsigned long i = 1 ; i < d_prev_thread_count + 1 ; i++)
    //         d_sssp_output_frontier_offset[i] = d_source_degrees[d_sssp_queue_2[i - 1]] + d_sssp_output_frontier_offset[i - 1];

    // printf("Current input frontier\n");
    // if(!type)
    //     for(unsigned long i = 0 ; i < d_prev_thread_count ; i++)
    //         printf("%lu ", d_sssp_queue_1[i]);
    // else
    //     for(unsigned long i = 0 ; i < d_prev_thread_count ; i++)
    //         printf("%lu ", d_sssp_queue_2[i]);
    // printf("\n");
    // printf("Current output frontier offset\n");
    // for(unsigned long i = 0 ; i < d_prev_thread_count + 1 ; i++)
    //     printf("%lu ", d_sssp_output_frontier_offset[i]);
    // printf("\n\n");
    // printf("Threads needed %lu, thread_count is %lu\n", d_sssp_output_frontier_offset[d_prev_thread_count], d_prev_thread_count);

    *d_sssp_e_threads = d_sssp_output_frontier_offset[d_prev_thread_count];

}

// void parallel_sssp_output_frontier_calculation(unsigned long h_dp_threads, unsigned long type, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long long *d_sssp_queue_1, unsigned long long *d_sssp_queue_2, unsigned long long *d_sssp_output_frontier_offset, unsigned long long *d_sssp_e_threads) {

//     while()

// }

// __global__ void sssp_kernel_child_VC(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned int *d_shortest_path, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long *d_TC_edge_vector, unsigned long *d_source_vector_1, unsigned long *d_mutex, float *d_sssp_queue_1, float *d_sssp_queue_2) {
__global__ void sssp_kernel_child_VC(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned int *d_shortest_path, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long *d_TC_edge_vector, unsigned long *d_source_vector_1, unsigned long *d_mutex, unsigned long long *d_sssp_queue_1, unsigned long long *d_sssp_queue_2, unsigned long long *d_dp_thread_count) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("Hit\n");
    // printf("ID=%lu, count=%lu\n", id, int(d_dp_thread_count));

    if((id < d_prev_thread_count)) {

        // printf("ID = %lu,\n", id);
        // if((id < vertex_size)) {
        // if((id < vertex_size) && (d_shortest_path[id] != UINT_MAX)) {
        // printf("Hit Inner\n");

        // unsigned long start_index = d_csr_offset[int(d_sssp_queue_1[id])];
        // unsigned long end_index = d_csr_offset[int(d_sssp_queue_1[id]) + 1];

        if((!type)) {

            // unsigned long start_index = d_csr_offset[long(d_sssp_queue_1[id])];
            // unsigned long end_index = d_csr_offset[long(d_sssp_queue_1[id]) + 1];

            unsigned long start_index = d_csr_offset[(d_sssp_queue_1[id])];
            unsigned long end_index = d_csr_offset[(d_sssp_queue_1[id]) + 1];

            // printf("start_index\n");

            // printf("Beginning ID %lu, source %lu\n", id, d_sssp_queue_2[id]);


            for(unsigned long i = start_index ; i < end_index ; i++) {

                // if(d_shortest_path[long(d_sssp_queue_1[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {
                if(d_shortest_path[(d_sssp_queue_1[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {

                    // printf("Hit\n");

                    // atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[long(d_sssp_queue_1[id])] + 1);
                    atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[(d_sssp_queue_1[id])] + 1);

                    // unsigned long temp = d_TC_edge_vector[i];

                    // if(!type)
                    // atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_edge_vector[i]);
                    // atomicAdd(d_sssp_queue_1 + int(atomicAdd(&d_dp_thread_count), 1), d_TC_edge_vector[i]);
                    // else
                    // atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_edge_vector[i]);

                    // if(d_search_flag != 1)
                    //     d_search_flag = 1;
                    // atomicOp below
                    // d_dp_thread_count += 1;

                    // unsigned long long temp = atomicAdd(&d_dp_thread_count, 1);
                    // printf("Hit 2 with thread count %lu\n", d_dp_thread_count);
                    unsigned long long temp = atomicAdd(d_dp_thread_count, 1);

                    // cudaDeviceSynchronize();
                    // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
                    *(d_sssp_queue_2 + (temp)) = d_TC_edge_vector[i];
                    // printf("ID %lu, source %lu ,thread_counter %lu, d_TC_edge_vector is %lu, d_sssp_queue_2 is %p with vertex %lu\n", id, d_sssp_queue_1[id], long(d_dp_thread_count), d_TC_edge_vector[i], d_sssp_queue_2 + long(d_dp_thread_count), long(d_sssp_queue_2[long(d_dp_thread_count)]));
                    // printf("ID %lu, source %lu ,thread_counter %lu, d_TC_edge_vector is %lu, d_sssp_queue_2 is %p with vertex %lu\n", id, d_sssp_queue_1[id], (temp), d_TC_edge_vector[i], d_sssp_queue_2 + (temp) - 1, (d_sssp_queue_2[(temp) - 1]));

                    //     atomicAdd(d_sssp_queue_1 + int(d_dp_thread_count), d_TC_edge_vector[i]);
                    // else
                    //     atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_ed

                }



            }

        }

        else {

            // unsigned long start_index = d_csr_offset[long(d_sssp_queue_1[id])];
            // unsigned long end_index = d_csr_offset[long(d_sssp_queue_1[id]) + 1];

            unsigned long start_index = d_csr_offset[(d_sssp_queue_2[id])];
            unsigned long end_index = d_csr_offset[(d_sssp_queue_2[id]) + 1];

            // printf("start_index\n");
            // printf("Beginning ID %lu, source %lu\n", id, d_sssp_queue_2[id]);

            for(unsigned long i = start_index ; i < end_index ; i++) {

                // if(d_shortest_path[long(d_sssp_queue_1[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {
                if(d_shortest_path[(d_sssp_queue_2[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {

                    // printf("Hit\n");

                    // atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[long(d_sssp_queue_1[id])] + 1);
                    atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[(d_sssp_queue_2[id])] + 1);

                    // unsigned long temp = d_TC_edge_vector[i];

                    // if(!type)
                    // atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_edge_vector[i]);
                    // atomicAdd(d_sssp_queue_1 + int(atomicAdd(&d_dp_thread_count), 1), d_TC_edge_vector[i]);
                    // else
                    // atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_edge_vector[i]);

                    // if(d_search_flag != 1)
                    //     d_search_flag = 1;
                    // atomicOp below
                    // d_dp_thread_count += 1;

                    // atomicAdd(&d_dp_thread_count, 1);
                    // // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
                    // *(d_sssp_queue_1 + (d_dp_thread_count) - 1) = d_TC_edge_vector[i];

                    // unsigned long long temp = atomicAdd(&d_dp_thread_count, 1);
                    unsigned long long temp = atomicAdd(d_dp_thread_count, 1);
                    // cudaDeviceSynchronize();

                    // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
                    *(d_sssp_queue_1 + (temp) ) = d_TC_edge_vector[i];

                    // printf("ID %lu has thread_counter %lu, d_TC_edge_vector is %lu, d_sssp_queue_2 is %p with vertex %lu\n", id, long(d_dp_thread_count), d_TC_edge_vector[i], d_sssp_queue_2 + long(d_dp_thread_count), long(d_sssp_queue_2[long(d_dp_thread_count)]));
                    // printf("ID %lu, source %lu ,thread_counter %lu, d_TC_edge_vector is %lu, d_sssp_queue_1 is %p with vertex %lu\n", id, d_sssp_queue_2[id], (temp), d_TC_edge_vector[i], d_sssp_queue_1 + (temp) - 1, (d_sssp_queue_1[(temp) - 1]));

                    //     atomicAdd(d_sssp_queue_1 + int(d_dp_thread_count), d_TC_edge_vector[i]);
                    // else
                    //     atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_ed

                }



            }

        }

    }

    // if(id == 0) {

    //     printf("Printing SSSP queue\n");
    //     if(!type) {
    //         for(unsigned long i = 0 ; i < *d_dp_thread_count ; i++)
    //             printf("%lu ", long(d_sssp_queue_2[i]));
    //     }
    //     else {
    //         for(unsigned long i = 0 ; i < *d_dp_thread_count ; i++)
    //             printf("%lu ", long(d_sssp_queue_1[i]));
    //     }
    //     printf("\n\n");

    // }

}

__device__ void sssp_preorder_traversal(struct edge_block *root, unsigned int *d_shortest_path, unsigned long long source, unsigned long long *d_sssp_queue_2, unsigned long long *d_dp_thread_count) {

    if(root == NULL)
        return;

    else {

        // printf("\nedge block edge count = %lu, %p, ", root->active_edge_count, root);

        for(unsigned long index = 0 ; index < EDGE_BLOCK_SIZE ; index++) {

            // printf("%lu ", root->edge_block_entry[j].destination_vertex);

            if(root->edge_block_entry[index].destination_vertex == 0)
                break;

            unsigned long destination = root->edge_block_entry[index].destination_vertex - 1;

            // for(unsigned long i = start_index ; i < end_index ; i++) {

            // if(d_shortest_path[long(d_sssp_queue_1[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {
            if(d_shortest_path[source] + 1 < d_shortest_path[destination]) {

                // printf("Hit\n");

                // atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[long(d_sssp_queue_1[id])] + 1);
                atomicMin(&(d_shortest_path[destination]), d_shortest_path[source] + 1);

                unsigned long long temp = atomicAdd(d_dp_thread_count, 1);

                *(d_sssp_queue_2 + (temp)) = destination;


            }

            // }


        }

        sssp_preorder_traversal(root->lptr, d_shortest_path, source, d_sssp_queue_2, d_dp_thread_count);



        // printf("\n");

        sssp_preorder_traversal(root->rptr, d_shortest_path, source, d_sssp_queue_2, d_dp_thread_count);

    }

}

__global__ void sssp_kernel_VC_preorder(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned int *d_shortest_path, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long *d_TC_edge_vector, unsigned long *d_source_vector_1, unsigned long *d_mutex, unsigned long long *d_sssp_queue_1, unsigned long long *d_sssp_queue_2, unsigned long long *d_dp_thread_count) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("Hit\n");
    // printf("ID=%lu, count=%lu\n", id, int(d_dp_thread_count));

    if((id < d_prev_thread_count)) {

        // printf("ID = %lu,\n", id);
        // if((id < vertex_size)) {
        // if((id < vertex_size) && (d_shortest_path[id] != UINT_MAX)) {
        // printf("Hit Inner\n");

        // unsigned long start_index = d_csr_offset[int(d_sssp_queue_1[id])];
        // unsigned long end_index = d_csr_offset[int(d_sssp_queue_1[id]) + 1];

        // if((!type)) {

        // unsigned long start_index = d_csr_offset[long(d_sssp_queue_1[id])];
        // unsigned long end_index = d_csr_offset[long(d_sssp_queue_1[id]) + 1];

        unsigned long source = d_sssp_queue_1[id];

        struct edge_block *root = device_vertex_dictionary->vertex_adjacency[source]->edge_block_address;

        // printf("ID is %lu, source is %lu, root is %p and %p\n", id, source, root, device_vertex_dictionary->vertex_adjacency[0]->edge_block_address);


        sssp_preorder_traversal(root, d_shortest_path, source, d_sssp_queue_2, d_dp_thread_count);


        // unsigned long start_index = d_csr_offset[(d_sssp_queue_1[id])];
        // unsigned long end_index = d_csr_offset[(d_sssp_queue_1[id]) + 1];

        // // printf("start_index\n");

        // // printf("Beginning ID %lu, source %lu\n", id, d_sssp_queue_2[id]);


        // for(unsigned long i = start_index ; i < end_index ; i++) {

        //     // if(d_shortest_path[long(d_sssp_queue_1[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {
        //     if(d_shortest_path[(d_sssp_queue_1[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {

        //         // printf("Hit\n");

        //         // atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[long(d_sssp_queue_1[id])] + 1);
        //         atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[(d_sssp_queue_1[id])] + 1);

        //         // unsigned long temp = d_TC_edge_vector[i];

        //         // if(!type)
        //             // atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_edge_vector[i]);
        //             // atomicAdd(d_sssp_queue_1 + int(atomicAdd(&d_dp_thread_count), 1), d_TC_edge_vector[i]);
        //         // else
        //             // atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_edge_vector[i]);

        //         // if(d_search_flag != 1)
        //         //     d_search_flag = 1;
        //             // atomicOp below
        //             // d_dp_thread_count += 1;

        //             // unsigned long long temp = atomicAdd(&d_dp_thread_count, 1);
        //             // printf("Hit 2 with thread count %lu\n", d_dp_thread_count);
        //             unsigned long long temp = atomicAdd(d_dp_thread_count, 1);

        //             // cudaDeviceSynchronize();
        //             // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
        //             *(d_sssp_queue_2 + (temp)) = d_TC_edge_vector[i];
        //             // printf("ID %lu, source %lu ,thread_counter %lu, d_TC_edge_vector is %lu, d_sssp_queue_2 is %p with vertex %lu\n", id, d_sssp_queue_1[id], long(d_dp_thread_count), d_TC_edge_vector[i], d_sssp_queue_2 + long(d_dp_thread_count), long(d_sssp_queue_2[long(d_dp_thread_count)]));
        //             // printf("ID %lu, source %lu ,thread_counter %lu, d_TC_edge_vector is %lu, d_sssp_queue_2 is %p with vertex %lu\n", id, d_sssp_queue_1[id], (temp), d_TC_edge_vector[i], d_sssp_queue_2 + (temp) - 1, (d_sssp_queue_2[(temp) - 1]));

        //         //     atomicAdd(d_sssp_queue_1 + int(d_dp_thread_count), d_TC_edge_vector[i]);
        //         // else
        //         //     atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_ed

        //     }



        // }

        // }

        // else {

        //         // unsigned long start_index = d_csr_offset[long(d_sssp_queue_1[id])];
        //         // unsigned long end_index = d_csr_offset[long(d_sssp_queue_1[id]) + 1];

        //         unsigned long start_index = d_csr_offset[(d_sssp_queue_2[id])];
        //         unsigned long end_index = d_csr_offset[(d_sssp_queue_2[id]) + 1];

        //         // printf("start_index\n");
        //         // printf("Beginning ID %lu, source %lu\n", id, d_sssp_queue_2[id]);

        //         for(unsigned long i = start_index ; i < end_index ; i++) {

        //             // if(d_shortest_path[long(d_sssp_queue_1[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {
        //             if(d_shortest_path[(d_sssp_queue_2[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {

        //                 // printf("Hit\n");

        //                 // atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[long(d_sssp_queue_1[id])] + 1);
        //                 atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[(d_sssp_queue_2[id])] + 1);

        //                 // unsigned long temp = d_TC_edge_vector[i];

        //                 // if(!type)
        //                     // atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_edge_vector[i]);
        //                     // atomicAdd(d_sssp_queue_1 + int(atomicAdd(&d_dp_thread_count), 1), d_TC_edge_vector[i]);
        //                 // else
        //                     // atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_edge_vector[i]);

        //                 // if(d_search_flag != 1)
        //                 //     d_search_flag = 1;
        //                     // atomicOp below
        //                     // d_dp_thread_count += 1;

        //                     // atomicAdd(&d_dp_thread_count, 1);
        //                     // // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
        //                     // *(d_sssp_queue_1 + (d_dp_thread_count) - 1) = d_TC_edge_vector[i];

        //                     // unsigned long long temp = atomicAdd(&d_dp_thread_count, 1);
        //                     unsigned long long temp = atomicAdd(d_dp_thread_count, 1);
        //                     // cudaDeviceSynchronize();

        //                     // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
        //                     *(d_sssp_queue_1 + (temp) ) = d_TC_edge_vector[i];

        //                     // printf("ID %lu has thread_counter %lu, d_TC_edge_vector is %lu, d_sssp_queue_2 is %p with vertex %lu\n", id, long(d_dp_thread_count), d_TC_edge_vector[i], d_sssp_queue_2 + long(d_dp_thread_count), long(d_sssp_queue_2[long(d_dp_thread_count)]));
        //                     // printf("ID %lu, source %lu ,thread_counter %lu, d_TC_edge_vector is %lu, d_sssp_queue_1 is %p with vertex %lu\n", id, d_sssp_queue_2[id], (temp), d_TC_edge_vector[i], d_sssp_queue_1 + (temp) - 1, (d_sssp_queue_1[(temp) - 1]));

        //                 //     atomicAdd(d_sssp_queue_1 + int(d_dp_thread_count), d_TC_edge_vector[i]);
        //                 // else
        //                 //     atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_ed

        //             }



        //         }

        // }

    }

    // if(id == 0) {

    //     printf("Printing SSSP queue\n");
    //     if(!type) {
    //         for(unsigned long i = 0 ; i < *d_dp_thread_count ; i++)
    //             printf("%lu ", long(d_sssp_queue_2[i]));
    //     }
    //     else {
    //         for(unsigned long i = 0 ; i < *d_dp_thread_count ; i++)
    //             printf("%lu ", long(d_sssp_queue_1[i]));
    //     }
    //     printf("\n\n");

    // }

}



__device__ void push(struct edge_block **stack, long long *top, unsigned long stack_size, struct edge_block *root) {

    int x;

    if (*top == stack_size - 1) {
        // printf("\nOverflow!!");
        ;
    }
    else {
        // printf("\nEnter the element to be added onto the stack: ");
        // scanf("%d", &x);
        // top = top + 1;
        // *top++;
        stack[*top + 1] = root;
    }
}

__device__ struct edge_block * pop(struct edge_block **stack, long long *top) {

    if (*top == -1)
    {
        printf("\nUnderflow!!");
    }
    else
    {
        // printf("\nPopped element: %d", inp_array[top]);
        // top = top - 1;
        struct edge_block *block = stack[*top];
        // *top--;
        return block;
    }
}

__global__ void sssp_kernel_VC_iterative(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned int *d_shortest_path, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long *d_TC_edge_vector, unsigned long *d_source_vector_1, unsigned long *d_mutex, unsigned long long *d_sssp_queue_1, unsigned long long *d_sssp_queue_2, unsigned long long *d_dp_thread_count) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if((id < d_prev_thread_count)) {

        long long top = -1;
        const unsigned long stack_size = 100;
        struct edge_block *stack[stack_size];

        unsigned long source = d_sssp_queue_1[id];

        struct edge_block *root = device_vertex_dictionary->edge_block_address[source];

        if(root == NULL)
            return;

        struct edge_block *curr_node = root;

        // printf("ID is %lu, source is %lu with address %p, stack base is %p, root is %p, curr_node is %p\n", id, source, &source, stack, root, curr_node);

        while((top > -1) || (curr_node != NULL)) {

            if(curr_node) {

                // code here
                for(unsigned long index = 0 ; index < EDGE_BLOCK_SIZE ; index++) {

                    // printf("%lu ", root->edge_block_entry[j].destination_vertex);

                    if(curr_node->edge_block_entry[index].destination_vertex == 0)
                        break;

                    unsigned long destination = curr_node->edge_block_entry[index].destination_vertex - 1;

                    // for(unsigned long i = start_index ; i < end_index ; i++) {

                    // if(d_shortest_path[long(d_sssp_queue_1[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {
                    if(d_shortest_path[source] + 1 < d_shortest_path[destination]) {

                        // printf("Hit for source %lu and destination %lu\n", source, destination);

                        // atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[long(d_sssp_queue_1[id])] + 1);
                        atomicMin(&(d_shortest_path[destination]), d_shortest_path[source] + 1);

                        unsigned long long temp = atomicAdd(d_dp_thread_count, 1);

                        *(d_sssp_queue_2 + (temp)) = destination;


                    }

                    // }


                }

                if(curr_node->rptr != NULL) {
                    push(stack, &top, stack_size, curr_node->rptr);
                    top++;
                    // printf("ID %lu pushed %p to stack, top=%lu\n", id, curr_node->rptr, top);
                }
                curr_node = curr_node->lptr;

            }
            else {

                curr_node = pop(stack, &top);
                top--;
                // printf("ID %lu popped %p from the stack\n", id, curr_node);
            }

            // printf("ID %lu, curr_node is now %p\n", id, curr_node);

        }

        // sssp_preorder_traversal(root, d_shortest_path, source, d_sssp_queue_2, d_dp_thread_count);

    }

    // if(id == 0) {

    //     printf("Printing SSSP input frontier\n");
    //         for(unsigned long i = 0 ; i < d_prev_thread_count ; i++)
    //             printf("%lu ", long(d_sssp_queue_1[i]));
    //     printf("\nPrinting SSSP output frontier\n");
    //     // if(!type) {
    //         for(unsigned long i = 0 ; i < *d_dp_thread_count ; i++)
    //             printf("%lu ", long(d_sssp_queue_2[i]));
    //     // }
    //     // else {

    //     // }
    //     printf("\n\n");

    // }

}

__global__ void sssp_kernel_VC_iterative_LB(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned int *d_shortest_path, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long *d_TC_edge_vector, unsigned long *d_source_vector_1, unsigned long *d_mutex, unsigned long long *d_sssp_queue_1, unsigned long long *d_sssp_queue_2, unsigned long long *d_dp_thread_count, unsigned long thread_multiplier) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if((id < (d_prev_thread_count * thread_multiplier))) {

        long long top = -1;
        const unsigned long stack_size = 1;
        struct edge_block *stack[stack_size];

        unsigned long source = d_sssp_queue_1[id / thread_multiplier];

        // unsigned long source_index = 0;
        // source_index = device_binary_search(d_sssp_output_frontier_offset, id / thread_multiplier, d_prev_thread_count);

        // unsigned long source = d_sssp_queue_1[source_index];

        struct edge_block *root = device_vertex_dictionary->edge_block_address[source];

        if(root == NULL)
            return;

        struct edge_block *curr_node = root;

        // printf("ID is %lu, source is %lu with address %p, stack base is %p, root is %p, curr_node is %p\n", id, source, &source, stack, root, curr_node);

        while((top > -1) || (curr_node != NULL)) {

            if(curr_node) {

                // code here
                // unsigned long source = d_sssp_queue_1[source_index];
                unsigned long destination;
                // unsigned long index_counter = (id / 2) - d_sssp_output_frontier_offset[source_index];
                // unsigned long index_counter = curr_index - d_sssp_output_frontier_offset[source_index] - 1;

                // unsigned long edge_block_count = device_vertex_dictionary->vertex_adjacency[source]->edge_block_count;
                // edge_block_count -= ceil(double(index_counter) / EDGE_BLOCK_SIZE);
                // edge_block_count -= floor(double(index_counter) / EDGE_BLOCK_SIZE);
                // unsigned long edge_block_index = floor(double(index_counter) / EDGE_BLOCK_SIZE);
                // unsigned long edge_block_index = floor(double(index_counter) / EDGE_BLOCK_SIZE);
                unsigned long edge_entry_index = ((id) % thread_multiplier);
                unsigned long start_index = edge_entry_index * 10;
                unsigned long end_index = start_index + 10;
                // if(end_index > EDGE_BLOCK_SIZE)
                //     end_index = EDGE_BLOCK_SIZE;

                for(unsigned long index = start_index ; index < end_index ; index++) {

                    // printf("%lu ", root->edge_block_entry[j].destination_vertex);

                    if(root->edge_block_entry[index].destination_vertex == 0)
                        break;

                    unsigned long destination = root->edge_block_entry[index].destination_vertex - 1;

                    // for(unsigned long i = start_index ; i < end_index ; i++) {

                    // if(d_shortest_path[long(d_sssp_queue_1[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {
                    if(d_shortest_path[source] + 1 < d_shortest_path[destination]) {

                        // printf("Hit for source %lu and destination %lu\n", source, destination);

                        // atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[long(d_sssp_queue_1[id])] + 1);
                        atomicMin(&(d_shortest_path[destination]), d_shortest_path[source] + 1);

                        unsigned long long temp = atomicAdd(d_dp_thread_count, 1);

                        *(d_sssp_queue_2 + (temp)) = destination;


                    }

                    // }


                }

                if(curr_node->rptr != NULL) {
                    push(stack, &top, stack_size, curr_node->rptr);
                    top++;
                    // printf("ID %lu pushed %p to stack, top=%lu\n", id, curr_node->rptr, top);
                }
                curr_node = curr_node->lptr;

            }
            else {

                curr_node = pop(stack, &top);
                top--;
                // printf("ID %lu popped %p from the stack\n", id, curr_node);
            }

            // printf("ID %lu, curr_node is now %p\n", id, curr_node);

        }

        // sssp_preorder_traversal(root, d_shortest_path, source, d_sssp_queue_2, d_dp_thread_count);

    }

    // if(id == 0) {

    //     printf("Printing SSSP input frontier\n");
    //         for(unsigned long i = 0 ; i < d_prev_thread_count ; i++)
    //             printf("%lu ", long(d_sssp_queue_1[i]));
    //     printf("\nPrinting SSSP output frontier\n");
    //     // if(!type) {
    //         for(unsigned long i = 0 ; i < *d_dp_thread_count ; i++)
    //             printf("%lu ", long(d_sssp_queue_2[i]));
    //     // }
    //     // else {

    //     // }
    //     printf("\n\n");

    // }

}

__device__ unsigned long long device_binary_search(unsigned long long *input_array, unsigned long long key, unsigned long long size) {

    unsigned long start = 0;
    unsigned long end = size;
    unsigned long mid;

    while (start <= end) {

        mid = (start + end) / 2;

        // Check if x is present at mid
        if (input_array[mid] == key)
            return mid;

        // If x greater, ignore left half
        if (input_array[mid] < key)
            start = mid + 1;

            // If x is smaller, ignore right half
        else
            end = mid - 1;
    }

    // If we reach here, then element was not present
    return start - 1;

}

__global__ void sssp_kernel_EC(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned int *d_shortest_path, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long *d_TC_edge_vector, unsigned long *d_source_vector_1, unsigned long *d_mutex, unsigned long long *d_sssp_queue_1, unsigned long long *d_sssp_queue_2, unsigned long long *d_dp_thread_count, unsigned long long *d_sssp_output_frontier_offset) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id == 0) {

    //     printf("Current output frontier offset at kernel\n");
    //     for(unsigned long i = 0 ; i < d_prev_thread_count + 1 ; i++)
    //         printf("%lu ", d_sssp_output_frontier_offset[i]);
    //     printf("\n\n");

    // }

    // __syncthreads();

    if((id < d_sssp_output_frontier_offset[d_prev_thread_count])) {



        // if((!type)) {

        // unsigned long i = 0;
        // unsigned long source_index = 1;

        // while(id >= d_sssp_output_frontier_offset[source_index]) {

        //     // printf("ID is %lu, source_index is %lu, d_sssp_output_frontier_offset is %lu\n", id, source_index, d_sssp_output_frontier_offset[source_index]);

        //     source_index++;
        // }

        // source_index--;

        unsigned long source_index = 0;

        // if(id)
        source_index = device_binary_search(d_sssp_output_frontier_offset, id, d_prev_thread_count);

        // printf("ID is %lu, linear search index is %lu, binary search index is %lu\n", id, source_index, device_binary_search(d_sssp_output_frontier_offset, id, d_prev_thread_count));

        // unsigned long source_vertex = d_source_vector[id];
        // unsigned long index_counter = id - d_prefix_sum_edge_blocks[source_vertex];
        // unsigned long start_index = d_csr_offset[source_vertex] + (index_counter * EDGE_BLOCK_SIZE);
        // unsigned long end_index = start_index + EDGE_BLOCK_SIZE;

        // if(end_index > d_csr_offset[source_vertex + 1])
        //     end_index = d_csr_offset[source_vertex + 1];


        // if(start_index < end_index) {

        unsigned long source = d_sssp_queue_1[source_index];
        unsigned long destination;
        unsigned long index_counter = id - d_sssp_output_frontier_offset[source_index];
        unsigned long edge_entry_index = (index_counter) % EDGE_BLOCK_SIZE;
        unsigned long edge_block_index = floor(double(index_counter) / EDGE_BLOCK_SIZE);

        // unsigned long start_index = floor(double(index_counter) / 8);
        // unsigned long end_index = floor(double(index_counter) / 8) + 8;

        // if(end_index > EDGE_BLOCK_SIZE)
        //     end_index = EDGE_BLOCK_SIZE;

        unsigned long bit_string = bit_string_lookup[edge_block_index];
        struct edge_block *root = device_vertex_dictionary->edge_block_address[source];

        // printf("id=%lu, source=%lu, index_counter=%lu, edge_block_index=%lu, root=%p, start_index=%lu, end_index=%lu\n", id, source, index_counter, edge_block_index, root, start_index, end_index);


        if(root != NULL) {

            root = traverse_bit_string(root, bit_string);

            destination = root->edge_block_entry[edge_entry_index].destination_vertex - 1;


        }

        else
            printf("Hit here\n");

        // }

        // }


        // printf("ID is %lu, source is %lu and destination is %lu, destination_index is %lu, source_index is %lu\n", id, source, destination, d_csr_offset[source] + id - d_sssp_output_frontier_offset[source_index], source_index);

        // printf("ID is %lu, source is %lu and destination is %lu, destination_index is %lu, source_index is %lu, frontier offset is %lu\n", id, source, destination, d_csr_offset[source] + id - d_sssp_output_frontier_offset[source_index], source_index, d_sssp_output_frontier_offset[source_index]);

        // if((d_csr_offset[source] + id - d_sssp_output_frontier_offset[source_index]) == 30491458)
        //     printf("ID is %lu, source is %lu and destination is %lu, destination_index is %lu, source_index is %lu, csr_offset[source] is %lu, csr_edge[last] is %lu\n", id, source, destination, d_csr_offset[source] + id - d_sssp_output_frontier_offset[source_index], source_index, d_csr_offset[source], d_TC_edge_vector[30491457]);

        // if((source > 550000) || (destination > 550000) || (source_index > 100000000) || ((d_csr_offset[source] + id - d_sssp_output_frontier_offset[source_index]) > 100000000))
        //     printf("Hit error\n");

        // if(destination == 0)
        //     printf("ID is %lu, source is %lu and destination is %lu, destination_index is %lu, source_index is %lu\n", id, source, destination, d_csr_offset[source] + id - d_sssp_output_frontier_offset[source_index], source_index);

        // for(unsigned long i = start_index ; i < end_index ; i++) {
        // // for(unsigned long i = 0 ; i < root->active_edge_count ; i++) {

        //     if(root->edge_block_entry[i].destination_vertex == 0)
        //         break;
        //     else {

        //         // for(unsigned long j = start_index ; j < end_index ; j++) {

        //             // if((root == NULL) || ((root->edge_block_entry[i].destination_vertex - 1) >= (vertex_size)) || ((root->edge_block_entry[i].destination_vertex - 1) < 0))
        //             //     printf("Hit error at %lu\n", root->edge_block_entry[i].destination_vertex - 1);

        //             // page_factor = page_factor * 2;

        //             // float kochappi = 2;

        //             // d_pageRankVector_2[root->edge_block_entry[i].destination_vertex - 1] += page_factor;

        //         destination = root->edge_block_entry[i].destination_vertex - 1;


        //         if(d_shortest_path[source] + 1 < d_shortest_path[destination]) {

        //             // printf("ID is %lu, destination is %lu, hit\n", id, destination);

        //             unsigned int tempX = atomicMin(&(d_shortest_path[destination]), d_shortest_path[source] + 1);

        //             // if(tempX == d_shortest_path[source] + 1) {

        //                 unsigned long long temp = atomicAdd(d_dp_thread_count, 1);

        //                 *(d_sssp_queue_2 + (temp)) = destination;
        //             // }
        //         }

        //             // if(root->edge_block_entry[i].destination_vertex == d_csr_edges[j]) {
        //             //     root->edge_block_entry[i].destination_vertex = INFTY;
        //             //     break;
        //             // }

        //         }

        //     // }

        // }


        if(d_shortest_path[source] + 1 < d_shortest_path[destination]) {

            // printf("ID is %lu, destination is %lu, hit\n", id, destination);

            unsigned int tempX = atomicMin(&(d_shortest_path[destination]), d_shortest_path[source] + 1);

            // if(tempX == d_shortest_path[source] + 1) {

            unsigned long long temp = atomicAdd(d_dp_thread_count, 1);

            *(d_sssp_queue_2 + (temp)) = destination;
            // }
        }

        // }

        // else {

        //         // unsigned long source_index = 1;

        //         // while(id >= d_sssp_output_frontier_offset[source_index]) {

        //         //     // printf("ID is %lu, source_index is %lu, d_sssp_output_frontier_offset is %lu\n", id, source_index, d_sssp_output_frontier_offset[source_index]);

        //         //     source_index++;
        //         // }

        //         // source_index--;

        //         unsigned long source_index = 0;

        //         // if(id)
        //             source_index = device_binary_search(d_sssp_output_frontier_offset, id, d_prev_thread_count);

        //         // printf("ID is %lu, linear search index is %lu, binary search index is %lu\n", id, source_index, device_binary_search(d_sssp_output_frontier_offset, id, d_prev_thread_count));

        //         unsigned long source = d_sssp_queue_2[source_index];
        //         unsigned long destination = d_TC_edge_vector[d_csr_offset[source] + id - d_sssp_output_frontier_offset[source_index]];

        //         // printf("ID is %lu, source is %lu and destination is %lu, destination_index is %lu, source_index is %lu\n", id, source, destination, d_csr_offset[source] + id - d_sssp_output_frontier_offset[source_index], source_index);
        //         // printf("ID is %lu, source is %lu and destination is %lu, destination_index is %lu, source_index is %lu, frontier offset is %lu\n", id, source, destination, d_csr_offset[source] + id - d_sssp_output_frontier_offset[source_index], source_index, d_sssp_output_frontier_offset[source_index]);

        //         // csr offset giving wrong value!!!!
        //         // if((d_csr_offset[source] + id - d_sssp_output_frontier_offset[source_index]) == 30491458)
        //         //     printf("ID is %lu, source is %lu and destination is %lu, destination_index is %lu, source_index is %lu, csr_offset[source] is %lu, csr_offset[source - 1] is %lu, csr_offset[540K] is %lu, csr_edge[last] is %lu, frontier offset is %lu\n", id, source, destination, d_csr_offset[source] + id - d_sssp_output_frontier_offset[source_index], source_index, d_csr_offset[source], d_csr_offset[source - 1], d_csr_offset[540485], d_TC_edge_vector[30491457], d_sssp_output_frontier_offset[source_index]);


        //         // if((source > 550000) || (destination > 550000) || (source_index > 100000000) || ((d_csr_offset[source] + id - d_sssp_output_frontier_offset[source_index]) > 100000000))
        //         //     printf("Hit error\n");

        //         // if(destination == 0)
        //         //     printf("ID is %lu, source is %lu and destination is %lu, destination_index is %lu, source_index is %lu\n", id, source, destination, d_csr_offset[source] + id - d_sssp_output_frontier_offset[source_index], source_index);

        //         if(d_shortest_path[source] + 1 < d_shortest_path[destination]) {

        //             // printf("ID is %lu, destination is %lu, hit\n", id, destination);

        //             unsigned int tempX = atomicMin(&(d_shortest_path[destination]), d_shortest_path[source] + 1);


        //             // if(tempX == d_shortest_path[source] + 1) {

        //                 unsigned long long temp = atomicAdd(d_dp_thread_count, 1);

        //                 *(d_sssp_queue_1 + (temp)) = destination;

        //             // }

        //         }


        // }

    }

    // if(id == 0) {

    //     printf("Printing SSSP queue\n");
    //     if(!type) {
    //         for(unsigned long long i = 0 ; i < *d_dp_thread_count ; i++)
    //             printf("%lu ", long(d_sssp_queue_2[i]));
    //     }
    //     else {
    //         for(unsigned long long i = 0 ; i < *d_dp_thread_count ; i++)
    //             printf("%lu ", long(d_sssp_queue_1[i]));
    //     }
    //     printf("\n\n");

    // }

}

__global__ void sssp_kernel_EC_load_balanced(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long edge_size, unsigned long batch_size, unsigned int *d_shortest_path, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long *d_TC_edge_vector, unsigned long *d_source_vector_1, unsigned long *d_mutex, unsigned long long *d_sssp_queue_1, unsigned long long *d_sssp_queue_2, unsigned long long *d_dp_thread_count, unsigned long long *d_sssp_output_frontier_offset) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id == 0) {

    //     printf("Current output frontier offset at kernel\n");
    //     for(unsigned long i = 0 ; i < d_prev_thread_count + 1 ; i++)
    //         printf("%lu ", d_sssp_output_frontier_offset[i]);
    //     printf("\n\n");

    // }

    // __syncthreads();

    if((id < ceil((double)d_sssp_output_frontier_offset[d_prev_thread_count] / SSSP_LOAD_FACTOR))) {

        unsigned long start_index = id * SSSP_LOAD_FACTOR;
        unsigned long end_index = (id + 1) * SSSP_LOAD_FACTOR;
        if(end_index > d_sssp_output_frontier_offset[d_prev_thread_count])
            end_index = d_sssp_output_frontier_offset[d_prev_thread_count];
        unsigned long curr_index = start_index;

        // printf("ID=%lu, start_index=%lu, end_index=%lu\n", id, start_index, end_index);

        do {

            unsigned long source_index = 0;
            source_index = device_binary_search(d_sssp_output_frontier_offset, curr_index, d_prev_thread_count);

            unsigned long source = d_sssp_queue_1[source_index];
            unsigned long destination;
            unsigned long index_counter = curr_index - d_sssp_output_frontier_offset[source_index];
            // unsigned long index_counter = curr_index - d_sssp_output_frontier_offset[source_index] - 1;

            unsigned long edge_block_count = device_vertex_dictionary->edge_block_count[source];
            // edge_block_count -= ceil(double(index_counter) / EDGE_BLOCK_SIZE);
            // edge_block_count -= floor(double(index_counter) / EDGE_BLOCK_SIZE);
            // unsigned long edge_block_index = floor(double(index_counter) / EDGE_BLOCK_SIZE);
            unsigned long edge_block_index = floor(double(index_counter) / EDGE_BLOCK_SIZE);
            unsigned long edge_entry_index = (index_counter) % EDGE_BLOCK_SIZE;
            // ok till here

            // printf("ID=%lu, source=%lu, start_index=%lu, end_index=%lu, index_counter=%lu, edge_block_count=%lu, edge_block_index=%lu, curr_index=%lu\n", id, source, start_index, end_index, index_counter, edge_block_count, edge_block_index, curr_index);

            for(unsigned long i = edge_block_index ; (i < edge_block_count) && (curr_index < end_index) ; i++) {

                // unsigned long edge_block_index = floor(double(index_counter) / EDGE_BLOCK_SIZE);
                // unsigned long edge_block_index = i;

                // printf("ID=%lu, start_index=%lu, end_index=%lu, index_counter=%lu, edge_block_count\n", id, start_index, end_index, index_counter);

                // if(end_index > EDGE_BLOCK_SIZE)
                //     end_index = EDGE_BLOCK_SIZE;

                unsigned long bit_string = bit_string_lookup[i];
                struct edge_block *root = device_vertex_dictionary->edge_block_address[source];

                // printf("id=%lu, source=%lu, index_counter=%lu, edge_block_index=%lu, root=%p, start_index=%lu, end_index=%lu\n", id, source, index_counter, edge_block_index, root, start_index, end_index);


                if(root != NULL) {

                    root = traverse_bit_string(root, bit_string);
                    // destination = root->edge_block_entry[edge_entry_index].destination_vertex - 1;

                }

                else
                    printf("Hit here\n");

                for(unsigned long index = edge_entry_index ; (index < EDGE_BLOCK_SIZE) && (curr_index < end_index) ; index++) {

                    // printf("%lu ", root->edge_block_entry[j].destination_vertex);

                    if(root->edge_block_entry[index].destination_vertex == 0)
                        break;

                    destination = root->edge_block_entry[index].destination_vertex - 1;
                    curr_index++;

                    // for(unsigned long i = start_index ; i < end_index ; i++) {

                    // if(d_shortest_path[long(d_sssp_queue_1[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {
                    if(d_shortest_path[source] + 1 < d_shortest_path[destination]) {

                        // printf("Hit for source %lu and destination %lu\n", source, destination);

                        // atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[long(d_sssp_queue_1[id])] + 1);
                        atomicMin(&(d_shortest_path[destination]), d_shortest_path[source] + 1);

                        unsigned long long temp = atomicAdd(d_dp_thread_count, 1);

                        *(d_sssp_queue_2 + (temp)) = destination;


                    }

                    // }


                }

                edge_entry_index = 0;

            }

        } while(curr_index < end_index);

    }

    // if(id == 0) {

    //     printf("Printing SSSP queue\n");
    //     if(!type) {
    //         for(unsigned long long i = 0 ; i < *d_dp_thread_count ; i++)
    //             printf("%lu ", long(d_sssp_queue_2[i]));
    //     }
    //     else {
    //         for(unsigned long long i = 0 ; i < *d_dp_thread_count ; i++)
    //             printf("%lu ", long(d_sssp_queue_1[i]));
    //     }
    //     printf("\n\n");

    // }

}

__global__ void sssp_kernel_EBC(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long edge_size, unsigned long batch_size, unsigned int *d_shortest_path, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long *d_TC_edge_vector, unsigned long *d_source_vector_1, unsigned long *d_mutex, unsigned long long *d_sssp_queue_1, unsigned long long *d_sssp_queue_2, unsigned long long *d_dp_thread_count, unsigned long long *d_sssp_output_frontier_offset, unsigned long thread_multiplier) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id == 0) {

    //     printf("Current output frontier offset at kernel\n");
    //     for(unsigned long i = 0 ; i < d_prev_thread_count + 1 ; i++)
    //         printf("%lu ", d_sssp_output_frontier_offset[i]);
    //     printf("\n\n");

    // }

    // __syncthreads();

    if((id < (d_sssp_output_frontier_offset[d_prev_thread_count] * thread_multiplier))) {

        // unsigned long start_index = id * SSSP_LOAD_FACTOR;
        // unsigned long end_index = (id + 1) * SSSP_LOAD_FACTOR;
        // if(end_index > d_sssp_output_frontier_offset[d_prev_thread_count])
        //     end_index = d_sssp_output_frontier_offset[d_prev_thread_count];
        // unsigned long curr_index = start_index;

        // printf("ID=%lu, start_index=%lu, end_index=%lu\n", id, start_index, end_index);

        // do {

        unsigned long source_index = 0;
        source_index = device_binary_search(d_sssp_output_frontier_offset, id / thread_multiplier, d_prev_thread_count);

        unsigned long source = d_sssp_queue_1[source_index];
        unsigned long destination;
        unsigned long index_counter = (id / thread_multiplier) - d_sssp_output_frontier_offset[source_index];
        // unsigned long index_counter = curr_index - d_sssp_output_frontier_offset[source_index] - 1;

        // unsigned long edge_block_count = device_vertex_dictionary->vertex_adjacency[source]->edge_block_count;
        // edge_block_count -= ceil(double(index_counter) / EDGE_BLOCK_SIZE);
        // edge_block_count -= floor(double(index_counter) / EDGE_BLOCK_SIZE);
        // unsigned long edge_block_index = floor(double(index_counter) / EDGE_BLOCK_SIZE);
        // unsigned long edge_block_index = floor(double(index_counter) / EDGE_BLOCK_SIZE);
        unsigned long edge_entry_index = ((id) % thread_multiplier);
        unsigned long start_index = edge_entry_index * 11;
        unsigned long end_index = start_index + 11;
        // if(end_index > EDGE_BLOCK_SIZE)
        //     end_index = EDGE_BLOCK_SIZE;
        // ok till here

        // printf("ID=%lu, source=%lu, start_index=%lu, end_index=%lu, index_counter=%lu, edge_block_count=%lu, edge_block_index=%lu, curr_index=%lu\n", id, source, start_index, end_index, index_counter, edge_block_count, edge_block_index, curr_index);

        // for(unsigned long i = edge_block_index ; (i < edge_block_count) && (curr_index < end_index) ; i++) {

        // unsigned long edge_block_index = floor(double(index_counter) / EDGE_BLOCK_SIZE);
        // unsigned long edge_block_index = i;

        // printf("ID=%lu, start_index=%lu, end_index=%lu, index_counter=%lu, edge_block_count\n", id, start_index, end_index, index_counter);

        // if(end_index > EDGE_BLOCK_SIZE)
        //     end_index = EDGE_BLOCK_SIZE;

        unsigned long bit_string = bit_string_lookup[index_counter];
        struct edge_block *root = device_vertex_dictionary->edge_block_address[source];

        // printf("id=%lu, source=%lu, index_counter=%lu, edge_block_index=%lu, root=%p, start_index=%lu, end_index=%lu\n", id, source, index_counter, edge_block_index, root, start_index, end_index);


        if(root != NULL) {

            root = traverse_bit_string(root, bit_string);
            // destination = root->edge_block_entry[edge_entry_index].destination_vertex - 1;

        }

        else
            goto end;
        // printf("Hit here, id=%lu, source=%lu, index_counter=%lu, root=%p\n", id, source, index_counter, root);

        for(unsigned long index = start_index ; (index < end_index) ; index++) {

            // printf("%lu ", root->edge_block_entry[j].destination_vertex);

            if(root->edge_block_entry[index].destination_vertex == 0)
                break;

            destination = root->edge_block_entry[index].destination_vertex - 1;
            // curr_index++;
            // printf("id=%lu, source=%lu, destination=%lu, index_counter=%lu, root=%p\n", id, source, destination, index_counter, root);

            // for(unsigned long i = start_index ; i < end_index ; i++) {

            // if(d_shortest_path[long(d_sssp_queue_1[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {
            if((d_shortest_path[source] + 1 < d_shortest_path[destination])) {

                // d_mutex[destination] = 1;

                // printf("Hit for source %lu and destination %lu\n", source, destination);

                // atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[long(d_sssp_queue_1[id])] + 1);
                atomicMin(&(d_shortest_path[destination]), d_shortest_path[source] + 1);

                unsigned long long temp = atomicAdd(d_dp_thread_count, 1);

                *(d_sssp_queue_2 + (temp)) = destination;

            }

            // }


        }

        // edge_entry_index = 0;

        // }

        // } while(curr_index < end_index);

    }

    end:

    // if(id == 0) {

    // printf("Printing SSSP input frontier\n");
    //     for(unsigned long i = 0 ; i < d_prev_thread_count ; i++)
    //         printf("%lu ", long(d_sssp_queue_1[i]));
    // printf("\nPrinting SSSP output frontier\n");
    // // if(!type) {
    //     for(unsigned long i = 0 ; i < *d_dp_thread_count ; i++)
    //         printf("%lu ", long(d_sssp_queue_2[i]));
    // }
    // else {

    // }
    // printf("\n\n");

    // }

}

__global__ void sssp_kernel_multiple_worklists_0to15(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned int *d_shortest_path, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long *d_TC_edge_vector, unsigned long *d_source_vector_1, unsigned long *d_mutex, unsigned long long *d_sssp_queue_1, unsigned long long *d_sssp_queue_2, unsigned long long *d_dp_thread_count, unsigned long long *d_sssp_queue_1_16to31, unsigned long long *d_sssp_queue_2_16to31, unsigned long long *d_dp_thread_count_16to31) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("Hit\n");
    // printf("ID=%lu, count=%lu\n", id, int(d_dp_thread_count));

    if((id < d_prev_thread_count)) {

        // printf("ID = %lu,\n", id);
        // if((id < vertex_size)) {
        // if((id < vertex_size) && (d_shortest_path[id] != UINT_MAX)) {
        // printf("Hit Inner\n");

        // unsigned long start_index = d_csr_offset[int(d_sssp_queue_1[id])];
        // unsigned long end_index = d_csr_offset[int(d_sssp_queue_1[id]) + 1];

        if((!type)) {

            // unsigned long start_index = d_csr_offset[long(d_sssp_queue_1[id])];
            // unsigned long end_index = d_csr_offset[long(d_sssp_queue_1[id]) + 1];

            unsigned long start_index = d_csr_offset[(d_sssp_queue_1[id])];
            unsigned long end_index = d_csr_offset[(d_sssp_queue_1[id]) + 1];

            // printf("start_index\n");

            // printf("Beginning ID %lu, source %lu\n", id, d_sssp_queue_2[id]);


            for(unsigned long i = start_index ; i < end_index ; i++) {

                // if(d_shortest_path[long(d_sssp_queue_1[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {
                if(d_shortest_path[(d_sssp_queue_1[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {

                    // printf("Hit\n");

                    // atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[long(d_sssp_queue_1[id])] + 1);
                    atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[(d_sssp_queue_1[id])] + 1);

                    // unsigned long temp = d_TC_edge_vector[i];

                    // if(!type)
                    // atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_edge_vector[i]);
                    // atomicAdd(d_sssp_queue_1 + int(atomicAdd(&d_dp_thread_count), 1), d_TC_edge_vector[i]);
                    // else
                    // atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_edge_vector[i]);

                    // if(d_search_flag != 1)
                    //     d_search_flag = 1;
                    // atomicOp below
                    // d_dp_thread_count += 1;

                    // unsigned long long temp = atomicAdd(&d_dp_thread_count, 1);
                    // printf("Hit 2 with thread count %lu\n", d_dp_thread_count);

                    unsigned long destination_degree = d_csr_offset[d_TC_edge_vector[i] + 1] - d_csr_offset[d_TC_edge_vector[i]];

                    if(destination_degree < 16) {

                        // printf("Hit\n");

                        unsigned long long temp = atomicAdd(d_dp_thread_count, 1);

                        // cudaDeviceSynchronize();
                        // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
                        *(d_sssp_queue_2 + (temp)) = d_TC_edge_vector[i];

                    }

                    else {

                        unsigned long long temp = atomicAdd(d_dp_thread_count_16to31, 1);

                        // cudaDeviceSynchronize();
                        // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
                        *(d_sssp_queue_2_16to31 + (temp)) = d_TC_edge_vector[i];

                    }


                    // unsigned long long temp = atomicAdd(d_dp_thread_count, 1);

                    // // cudaDeviceSynchronize();
                    // // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
                    // *(d_sssp_queue_2 + (temp)) = d_TC_edge_vector[i];
                    // printf("ID %lu, source %lu ,thread_counter %lu, d_TC_edge_vector is %lu, d_sssp_queue_2 is %p with vertex %lu\n", id, d_sssp_queue_1[id], long(d_dp_thread_count), d_TC_edge_vector[i], d_sssp_queue_2 + long(d_dp_thread_count), long(d_sssp_queue_2[long(d_dp_thread_count)]));
                    // printf("ID %lu, source %lu ,thread_counter %lu, d_TC_edge_vector is %lu, d_sssp_queue_2 is %p with vertex %lu, destination degree is %lu\n", id, d_sssp_queue_1[id], (temp), d_TC_edge_vector[i], d_sssp_queue_2 + (temp) - 1, (d_sssp_queue_2[(temp) - 1]), destination_degree);

                    //     atomicAdd(d_sssp_queue_1 + int(d_dp_thread_count), d_TC_edge_vector[i]);
                    // else
                    //     atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_ed

                }



            }

        }

        else {

            // unsigned long start_index = d_csr_offset[long(d_sssp_queue_1[id])];
            // unsigned long end_index = d_csr_offset[long(d_sssp_queue_1[id]) + 1];

            unsigned long start_index = d_csr_offset[(d_sssp_queue_2[id])];
            unsigned long end_index = d_csr_offset[(d_sssp_queue_2[id]) + 1];

            // printf("start_index\n");
            // printf("Beginning ID %lu, source %lu\n", id, d_sssp_queue_2[id]);

            for(unsigned long i = start_index ; i < end_index ; i++) {

                // if(d_shortest_path[long(d_sssp_queue_1[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {
                if(d_shortest_path[(d_sssp_queue_2[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {

                    // printf("Hit\n");

                    // atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[long(d_sssp_queue_1[id])] + 1);
                    atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[(d_sssp_queue_2[id])] + 1);

                    // unsigned long temp = d_TC_edge_vector[i];

                    // if(!type)
                    // atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_edge_vector[i]);
                    // atomicAdd(d_sssp_queue_1 + int(atomicAdd(&d_dp_thread_count), 1), d_TC_edge_vector[i]);
                    // else
                    // atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_edge_vector[i]);

                    // if(d_search_flag != 1)
                    //     d_search_flag = 1;
                    // atomicOp below
                    // d_dp_thread_count += 1;

                    // atomicAdd(&d_dp_thread_count, 1);
                    // // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
                    // *(d_sssp_queue_1 + (d_dp_thread_count) - 1) = d_TC_edge_vector[i];

                    unsigned long destination_degree = d_csr_offset[d_TC_edge_vector[i] + 1] - d_csr_offset[d_TC_edge_vector[i]];

                    if(destination_degree < 16) {

                        unsigned long long temp = atomicAdd(d_dp_thread_count, 1);

                        // cudaDeviceSynchronize();
                        // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
                        *(d_sssp_queue_1 + (temp)) = d_TC_edge_vector[i];

                    }

                    else {

                        unsigned long long temp = atomicAdd(d_dp_thread_count_16to31, 1);

                        // cudaDeviceSynchronize();
                        // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
                        *(d_sssp_queue_1_16to31 + (temp)) = d_TC_edge_vector[i];

                    }

                    // unsigned long long temp = atomicAdd(&d_dp_thread_count, 1);
                    // unsigned long long temp = atomicAdd(d_dp_thread_count, 1);
                    // // cudaDeviceSynchronize();

                    // // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
                    // *(d_sssp_queue_1 + (temp) ) = d_TC_edge_vector[i];

                    // printf("ID %lu has thread_counter %lu, d_TC_edge_vector is %lu, d_sssp_queue_2 is %p with vertex %lu\n", id, long(d_dp_thread_count), d_TC_edge_vector[i], d_sssp_queue_2 + long(d_dp_thread_count), long(d_sssp_queue_2[long(d_dp_thread_count)]));
                    // printf("ID %lu, source %lu ,thread_counter %lu, d_TC_edge_vector is %lu, d_sssp_queue_1 is %p with vertex %lu\n", id, d_sssp_queue_2[id], (temp), d_TC_edge_vector[i], d_sssp_queue_1 + (temp) - 1, (d_sssp_queue_1[(temp) - 1]));

                    //     atomicAdd(d_sssp_queue_1 + int(d_dp_thread_count), d_TC_edge_vector[i]);
                    // else
                    //     atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_ed

                }



            }

        }

    }

    // if(id == 0) {

    //     printf("Printing SSSP queue\n");
    //     if(!type) {
    //         for(unsigned long i = 0 ; i < *d_dp_thread_count ; i++)
    //             printf("%lu ", long(d_sssp_queue_2[i]));
    //     }
    //     else {
    //         for(unsigned long i = 0 ; i < *d_dp_thread_count ; i++)
    //             printf("%lu ", long(d_sssp_queue_1[i]));
    //     }
    //     printf("\n\n");

    // }

}

__global__ void sssp_kernel_multiple_worklists_16to31(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned int *d_shortest_path, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long *d_TC_edge_vector, unsigned long *d_source_vector_1, unsigned long *d_mutex, unsigned long long *d_sssp_queue_1, unsigned long long *d_sssp_queue_2, unsigned long long *d_dp_thread_count, unsigned long long *d_sssp_queue_1_16to31, unsigned long long *d_sssp_queue_2_16to31, unsigned long long *d_dp_thread_count_16to31) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("Hit\n");
    // printf("ID=%lu, count=%lu\n", id, int(d_dp_thread_count));

    if((id < d_prev_thread_count_16to31)) {

        // printf("ID = %lu,\n", id);
        // if((id < vertex_size)) {
        // if((id < vertex_size) && (d_shortest_path[id] != UINT_MAX)) {
        // printf("Hit Inner\n");

        // unsigned long start_index = d_csr_offset[int(d_sssp_queue_1[id])];
        // unsigned long end_index = d_csr_offset[int(d_sssp_queue_1[id]) + 1];

        if((!type)) {

            // unsigned long start_index = d_csr_offset[long(d_sssp_queue_1[id])];
            // unsigned long end_index = d_csr_offset[long(d_sssp_queue_1[id]) + 1];

            unsigned long start_index = d_csr_offset[(d_sssp_queue_1_16to31[id])];
            unsigned long end_index = d_csr_offset[(d_sssp_queue_1_16to31[id]) + 1];

            // printf("start_index\n");

            // printf("Beginning ID %lu, source %lu\n", id, d_sssp_queue_2[id]);


            for(unsigned long i = start_index ; i < end_index ; i++) {

                // if(d_shortest_path[long(d_sssp_queue_1[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {
                if(d_shortest_path[(d_sssp_queue_1_16to31[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {

                    // printf("Hit\n");

                    // atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[long(d_sssp_queue_1[id])] + 1);
                    atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[(d_sssp_queue_1_16to31[id])] + 1);

                    // unsigned long temp = d_TC_edge_vector[i];

                    // if(!type)
                    // atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_edge_vector[i]);
                    // atomicAdd(d_sssp_queue_1 + int(atomicAdd(&d_dp_thread_count), 1), d_TC_edge_vector[i]);
                    // else
                    // atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_edge_vector[i]);

                    // if(d_search_flag != 1)
                    //     d_search_flag = 1;
                    // atomicOp below
                    // d_dp_thread_count += 1;

                    // unsigned long long temp = atomicAdd(&d_dp_thread_count, 1);
                    // printf("Hit 2 with thread count %lu\n", d_dp_thread_count);

                    unsigned long destination_degree = d_csr_offset[d_TC_edge_vector[i] + 1] - d_csr_offset[d_TC_edge_vector[i]];

                    if(destination_degree < 16) {

                        unsigned long long temp = atomicAdd(d_dp_thread_count, 1);

                        // cudaDeviceSynchronize();
                        // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
                        *(d_sssp_queue_2 + (temp)) = d_TC_edge_vector[i];

                    }

                    else {

                        unsigned long long temp = atomicAdd(d_dp_thread_count_16to31, 1);

                        // cudaDeviceSynchronize();
                        // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
                        *(d_sssp_queue_2_16to31 + (temp)) = d_TC_edge_vector[i];

                    }


                    // unsigned long long temp = atomicAdd(d_dp_thread_count, 1);

                    // // cudaDeviceSynchronize();
                    // // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
                    // *(d_sssp_queue_2 + (temp)) = d_TC_edge_vector[i];
                    // printf("ID %lu, source %lu ,thread_counter %lu, d_TC_edge_vector is %lu, d_sssp_queue_2 is %p with vertex %lu\n", id, d_sssp_queue_1[id], long(d_dp_thread_count), d_TC_edge_vector[i], d_sssp_queue_2 + long(d_dp_thread_count), long(d_sssp_queue_2[long(d_dp_thread_count)]));
                    // printf("ID %lu, source %lu ,thread_counter %lu, d_TC_edge_vector is %lu, d_sssp_queue_2 is %p with vertex %lu\n", id, d_sssp_queue_1[id], (temp), d_TC_edge_vector[i], d_sssp_queue_2 + (temp) - 1, (d_sssp_queue_2[(temp) - 1]));

                    //     atomicAdd(d_sssp_queue_1 + int(d_dp_thread_count), d_TC_edge_vector[i]);
                    // else
                    //     atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_ed

                }



            }

        }

        else {

            // unsigned long start_index = d_csr_offset[long(d_sssp_queue_1[id])];
            // unsigned long end_index = d_csr_offset[long(d_sssp_queue_1[id]) + 1];

            unsigned long start_index = d_csr_offset[(d_sssp_queue_2_16to31[id])];
            unsigned long end_index = d_csr_offset[(d_sssp_queue_2_16to31[id]) + 1];

            // printf("start_index\n");
            // printf("Beginning ID %lu, source %lu\n", id, d_sssp_queue_2[id]);

            for(unsigned long i = start_index ; i < end_index ; i++) {

                // if(d_shortest_path[long(d_sssp_queue_1[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {
                if(d_shortest_path[(d_sssp_queue_2_16to31[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {

                    // printf("Hit\n");

                    // atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[long(d_sssp_queue_1[id])] + 1);
                    atomicMin(&(d_shortest_path[d_TC_edge_vector[i]]), d_shortest_path[(d_sssp_queue_2_16to31[id])] + 1);

                    // unsigned long temp = d_TC_edge_vector[i];

                    // if(!type)
                    // atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_edge_vector[i]);
                    // atomicAdd(d_sssp_queue_1 + int(atomicAdd(&d_dp_thread_count), 1), d_TC_edge_vector[i]);
                    // else
                    // atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_edge_vector[i]);

                    // if(d_search_flag != 1)
                    //     d_search_flag = 1;
                    // atomicOp below
                    // d_dp_thread_count += 1;

                    // atomicAdd(&d_dp_thread_count, 1);
                    // // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
                    // *(d_sssp_queue_1 + (d_dp_thread_count) - 1) = d_TC_edge_vector[i];

                    unsigned long destination_degree = d_csr_offset[d_TC_edge_vector[i] + 1] - d_csr_offset[d_TC_edge_vector[i]];

                    if(destination_degree < 16) {

                        unsigned long long temp = atomicAdd(d_dp_thread_count, 1);

                        // cudaDeviceSynchronize();
                        // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
                        *(d_sssp_queue_1 + (temp)) = d_TC_edge_vector[i];

                    }

                    else {

                        unsigned long long temp = atomicAdd(d_dp_thread_count_16to31, 1);

                        // cudaDeviceSynchronize();
                        // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
                        *(d_sssp_queue_1_16to31 + (temp)) = d_TC_edge_vector[i];

                    }

                    // unsigned long long temp = atomicAdd(&d_dp_thread_count, 1);
                    // unsigned long long temp = atomicAdd(d_dp_thread_count, 1);
                    // // cudaDeviceSynchronize();

                    // // *(d_sssp_queue_2 + long(d_dp_thread_count) - 1) = float(d_TC_edge_vector[i]);
                    // *(d_sssp_queue_1 + (temp) ) = d_TC_edge_vector[i];

                    // printf("ID %lu has thread_counter %lu, d_TC_edge_vector is %lu, d_sssp_queue_2 is %p with vertex %lu\n", id, long(d_dp_thread_count), d_TC_edge_vector[i], d_sssp_queue_2 + long(d_dp_thread_count), long(d_sssp_queue_2[long(d_dp_thread_count)]));
                    // printf("ID %lu, source %lu ,thread_counter %lu, d_TC_edge_vector is %lu, d_sssp_queue_1 is %p with vertex %lu\n", id, d_sssp_queue_2[id], (temp), d_TC_edge_vector[i], d_sssp_queue_1 + (temp) - 1, (d_sssp_queue_1[(temp) - 1]));

                    //     atomicAdd(d_sssp_queue_1 + int(d_dp_thread_count), d_TC_edge_vector[i]);
                    // else
                    //     atomicAdd(d_sssp_queue_2 + int(d_dp_thread_count), d_TC_ed

                }



            }

        }

    }

    // if(id == 0) {

    //     printf("Printing SSSP queue\n");
    //     if(!type) {
    //         for(unsigned long i = 0 ; i < *d_dp_thread_count ; i++)
    //             printf("%lu ", long(d_sssp_queue_2[i]));
    //     }
    //     else {
    //         for(unsigned long i = 0 ; i < *d_dp_thread_count ; i++)
    //             printf("%lu ", long(d_sssp_queue_1[i]));
    //     }
    //     printf("\n\n");

    // }

}

__global__ void sssp_kernel_postprocess(unsigned long long *d_dp_thread_count) {

    d_prev_thread_count = *d_dp_thread_count;

    // printf("Current frontier vertex count is %lu\n", *d_dp_thread_count);

    // if(type)
    //     type = 0;
    // else
    //     type = 1;

}

__global__ void sssp_kernel_postprocess(unsigned long long *d_dp_thread_count, unsigned long long *d_dp_thread_count_16to31, unsigned long long *d_sssp_queue_1, unsigned long long *d_sssp_queue_2, unsigned long long *d_sssp_queue_1_16to31, unsigned long long *d_sssp_queue_2_16to31) {

    d_prev_thread_count = *d_dp_thread_count;
    d_prev_thread_count_16to31 = *d_dp_thread_count_16to31;

    // printf("Current frontier vertex count is %lu\n", *d_dp_thread_count);

    // printf("Printing SSSP queue group 0-15\n");
    // if(!type) {
    //     for(unsigned long i = 0 ; i < *d_dp_thread_count ; i++)
    //         printf("%lu ", long(d_sssp_queue_2[i]));
    // }
    // else {
    //     for(unsigned long i = 0 ; i < *d_dp_thread_count ; i++)
    //         printf("%lu ", long(d_sssp_queue_1[i]));
    // }
    // printf("\nPrinting SSSP queue group 16-31\n");
    // if(!type) {
    //     for(unsigned long i = 0 ; i < *d_dp_thread_count_16to31 ; i++)
    //         printf("%lu ", long(d_sssp_queue_2_16to31[i]));
    // }
    // else {
    //     for(unsigned long i = 0 ; i < *d_dp_thread_count_16to31 ; i++)
    //         printf("%lu ", long(d_sssp_queue_1_16to31[i]));
    // }
    // printf("\n\n");

    if(type)
        type = 0;
    else
        type = 1;



}


// __global__ void sssp_kernel_master(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned int *d_shortest_path, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long *d_TC_edge_vector, unsigned long *d_source_vector_1, float *d_mutex, float *d_sssp_queue_1, float *d_sssp_queue_2) {
// __global__ void sssp_kernel_master(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned int *d_shortest_path, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector, unsigned long *d_TC_edge_vector, unsigned long *d_source_vector_1, float *d_mutex, unsigned long long *d_sssp_queue_1, unsigned long long *d_sssp_queue_2) {

//     // init with source as vertex 1
//     // d_search_flag = 0;
//     d_shortest_path[0] = 0;
//     d_sssp_queue_1[0] = 0;
//     unsigned long iterations = 0;

//     do {

//         // d_search_flag = 0;
//         // printf("Iteration #%lu, thread_count=%lu\n", iterations, int(d_dp_thread_count));
//         // printf("Iteration #%lu, thread_count=%lu\n", iterations, d_dp_thread_count);

//         unsigned long thread_blocks = ceil(double(d_dp_thread_count) / THREADS_PER_BLOCK);

//         // if(iterations)
//         d_dp_thread_count = 0;

//         // sssp_kernel_child<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path, d_source_degrees, d_csr_offset, d_csr_edges, d_prefix_sum_edge_blocks, d_source_vector, d_TC_edge_vector, d_source_vector_1, d_mutex);
//         sssp_kernel_child_VC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path, d_source_degrees, d_csr_offset, d_csr_edges, d_prefix_sum_edge_blocks, d_source_vector, d_TC_edge_vector, d_source_vector_1, d_mutex, d_sssp_queue_1, d_sssp_queue_2);

//         cudaDeviceSynchronize();
//         // d_dp_thread_count = 0;

//         if(type)
//             type = 0;
//         else
//             type = 1;

//         iterations++;
//         d_prev_thread_count = d_dp_thread_count;

//     } while(d_dp_thread_count);

//     printf("Iterations for SSSP: %lu", iterations);

// }

__global__ void print_triangle_counts(float *d_triangle_count, unsigned long vertex_size) {

    unsigned long length = 40;
    if(length > vertex_size)
        length = vertex_size;

    printf("\nPer-vertex triangle count values\n");
    for(unsigned long i = 0 ; i < length ; i++)
        printf("%lu ", (unsigned long)d_triangle_count[i]);
    printf("\n\n");
    // printf("Tester is %f seconds\n",  (float)temp/ (1000000 * 1410));

}

__global__ void print_sssp_values(unsigned int *d_shortest_path, unsigned long vertex_size) {

    unsigned long length = 40;
    if(length > vertex_size)
        length = vertex_size;

    printf("\nSSSP values\n");
    for(unsigned long i = 0 ; i < length ; i++)
        printf("%u ", d_shortest_path[i]);
    printf("\n\n");
    // printf("Tester is %f seconds\n",  (float)temp/ (1000000 * 1410));

}

void readFile(char *fileLoc, struct graph_properties *h_graph_prop, thrust::host_vector <unsigned long> &h_source, thrust::host_vector <unsigned long> &h_destination, thrust::host_vector <unsigned long> &h_source_degrees, unsigned long type) {

    FILE* ptr;
    char buff[100];
    ptr = fopen(fileLoc, "a+");

    if (NULL == ptr) {
        printf("file can't be opened \n");
    }

    int dataFlag = 0;
    unsigned long index = 0;

    while (fgets(buff, 400, ptr) != NULL) {

        // ignore commented lines
        if(buff[0] == '%')
            continue;
            // reading edge values
        else if (dataFlag) {

            // printf("%s", buff);
            unsigned long source;
            unsigned long destination;
            int i = 0, j = 0;
            char temp[100];

            while( buff[i] != ' ' )
                temp[j++] = buff[i++];
            source = strtol( temp, NULL, 10);
            // printf("%lu ", nodeID);
            memset( temp, '\0', 100);

            i++;
            j=0;
            // while( buff[i] != ' ' )
            while((buff[i] != '\0') && (buff[i] != ' ') )
                temp[j++] = buff[i++];
            destination = strtol( temp, NULL, 10);
            // printf("%.8Lf ", x);
            memset( temp, '\0', 100);
            // h_edges[i] = ;
            if((index % 500000000) == 0)
                printf("%lu and %lu\n", source, destination);
            // if((source >= h_graph_prop->xDim) || (destination >= h_graph_prop->xDim))
            //     printf("%lu and %lu\n", source, destination);

            h_source[index] = source;
            h_destination[index++] = destination;
            h_source_degrees[source - 1]++;

            // below part makes it an undirected graph
            if(!DIRECTED) {

                h_source[index] = destination;
                h_destination[index++] = source;
                h_source_degrees[destination - 1]++;

            }
        }
            // reading xDim, yDim, and total_edges
        else {

            unsigned long xDim, yDim, total_edges;

            int i = 0,j = 0;
            char temp[100];

            while( buff[i] != ' ' )
                temp[j++] = buff[i++];
            xDim = strtol( temp, NULL, 10);
            // printf("%lu ", nodeID);
            memset( temp, '\0', 100);

            i++;
            j=0;
            while( buff[i] != ' ' )
                temp[j++] = buff[i++];
            yDim = strtol( temp, NULL, 10);
            // printf("%.8Lf ", x);
            memset( temp, '\0', 100);

            i++;
            j=0;
            while( buff[i] !='\0' )
                temp[j++] = buff[i++];
            total_edges = strtol( temp, NULL, 10);
            // printf("%.8Lf\n", y);
            memset( temp, '\0', 100);

            if(DIRECTED)
                printf("xDim = %lu, yDim = %lu, Total Edges = %lu\n", xDim, yDim, total_edges);
            else
                printf("xDim = %lu, yDim = %lu, Total Edges = %lu\n", xDim, yDim, total_edges * 2);

            h_graph_prop->xDim = xDim;
            h_graph_prop->yDim = yDim;
            // total edges doubles since undirected graph
            if(DIRECTED)
                h_graph_prop->total_edges = total_edges;
            else
                h_graph_prop->total_edges = total_edges * 2;

            // sleep(5);

            if(h_graph_prop->total_edges > BATCH_SIZE) {
                h_source.resize(h_graph_prop->total_edges);
                h_destination.resize(h_graph_prop->total_edges);
            }
            else {
                h_source.resize(BATCH_SIZE);
                h_destination.resize(BATCH_SIZE);
            }
            h_source_degrees.resize(h_graph_prop->xDim);


            dataFlag = 1;

            // type is 1 if random graph chosen, no need then to read original graph
            // if(type)
            //     break;

        }

    }

    fclose(ptr);

}

void generateCSR(unsigned long vertex_size, unsigned long edge_size, thrust::host_vector <unsigned long> &h_source, thrust::host_vector <unsigned long> &h_destination, thrust::host_vector <unsigned long> &h_csr_offset, thrust::host_vector <unsigned long> &h_csr_edges, thrust::host_vector <unsigned long> &h_source_degrees) {

    h_csr_offset[0] = 0;
    h_csr_offset[1] = h_source_degrees[0];
    for(unsigned long i = 2 ; i < vertex_size + 1 ; i++)
        h_csr_offset[i] += h_csr_offset[i-1] + h_source_degrees[i - 1];

    thrust::host_vector <unsigned long> index(vertex_size);
    thrust::fill(index.begin(), index.end(), 0);

    // std::cout << "Checkpoint 2" << std::endl;

    for(unsigned long i = 0 ; i < edge_size ; i++) {

        if(h_source[i] == 1)
            h_csr_edges[index[h_source[i] - 1]++] = h_destination[i];
        else
            h_csr_edges[h_csr_offset[h_source[i] - 1] + index[h_source[i] - 1]++] = h_destination[i];

    }

}

void generateCSRnew(unsigned long vertex_size, unsigned long edge_size, thrust::host_vector <unsigned long> &h_source, thrust::host_vector <unsigned long> &h_destination, thrust::host_vector <unsigned long> &h_csr_offset_new, thrust::host_vector <unsigned long> &h_csr_edges_new, thrust::host_vector <unsigned long> &h_source_degrees_new, thrust::host_vector <unsigned long> &h_edge_blocks_count, unsigned long batch_size, unsigned long total_batches) {

    unsigned long batch_offset = 0;
    unsigned long batch_number = 0;
    for(unsigned long i = 0 ; i < edge_size ; i++) {

        h_source_degrees_new[h_source[i] - 1 + batch_offset]++;
        // h_source_degrees_new[h_source[i] - 1]++;

        if((((i + 1) % batch_size ) == 0) && (i != 0)) {
            batch_offset = vertex_size * ++batch_number;
            // for(unsigned long i = 0 ; i < vertex_size * total_batches ; i++)
            //     std::cout << h_source_degrees_new[i] << " ";
            // std::cout << "Hit at CSR batch, new batch offset is " << batch_offset << std::endl;
        }

    }

    batch_number = 0;
    batch_offset = 0;
    unsigned long k = 0;
    for(unsigned long i = 0 ; i < total_batches ; i++) {


        h_csr_offset_new[batch_offset] = 0;
        h_csr_offset_new[batch_offset + 1] = h_source_degrees_new[batch_offset - k];
        // for(unsigned long i = 0 ; i < (vertex_size + 1) * total_batches ; i++) {
        for(unsigned long j = 2 ; j < (vertex_size + 1) ; j++)
            h_csr_offset_new[j + batch_offset] += h_csr_offset_new[j - 1 + batch_offset] + h_source_degrees_new[j - 1 + batch_offset - k];

        batch_offset = (vertex_size + 1) * ++batch_number;
        k++;

    }
    thrust::host_vector <unsigned long> index(vertex_size * total_batches);
    // thrust::fill(index.begin(), index.end(), 0);

    // // // std::cout << "Checkpoint 2" << std::endl;

    batch_number = 0;
    batch_offset = 0;
    unsigned long batch_offset_index = 0;
    unsigned long batch_offset_csr = 0;
    for(unsigned long i = 0 ; i < edge_size ; i++) {

        if(h_source[i] == 1)
            h_csr_edges_new[index[h_source[i] - 1 + batch_offset_index]++ + batch_offset] = h_destination[i];
        else
            h_csr_edges_new[h_csr_offset_new[h_source[i] - 1 + batch_offset_csr] + index[h_source[i] - 1 + batch_offset_index]++ + batch_offset] = h_destination[i];

        if((((i + 1) % batch_size ) == 0) && (i != 0)) {
            batch_offset = batch_size * ++batch_number;
            batch_offset_index = vertex_size * batch_number;
            batch_offset_csr = (vertex_size + 1) * batch_number;
            // for(unsigned long i = 0 ; i < vertex_size * total_batches ; i++)
            //     std::cout << h_source_degrees_new[i] << " ";
            std::cout << "Hit at CSR batch, new batch offset is " << batch_offset << std::endl;
        }

    }

    // for(unsigned long i = batch_number * batch_size; i < batch_number * batch_size + edge_size ; i++) {

    //     if(h_source[i] == 1)
    //         h_csr_edges_new[index[h_source[i] - 1]++] = h_destination[i];
    //     else
    //         h_csr_edges_new[h_csr_offset_new[h_source[i] - 1] + index[h_source[i] - 1]++] = h_destination[i];

    // }

    thrust::host_vector <unsigned long> space_remaining(vertex_size);
    unsigned long total_edge_blocks_count_init = 0;

    std::cout << "Space Remaining" << std::endl;
    // std::cout << "Edge blocks calculation" << std::endl << "Source\tEdge block count\tGPU address" << std::endl;
    for(unsigned long i = 0 ; i < total_batches ; i++) {

        for(unsigned long j = 0 ; j < vertex_size ; j++) {

            if(h_source_degrees_new[j + (i * vertex_size)]) {

                unsigned long edge_blocks;
                if(i != 0)
                    edge_blocks = ceil(double(h_source_degrees_new[j + (i * vertex_size)] - space_remaining[j]) / EDGE_BLOCK_SIZE);
                else
                    edge_blocks = ceil(double(h_source_degrees_new[j + (i * vertex_size)]) / EDGE_BLOCK_SIZE);


                h_edge_blocks_count[j + (i * vertex_size)] = edge_blocks;
                total_edge_blocks_count_init += edge_blocks;

                // if(h_source_degrees_new[j + (i * vertex_size)])
                space_remaining[j] = (h_source_degrees_new[j + (i * vertex_size)] + space_remaining[j]) % EDGE_BLOCK_SIZE;
                // else
                //     space_remaining[j] = 0;

            }

            else
                h_edge_blocks_count[j + (i * vertex_size)] = 0;


        }

        for(unsigned long j = 0 ; j < vertex_size ; j++)
            std::cout << space_remaining[j] << " ";
        std::cout << std::endl;

    }

    // std::cout << std::endl << std::endl << "Printing batched CSR" << std::endl << "Source degrees\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < vertex_size * total_batches ; i++) {
    //     std::cout << h_source_degrees_new[i] << " ";
    //     if((i + 1) % vertex_size == 0)
    //         std::cout << std::endl;
    // }
    // std::cout << std::endl << "CSR offset\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < (vertex_size + 1) * total_batches ; i++) {
    //     std::cout << h_csr_offset_new[i] << " ";
    //     if(((i + 1) % (vertex_size + 1)) == 0)
    //         std::cout << std::endl;
    // }
    // std::cout << std::endl << "CSR edges\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < batch_size * total_batches ; i++) {
    //     std::cout << h_csr_edges_new[i] << " ";
    //     if((i + 1) % batch_size == 0)
    //         std::cout << std::endl;
    // }
    // std::cout << std::endl << "Edge blocks count\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < vertex_size * total_batches ; i++) {
    //     std::cout << h_edge_blocks_count[i] << " ";
    //     if((i + 1) % vertex_size == 0)
    //         std::cout << std::endl;
    // }
    // std::cout << std::endl << std::endl << std::endl;
}

void generate_random_batch(unsigned long vertex_size, unsigned long batch_size, thrust::host_vector <unsigned long> &h_source, thrust::host_vector <unsigned long> &h_destination, thrust::host_vector <unsigned long> &h_source_degrees_new, thrust::host_vector <unsigned long> &h_source_degrees, unsigned long batch_number) {

    // unsigned long batch_size = 10;
    // unsigned long vertex_size = 30;

    // unsigned long seed = 0;
    unsigned long seed = batch_number;
    unsigned long range = 0;
    unsigned long offset = 0;

    // unsigned long source_array[10];
    // unsigned long destination_array[10];

    srand(seed + 1);
    for (unsigned long i = 0; i < batch_size / 2; ++i)
    {
        // EdgeUpdateType edge_update_data;
        unsigned long intermediate = rand() % ((range && (range < vertex_size)) ? range : vertex_size);
        unsigned long source;
        if(offset + intermediate < vertex_size)
            source = offset + intermediate;
        else
            source = intermediate;
        h_source[i] = source + 1;
        h_destination[i] = (rand() % vertex_size) + 1;
        h_source_degrees[source]++;
        // edge_update->edge_update.push_back(edge_update_data);

        // h_source[i] = 1;
        // h_source_degrees[0]++;
    }

    for (unsigned long i = batch_size / 2; i < batch_size; ++i)
    {
        // EdgeUpdateType edge_update_data;
        unsigned long intermediate = rand() % (vertex_size);
        unsigned long source;
        if(offset + intermediate < vertex_size)
            source = offset + intermediate;
        else
            source = intermediate;
        h_source[i] = source + 1;
        h_destination[i] = (rand() % vertex_size) + 1;
        h_source_degrees[source]++;
        // edge_update->edge_update.push_back(edge_update_data);

        // h_source[i] = 1;
        // h_source_degrees[0]++;

    }

    // std::cout << "Printing source-destination pairs" << std::endl;
    // for(unsigned long i = 0 ; i < batch_size ; i++)
    //     std::cout << h_source[i] << " and " << h_destination[i] << std::endl;

}

void generate_csr_batch(unsigned long vertex_size, unsigned long edge_size, unsigned long max_degree, thrust::host_vector <unsigned long> &h_source, thrust::host_vector <unsigned long> &h_destination, thrust::host_vector <unsigned long> &h_csr_offset_new, thrust::host_vector <unsigned long> &h_csr_edges_new, thrust::host_vector <unsigned long> &h_source_degrees_new, thrust::host_vector <unsigned long> &h_edge_blocks_count, thrust::host_vector <unsigned long> &h_prefix_sum_edge_blocks_new, unsigned long *h_batch_update_data, unsigned long batch_size, unsigned long total_batches, unsigned long batch_number,  thrust::host_vector <unsigned long> &space_remaining, unsigned long *total_edge_blocks_count_batch, clock_t *init_time) {

    thrust::host_vector <unsigned long> index(vertex_size);
    thrust::fill(h_source_degrees_new.begin(), h_source_degrees_new.end(), 0);

    // calculating start and end index of this batch, for use with h_source and h_destination
    // unsigned long start_index = batch_number * batch_size;
    // unsigned long end_index = start_index + batch_size;
    unsigned long start_index = 0;
    unsigned long end_index = start_index + batch_size;
    // std::cout << "At csr generation, start_index is " << start_index << ", end_index is " << end_index << std::endl;

    if(end_index > edge_size)
        end_index = edge_size;

    // thrust::host_vector <unsigned long> h_source_degrees_new_prev(vertex_size);
    // thrust::copy(h_source_degrees_new.begin(), h_source_degrees_new.end(), h_source_degrees_new_prev.begin());

    // unsigned long max = h_source_degrees_new[0];
    // unsigned long min = h_source_degrees_new[0];
    // unsigned long sum = h_source_degrees_new[0];
    // unsigned long non_zero_count = 0;

    // if(h_source_degrees_new[0])
    //     non_zero_count++;

    // calculating source degrees of this batch
    for(unsigned long i = start_index ; i < end_index ; i++) {

        h_source_degrees_new[h_source[i] - 1]++;

    }

    // std::cout << "Checkpoint 1" << std::endl;

    // for(unsigned long i = 1 ; i < vertex_size ; i++) {

    //     if(h_source_degrees_new[i] > max)
    //         max = h_source_degrees_new[i];

    //     if(h_source_degrees_new[i] < min)
    //         min = h_source_degrees_new[i];

    //     sum += h_source_degrees_new[i];

    //     if(h_source_degrees_new[i])
    //         non_zero_count++;

    // }



    // calculating csr offset of this batch
    h_csr_offset_new[0] = 0;
    h_csr_offset_new[1] = h_source_degrees_new[0];
    // h_batch_update_data[0] = 0;
    // h_batch_update_data[1] = h_source_degrees_new[0];
    for(unsigned long j = 2 ; j < (vertex_size + 1) ; j++) {
        h_csr_offset_new[j] = h_csr_offset_new[j - 1] + h_source_degrees_new[j - 1];
        // h_batch_update_data[j] = h_batch_update_data[j - 1] + h_source_degrees_new[j - 1];
    }

    // std::cout << "Checkpoint 2 , start_index is " << start_index << " and end_index is " << end_index << std::endl;


    // unsigned long offset = vertex_size + 1;
    unsigned long offset = 0;

    // calculating csr edges of this batch
    for(unsigned long i = start_index ; i < end_index ; i++) {

        if(h_source[i] == 1) {
            // h_csr_edges_new[index[h_source[i] - 1]++] = h_destination[i];
            h_csr_edges_new[index[h_source[i] - 1]] = h_destination[i];
            h_batch_update_data[offset + index[h_source[i] - 1]++] = h_destination[i];
        }
        else {
            // h_csr_edges_new[h_csr_offset_new[h_source[i] - 1] + index[h_source[i] - 1]++] = h_destination[i];
            h_csr_edges_new[h_csr_offset_new[h_source[i] - 1] + index[h_source[i] - 1]] = h_destination[i];
            h_batch_update_data[offset + h_csr_offset_new[h_source[i] - 1] + index[h_source[i] - 1]++] = h_destination[i];
        }
    }

    // comment below section for not sorting each adjacency in the CSR
    unsigned long start_index_edges;
    unsigned long end_index_edges;
    for(unsigned long i = 0 ; i < vertex_size ; i++) {

        start_index_edges = h_csr_offset_new[i];
        end_index_edges = h_csr_offset_new[i + 1];

        if(start_index_edges < end_index_edges)
            thrust::sort(thrust::host, thrust::raw_pointer_cast(h_csr_edges_new.data()) + start_index_edges, thrust::raw_pointer_cast(h_csr_edges_new.data()) + end_index_edges);

    }


    // calculating edge blocks required for this batch
    for(unsigned long j = 0 ; j < vertex_size ; j++) {

        if(h_source_degrees_new[j]) {

            unsigned long edge_blocks;
            if(batch_number != 0) {
                if(space_remaining[j] == 0) {
                    edge_blocks = ceil(double(h_source_degrees_new[j]) / EDGE_BLOCK_SIZE);
                    space_remaining[j] = (EDGE_BLOCK_SIZE - (h_source_degrees_new[j] % EDGE_BLOCK_SIZE)) % EDGE_BLOCK_SIZE;
                }
                else if(h_source_degrees_new[j] >= space_remaining[j]) {
                    edge_blocks = ceil(double(h_source_degrees_new[j] - space_remaining[j]) / EDGE_BLOCK_SIZE);
                    // space_remaining[j] = (h_source_degrees_new[j] - space_remaining[j]) % EDGE_BLOCK_SIZE;
                    space_remaining[j] = (EDGE_BLOCK_SIZE - ((h_source_degrees_new[j] - space_remaining[j]) % EDGE_BLOCK_SIZE)) % EDGE_BLOCK_SIZE;
                    // std::cout << "Vertex " << j << ", edge_blocks needed is " << edge_blocks << std::endl;
                }
                else {
                    edge_blocks = 0;
                    space_remaining[j] = space_remaining[j] - h_source_degrees_new[j];
                }
                // h_prefix_sum_edge_blocks_new[j] = h_prefix_sum_edge_blocks_new[j - 1] + h_edge_blocks_count[j];
            }
            else {
                edge_blocks = ceil(double(h_source_degrees_new[j]) / EDGE_BLOCK_SIZE);
                space_remaining[j] = (EDGE_BLOCK_SIZE - (h_source_degrees_new[j] % EDGE_BLOCK_SIZE)) % EDGE_BLOCK_SIZE;
                // h_prefix_sum_edge_blocks_new[0] = edge_blocks;
            }

            // std::cout << "Vertex " << j << " needs " << edge_blocks << " edge blocks" << std::endl;

            // if((vertex_adjacency->edge_block_address != NULL) && (vertex_adjacency->last_insert_edge_offset != 0)) {

            //     space_remaining = EDGE_BLOCK_SIZE - device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset;
            //     // printf("id=%lu, last_insert_edge_offset is %lu\n", id, device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset);
            // }

            h_edge_blocks_count[j] = edge_blocks;
            *total_edge_blocks_count_batch += edge_blocks;

            // if(space_remaining[j] <= h_source_degrees_new[j])
            //     space_remaining[j] = (h_source_degrees_new[j] - space_remaining[j]) % EDGE_BLOCK_SIZE;
            // else
            //     space_remaining[j] = space_remaining[j] - h_source_degrees_new[j];

        }

        else
            h_edge_blocks_count[j] = 0;


    }

    // if((vertex_adjacency->edge_block_address != NULL) && (vertex_adjacency->last_insert_edge_offset != 0)) {

    //     space_remaining = EDGE_BLOCK_SIZE - device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset;
    //     // printf("id=%lu, last_insert_edge_offset is %lu\n", id, device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset);
    // }

    // offset += batch_size;

    // clock_t temp_time;
    // temp_time = clock();

    clock_t prefix_sum_time;
    prefix_sum_time = clock();

    h_prefix_sum_edge_blocks_new[0] = 0;
    // h_prefix_sum_edge_blocks_new[1] = h_edge_blocks_count[0];
    // h_batch_update_data[offset] = h_edge_blocks_count[0];
    // printf("Prefix sum array edge blocks\n%ld ", h_prefix_sum_edge_blocks[0]);
    for(unsigned long i = 1 ; i < vertex_size + 1 ; i++) {

        h_prefix_sum_edge_blocks_new[i] = h_prefix_sum_edge_blocks_new[i - 1] + h_edge_blocks_count[i - 1];
        // h_batch_update_data[offset + i] = h_batch_update_data[offset + i - 1] + h_edge_blocks_count[i];
        // printf("%ld ", h_prefix_sum_edge_blocks[i]);

    }

    prefix_sum_time = clock() -prefix_sum_time;

    std::cout << "Prefix Sum Time: " << (float)prefix_sum_time/CLOCKS_PER_SEC << " seconds" << std::endl;

    // std::ofstream DataFile("data_points.txt");

    // for(unsigned long i = 0 ; i < vertex_size ; i++)
    //     DataFile << i << " ";
    // DataFile << "\n";
    // for(unsigned long i = 0 ; i < vertex_size ; i++)
    //     DataFile << h_source_degrees_new[i] << " ";
    // DataFile << "\n";

    // DataFile.close();

    // std::cout << "Data points written to file" << std::endl << std::endl;

    // bucket stats code start

    // unsigned long bucket_size_1 = 1;
    // unsigned long buckets_1 = ceil((double)max_degree / bucket_size_1);
    // thrust::host_vector <unsigned long> degree_change_1(buckets_1 + 1);

    // unsigned long bucket_size_2 = 10000;
    // unsigned long buckets_2 = ceil((double)vertex_size / bucket_size_2);
    // thrust::host_vector <unsigned long> degree_change_2(buckets_2);

    // // std::string filename = "output.csv";

    // // Write data to the CSV file
    // std::ofstream outputFile1("bucket_degree.csv");
    // std::ofstream outputFile2("degree_distribution.csv");

    // // if (!outputFile.is_open()) {
    // //     std::cerr << "Error opening file: " << filename << std::endl;
    // //     return;
    // // }

    // for (unsigned long i = 0 ; i < vertex_size ; i++) {
    //     // for (size_t i = 0; i < row.size(); ++i) {

    //         degree_change_1[floor((double)h_source_degrees_new[i] / bucket_size_1)]++;
    //         degree_change_2[floor(double(i) / bucket_size_2)] += h_source_degrees_new[i];


    //         // outputFile << i << "," << h_source_degrees_new[i];
    //         // if (i < row.size() - 1) {
    //             // outputFile << ",";
    //         // }
    //     // }
    // }

    // for(unsigned long i = 0 ; i < buckets_1 + 1 ; i++)
    //     outputFile1 << i << "," << degree_change_1[i] << std::endl;

    // for(unsigned long i = 0 ; i < buckets_2 ; i++)
    //     outputFile2 << i << "," << degree_change_2[i] << std::endl;
    // // outputFile << std::endl;

    // outputFile1.close();
    // outputFile2.close();

    // std::cout << "Data has been written to files" << std::endl;

    // bucket stats code end

    // temp_time = clock() - temp_time;
    // *init_time += temp_time;

    // printf("Max, Min, Average, and Non-zero Average degrees in this batch are %lu, %lu, %f, and %f respectively\n", max, min, float(sum) / vertex_size, float(sum) / non_zero_count);

    std::cout << std::endl << std::endl << "Printing batched CSR" << std::endl << "Source degrees\t\t" << std::endl;
    for(unsigned long i = 0 ; i < vertex_size ; i++) {
        std::cout << h_source_degrees_new[i] << " ";
        if((i + 1) % vertex_size == 0)
            std::cout << std::endl;
    }


    std::cout << std::endl << "CSR offset\t\t" << std::endl;
    for(unsigned long i = 0 ; i < (vertex_size + 1) ; i++) {
        std::cout << h_csr_offset_new[i] << " ";
        if(((i + 1) % (vertex_size + 1)) == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl << "CSR edges\t\t" << std::endl;
    for(unsigned long i = 0 ; i < batch_size ; i++) {
        std::cout << h_csr_edges_new[i] << " ";
        if((i + 1) % batch_size == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl << "Edge blocks count\t\t" << std::endl;
    for(unsigned long i = 0 ; i < vertex_size ; i++) {
        std::cout << h_edge_blocks_count[i] << " ";
        if((i + 1) % vertex_size == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl << "Prefix sum edge blocks\t\t" << std::endl;
    for(unsigned long i = 0 ; i < vertex_size ; i++) {
        std::cout << h_prefix_sum_edge_blocks_new[i] << " ";
        if((i + 1) % vertex_size == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl << "Space remaining\t\t" << std::endl;
    for(unsigned long j = 0 ; j < vertex_size ; j++)
        std::cout << space_remaining[j] << " ";
    std::cout << std::endl;
    // std::cout << std::endl << std::endl << std::endl;
}

void remove_batch_duplicates(unsigned long vertex_size, unsigned long edge_size, thrust::host_vector <unsigned long> &h_csr_offset_new, thrust::host_vector <unsigned long> &h_csr_edges_new, unsigned long *h_batch_update_data, unsigned long batch_size) {

    thrust::host_vector <unsigned long> h_csr_offset(vertex_size + 1);
    thrust::host_vector <unsigned long> h_csr_edges(batch_size);

    h_csr_offset[0] = 0;

    unsigned long index = 0;
    // unsigned long current_length = 0;

    for(unsigned long i = 0  ; i < vertex_size ; i++) {

        unsigned long start_index = h_csr_offset_new[i];
        unsigned long end_index = h_csr_offset_new[i + 1];
        unsigned long prev_value;
        // current_length = 0;

        if(start_index < end_index) {
            h_csr_edges[index++] = h_csr_edges_new[start_index];
            // h_csr_edges_new[index++] = h_csr_edges_new[start_index];
            prev_value = h_csr_edges_new[start_index];
            // index++;
            // current_length++;
        }


        for(unsigned long j = start_index + 1 ; j < end_index ; j++) {

            if(h_csr_edges_new[j] != prev_value) {

                h_csr_edges[index++] = h_csr_edges_new[j];
                // h_csr_edges_new[index++] = h_csr_edges_new[j];
                prev_value = h_csr_edges_new[j];
                // current_length++;

            }

            // index++;

        }

        h_csr_offset[i + 1] = index;
        // h_csr_offset_new[i + 1] = index;

    }

    for(unsigned long i = 0 ; i < (vertex_size + 1) ; i++)
        h_csr_offset_new[i] = h_csr_offset[i];
    for(unsigned long i = 0 ; i < batch_size ; i++)
        h_csr_edges_new[i] = h_csr_edges[i];

    // std::cout << "After removing duplicates" << std::endl;

    // std::cout << std::endl << "CSR offset\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < (vertex_size + 1) ; i++) {
    //     std::cout << h_csr_offset_new[i] << " ";
    //     if(((i + 1) % (vertex_size + 1)) == 0)
    //         std::cout << std::endl;
    // }
    // std::cout << "CSR edges\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < batch_size ; i++) {
    //     std::cout << h_csr_edges_new[i] << " ";
    //     if((i + 1) % batch_size == 0)
    //         std::cout << std::endl;
    // }

}

void generate_csr_batch_tester(unsigned long vertex_size, unsigned long edge_size, unsigned long max_degree, thrust::host_vector <unsigned long> &h_source, thrust::host_vector <unsigned long> &h_destination, thrust::host_vector <unsigned long> &h_csr_offset_new, thrust::host_vector <unsigned long> &h_csr_edges_new, thrust::host_vector <unsigned long> &h_source_degrees_new, thrust::host_vector <unsigned long> &h_edge_blocks_count, thrust::host_vector <unsigned long> &h_prefix_sum_edge_blocks_new, unsigned long *h_batch_update_data, unsigned long batch_size, unsigned long total_batches, unsigned long batch_number,  thrust::host_vector <unsigned long> &space_remaining, unsigned long *total_edge_blocks_count_batch, clock_t *init_time) {

    thrust::host_vector <unsigned long> index(vertex_size);
    thrust::fill(h_source_degrees_new.begin(), h_source_degrees_new.end(), 0);

    // calculating start and end index of this batch, for use with h_source and h_destination
    unsigned long start_index = 0;
    unsigned long end_index = start_index + batch_size;
    // std::cout << "At csr generation, start_index is " << start_index << ", end_index is " << end_index << std::endl;

    // if(end_index > batch_size)
    //     end_index = batch_size;

    // std::cout << "End index is " << end_index << std::endl;

    // calculating source degrees of this batch
    // for(unsigned long i = start_index ; i < end_index ; i++) {

    //     h_source_degrees_new[h_source[i] - 1]++;

    // }

    // calculating csr offset of this batch
    // h_csr_offset_new[0] = 0;
    // h_csr_offset_new[1] = h_source_degrees_new[0];
    // // h_batch_update_data[0] = 0;
    // // h_batch_update_data[1] = h_source_degrees_new[0];
    // for(unsigned long j = 2 ; j < (vertex_size + 1) ; j++) {
    //     h_csr_offset_new[j] = h_csr_offset_new[j - 1] + h_source_degrees_new[j - 1];
    //     // h_batch_update_data[j] = h_batch_update_data[j - 1] + h_source_degrees_new[j - 1];
    // }

    // std::cout << "Checkpoint 2 , start_index is " << start_index << " and end_index is " << end_index << std::endl;


    // unsigned long offset = vertex_size + 1;
    unsigned long offset = 0;

    // return;

    // calculating csr edges of this batch
    for(unsigned long i = start_index ; i < end_index ; i++) {

        if(h_source[i] == 1) {
            // h_csr_edges_new[index[h_source[i] - 1]++] = h_destination[i];
            h_csr_edges_new[index[h_source[i] - 1]++] = h_destination[i];
            // h_batch_update_data[offset + index[h_source[i] - 1]++] = h_destination[i];
        }
        else {
            // h_csr_edges_new[h_csr_offset_new[h_source[i] - 1] + index[h_source[i] - 1]++] = h_destination[i];
            h_csr_edges_new[h_csr_offset_new[h_source[i] - 1] + index[h_source[i] - 1]++] = h_destination[i];
            // h_batch_update_data[offset + h_csr_offset_new[h_source[i] - 1] + index[h_source[i] - 1]++] = h_destination[i];
        }
    }

    // comment below section for not sorting each adjacency in the CSR
    // unsigned long start_index_edges;
    // unsigned long end_index_edges;
    // for(unsigned long i = 0 ; i < vertex_size ; i++) {

    //     start_index_edges = h_csr_offset_new[i];
    //     end_index_edges = h_csr_offset_new[i + 1];

    //     if(start_index_edges < end_index_edges)
    //         thrust::sort(thrust::host, thrust::raw_pointer_cast(h_csr_edges_new.data()) + start_index_edges, thrust::raw_pointer_cast(h_csr_edges_new.data()) + end_index_edges);

    // }

    // std::cout << std::endl << std::endl << "Printing batched CSR at tester" << std::endl << "Source degrees\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //     std::cout << h_source_degrees_new[i] << " ";
    //     if((i + 1) % vertex_size == 0)
    //         std::cout << std::endl;
    // }


    // std::cout << std::endl << "CSR offset\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < (vertex_size + 1) ; i++) {
    //     std::cout << h_csr_offset_new[i] << " ";
    //     if(((i + 1) % (vertex_size + 1)) == 0)
    //         std::cout << std::endl;
    // }
    // std::cout << std::endl << "CSR edges\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < vertex_size ; i++) {

    //     for(unsigned long j = h_csr_offset_new[i] ; j < h_csr_offset_new[i + 1] ; j++)
    //         std::cout << h_csr_edges_new[j] << " ";

    //     std::cout << std::endl;

    // }

    // for(unsigned long i = 0 ; i < batch_size ; i++) {
    //     std::cout << h_csr_edges_new[i] << " ";
    //     // if((i + 1) % (h_csr_offset_new[i + 1] - h_csr_offset_new[i]) == 0)
    //         // std::cout << std::endl;
    // }
    // std::cout << std::endl << "Edge blocks count\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //     std::cout << h_edge_blocks_count[i] << " ";
    //     if((i + 1) % vertex_size == 0)
    //         std::cout << std::endl;
    // }
    // std::cout << std::endl << "Prefix sum edge blocks\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //     std::cout << h_prefix_sum_edge_blocks_new[i] << " ";
    //     if((i + 1) % vertex_size == 0)
    //         std::cout << std::endl;
    // }
    // std::cout << std::endl << "Space remaining\t\t" << std::endl;
    // for(unsigned long j = 0 ; j < vertex_size ; j++)
    //     std::cout << space_remaining[j] << " ";
    // std::cout << std::endl;
    // std::cout << std::endl << std::endl << std::endl;
}

void csr_sort(unsigned long id, unsigned long vertex_size, unsigned long *h_csr_offset, unsigned long *h_csr_edges) {

    unsigned long start_index_edges;
    unsigned long end_index_edges;
    // for(unsigned long i = 0 ; i < vertex_size ; i++) {

    start_index_edges = h_csr_offset[id];
    end_index_edges = h_csr_offset[id + 1];

    if(start_index_edges < end_index_edges)
        thrust::sort(thrust::host, h_csr_edges + start_index_edges, h_csr_edges + end_index_edges);

    // }

}

void csr_remove_duplicates(unsigned long id, unsigned long vertex_size, unsigned long *h_csr_offset, unsigned long *h_csr_edges, unsigned long *h_source_degrees) {

    unsigned long start_index = h_csr_offset[id];
    unsigned long end_index = h_csr_offset[id + 1];

    if(start_index == end_index)
        h_source_degrees[id] = 0;
    else {

        unsigned long index = start_index;
        unsigned long prev_value = h_csr_edges[index++];
        // current_length = 0;

        // if(start_index < end_index) {
        //     h_csr_edges[index++] = h_csr_edges_new[start_index];
        //     // h_csr_edges_new[index++] = h_csr_edges_new[start_index];
        //     prev_value = h_csr_edges_new[start_index];
        //     // index++;
        //     // current_length++;
        // }


        for(unsigned long i = start_index + 1 ; i < end_index ; i++) {

            if((h_csr_edges[i] != prev_value) && ((h_csr_edges[i] - 1) != id)) {

                h_csr_edges[index++] = h_csr_edges[i];
                // h_csr_edges_new[index++] = h_csr_edges_new[j];
                prev_value = h_csr_edges[i];
                // current_length++;

            }

            // index++;

        }

        h_source_degrees[id] = index - h_csr_offset[id];

    }
    // }

}

void reconstruct_deduplicated_csr(unsigned long vertex_size, unsigned long batch_size, unsigned long *h_csr_offset, unsigned long *h_csr_edges, unsigned long *h_source_degrees) {

    // if(id == 0) {

    // printf("\n\nBefore intra-batch\nSource degrees\t\t\n");
    // for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //     printf("%lu ", h_source_degrees[i]);
    //     if((i + 1) % vertex_size == 0)
    //         printf("\n");
    // }
    // printf("CSR offset\t\t\n");
    // for(unsigned long i = 0 ; i < vertex_size + 1 ; i++) {
    //     printf("%lu ", h_csr_offset[i]);
    //     if((i + 1) % (vertex_size + 1) == 0)
    //         printf("\n");
    // }
    // printf("CSR edges\t\t\n");
    // for(unsigned long i = 0 ; i < batch_size ; i++) {
    //     printf("%lu ", h_csr_edges[i]);
    //     if((i + 1) % batch_size == 0)
    //         printf("\n");
    // }

    // }

    unsigned long index = 0;
    unsigned long start_index = h_csr_offset[0];

    for(unsigned long i = 0 ; i < vertex_size ; i++) {

        unsigned long end_index = start_index + h_source_degrees[i];
        unsigned long new_end_index = index + h_source_degrees[i];

        for(unsigned long j = start_index ; j < end_index ; j++) {

            h_csr_edges[index++] = h_csr_edges[j];

        }

        start_index = h_csr_offset[i + 1];
        h_csr_offset[i + 1] = new_end_index;

    }

    // printf("\nAfter intra-batch\nSource degrees\t\t\n");
    // for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //     printf("%lu ", h_source_degrees[i]);
    //     if((i + 1) % vertex_size == 0)
    //         printf("\n");
    // }
    // printf("CSR offset\t\t\n");
    // for(unsigned long i = 0 ; i < vertex_size + 1 ; i++) {
    //     printf("%lu ", h_csr_offset[i]);
    //     if((i + 1) % (vertex_size + 1) == 0)
    //         printf("\n");
    // }
    // printf("CSR edges\t\t\n");
    // for(unsigned long i = 0 ; i < batch_size ; i++) {
    //     printf("%lu ", h_csr_edges[i]);
    //     if((i + 1) % batch_size == 0)
    //         printf("\n");
    // }

}

void host_insert_preprocessing(unsigned long id, unsigned long vertex_size, unsigned long *h_source_degrees_new, unsigned long *space_remaining, unsigned long batch_number, unsigned long *h_edge_blocks_count) {

    // for(unsigned long j = 0 ; j < vertex_size ; j++) {

    // if(id == 0) {
    //     printf("At host insert\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //         printf("%lu ", h_source_degrees_new[i]);
    //         if((i + 1) % vertex_size == 0)
    //             printf("\n");
    //     }

    // }

    if(h_source_degrees_new[id]) {

        unsigned long edge_blocks;
        if(batch_number != 0) {
            if(space_remaining[id] == 0) {
                edge_blocks = ceil(double(h_source_degrees_new[id]) / EDGE_BLOCK_SIZE);
                space_remaining[id] = (EDGE_BLOCK_SIZE - (h_source_degrees_new[id] % EDGE_BLOCK_SIZE)) % EDGE_BLOCK_SIZE;
            }
            else if(h_source_degrees_new[id] >= space_remaining[id]) {
                edge_blocks = ceil(double(h_source_degrees_new[id] - space_remaining[id]) / EDGE_BLOCK_SIZE);
                // space_remaining[j] = (h_source_degrees_new[j] - space_remaining[j]) % EDGE_BLOCK_SIZE;
                space_remaining[id] = (EDGE_BLOCK_SIZE - ((h_source_degrees_new[id] - space_remaining[id]) % EDGE_BLOCK_SIZE)) % EDGE_BLOCK_SIZE;
                // std::cout << "Vertex " << j << ", edge_blocks needed is " << edge_blocks << std::endl;
            }
            else {
                edge_blocks = 0;
                space_remaining[id] = space_remaining[id] - h_source_degrees_new[id];
            }
            // h_prefix_sum_edge_blocks_new[j] = h_prefix_sum_edge_blocks_new[j - 1] + h_edge_blocks_count[j];
        }
        else {
            edge_blocks = ceil(double(h_source_degrees_new[id]) / EDGE_BLOCK_SIZE);
            space_remaining[id] = (EDGE_BLOCK_SIZE - (h_source_degrees_new[id] % EDGE_BLOCK_SIZE)) % EDGE_BLOCK_SIZE;
            // h_prefix_sum_edge_blocks_new[0] = edge_blocks;
        }

        // std::cout << "Vertex " << j << " needs " << edge_blocks << " edge blocks" << std::endl;

        // if((vertex_adjacency->edge_block_address != NULL) && (vertex_adjacency->last_insert_edge_offset != 0)) {

        //     space_remaining = EDGE_BLOCK_SIZE - device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset;
        //     // printf("id=%lu, last_insert_edge_offset is %lu\n", id, device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset);
        // }

        h_edge_blocks_count[id] = edge_blocks;
        // *total_edge_blocks_count_batch += edge_blocks;

        // if(space_remaining[j] <= h_source_degrees_new[j])
        //     space_remaining[j] = (h_source_degrees_new[j] - space_remaining[j]) % EDGE_BLOCK_SIZE;
        // else
        //     space_remaining[j] = space_remaining[j] - h_source_degrees_new[j];

    }

    else
        h_edge_blocks_count[id] = 0;


    // }

}

__global__ void device_coo_to_csr(unsigned long vertex_size, unsigned long *d_source, unsigned long *d_destination, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long batch_size, unsigned long *d_source_degrees, unsigned long thread_divider) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < (vertex_size / thread_divider)) {

        for(unsigned long z = (id * thread_divider) ; z < ((id + 1) * thread_divider) ; z++) {

            if(d_source_degrees[z]) {

                unsigned long start_index = d_csr_offset[z];
                unsigned long end_index = d_csr_offset[z + 1];
                unsigned long offset = d_csr_offset[z];

                // calculating csr edges of this batch
                for(unsigned long i = 0 ; i < batch_size ; i++) {

                    if(d_source[i] == (z + 1)) {
                        // h_csr_edges_new[index[h_source[i] - 1]++] = h_destination[i];
                        //     h_csr_edges[index[h_source[i] - 1]] = h_destination[i];
                        //     // h_batch_update_data[offset + index[h_source[i] - 1]++] = h_destination[i];
                        // }
                        // else {
                        // h_csr_edges_new[h_csr_offset_new[h_source[i] - 1] + index[h_source[i] - 1]++] = h_destination[i];
                        d_csr_edges[offset++] = d_destination[i];
                        // offset++;
                        // h_batch_update_data[offset + h_csr_offset_new[h_source[i] - 1] + index[h_source[i] - 1]++] = h_destination[i];
                    }
                }

            }

        }
    }

}

__global__ void device_insert_preprocessing(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long batch_size, unsigned long *d_source_degrees, unsigned long batch_number, unsigned long *d_edge_blocks_count) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id == 0) {

    //     printf("\n\nPrinting batched CSR before init\nSource degrees\t\t\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //         printf("%lu ", d_source_degrees[i]);
    //         if((i + 1) % vertex_size == 0)
    //             printf("\n");
    //     }
    //     printf("\nCSR offset\t\t\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++) {
    //         printf("%lu ", d_csr_offset[i]);
    //         if((i + 1) % (vertex_size + 1) == 0)
    //             printf("\n");
    //     }
    //     printf("\nCSR edges\t\t\n");
    //     for(unsigned long i = 0 ; i < batch_size ; i++) {
    //         printf("%lu ", d_csr_edges[i]);
    //         if((i + 1) % batch_size == 0)
    //             printf("\n");
    //     }

    // }

    if(id < vertex_size) {

        // for(unsigned long j = 0 ; j < vertex_size ; j++) {

        // if(id == 0) {
        //     printf("At host insert\n");
        //     for(unsigned long i = 0 ; i < vertex_size ; i++) {
        //         printf("%lu ", h_source_degrees_new[i]);
        //         if((i + 1) % vertex_size == 0)
        //             printf("\n");
        //     }

        // }

        if(d_source_degrees[id]) {

            unsigned long last_insert_edge_offset = device_vertex_dictionary->last_insert_edge_offset[id];
            unsigned long space_remaining = 0;
            if(last_insert_edge_offset)
                space_remaining = EDGE_BLOCK_SIZE - last_insert_edge_offset;

            unsigned long edge_blocks;
            if(batch_number != 0) {
                if(space_remaining == 0) {
                    edge_blocks = ceil(double(d_source_degrees[id]) / EDGE_BLOCK_SIZE);
                    // space_remaining = (EDGE_BLOCK_SIZE - (d_source_degrees[id] % EDGE_BLOCK_SIZE)) % EDGE_BLOCK_SIZE;
                }
                else if(d_source_degrees[id] >= space_remaining) {
                    edge_blocks = ceil(double(d_source_degrees[id] - space_remaining) / EDGE_BLOCK_SIZE);
                    // space_remaining[j] = (h_source_degrees_new[j] - space_remaining[j]) % EDGE_BLOCK_SIZE;
                    // space_remaining = (EDGE_BLOCK_SIZE - ((d_source_degrees[id] - space_remaining) % EDGE_BLOCK_SIZE)) % EDGE_BLOCK_SIZE;
                    // std::cout << "Vertex " << j << ", edge_blocks needed is " << edge_blocks << std::endl;
                }
                else {
                    edge_blocks = 0;
                    // space_remaining = space_remaining - d_source_degrees[id];
                }
                // h_prefix_sum_edge_blocks_new[j] = h_prefix_sum_edge_blocks_new[j - 1] + h_edge_blocks_count[j];
            }
            else {
                edge_blocks = ceil(double(d_source_degrees[id]) / EDGE_BLOCK_SIZE);
                // space_remaining = (EDGE_BLOCK_SIZE - (d_source_degrees[id] % EDGE_BLOCK_SIZE)) % EDGE_BLOCK_SIZE;
                // h_prefix_sum_edge_blocks_new[0] = edge_blocks;
            }

            // std::cout << "Vertex " << j << " needs " << edge_blocks << " edge blocks" << std::endl;

            // if((vertex_adjacency->edge_block_address != NULL) && (vertex_adjacency->last_insert_edge_offset != 0)) {

            //     space_remaining = EDGE_BLOCK_SIZE - device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset;
            //     // printf("id=%lu, last_insert_edge_offset is %lu\n", id, device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset);
            // }
            // printf("ID=%lu, d_source_degrees=%lu, space_remaining=%lu, edge_blocks=%lu, last_insert_edge_offset=%lu\n", id, d_source_degrees[id], space_remaining, edge_blocks, device_vertex_dictionary->last_insert_edge_offset[id]);

            d_edge_blocks_count[id] = edge_blocks;
            // *total_edge_blocks_count_batch += edge_blocks;

            // if(space_remaining[j] <= h_source_degrees_new[j])
            //     space_remaining[j] = (h_source_degrees_new[j] - space_remaining[j]) % EDGE_BLOCK_SIZE;
            // else
            //     space_remaining[j] = space_remaining[j] - h_source_degrees_new[j];

        }

        else
            d_edge_blocks_count[id] = 0;


        // if(d_edge_blocks_count[id] > 2500)
        //     printf("Hit\n");

        // }
    }

    // if(id == 0) {

    // //     printf("\n\nPrinting batched CSR\nSource degrees\t\t\n");
    // //     for(unsigned long i = 0 ; i < vertex_size ; i++) {
    // //         printf("%lu ", d_source_degrees[i]);
    // //         if((i + 1) % vertex_size == 0)
    // //             printf("\n");
    // //     }
    // //     printf("\nCSR offset\t\t\n");
    // //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++) {
    // //         printf("%lu ", d_csr_offset[i]);
    // //         if((i + 1) % (vertex_size + 1) == 0)
    // //             printf("\n");
    // //     }
    // //     printf("\nCSR edges\t\t\n");
    // //     for(unsigned long i = 0 ; i < batch_size ; i++) {
    // //         printf("%lu ", d_csr_edges[i]);
    // //         if((i + 1) % batch_size == 0)
    // //             printf("\n");
    // //     }
    //     printf("\nEdge blocks needed\t\t\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //         printf("%lu ", d_edge_blocks_count[i]);
    //         if((i + 1) % vertex_size == 0)
    //             printf("\n");
    //     }

    // }

}

__global__ void tester_exclusive_scan(unsigned long vertex_size, unsigned long *d_edge_blocks_count, unsigned long *d_prefix_sum_edge_blocks) {

    for(unsigned long i = 0 ; i < vertex_size ; i++)
        d_prefix_sum_edge_blocks[i + 1] = d_prefix_sum_edge_blocks[i] + d_edge_blocks_count[i];


}

__global__ void device_inter_batch_deduplication_preprocessing_VC(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        unsigned long source = id;

        unsigned long start_index = d_csr_offset[source];
        unsigned long end_index = d_csr_offset[source + 1];

        long long top = -1;
        const unsigned long stack_size = 1;
        struct edge_block *stack[stack_size];

        // unsigned long source = d_sssp_queue_1[id / thread_multiplier];

        // unsigned long source_index = 0;
        // source_index = device_binary_search(d_sssp_output_frontier_offset, id / thread_multiplier, d_prev_thread_count);

        // unsigned long source = d_sssp_queue_1[source_index];

        struct edge_block *root = device_vertex_dictionary->edge_block_address[source];

        if(root == NULL)
            return;

        struct edge_block *curr_node = root;

        // printf("ID is %lu, source is %lu with address %p, stack base is %p, root is %p, curr_node is %p\n", id, source, &source, stack, root, curr_node);

        while((top > -1) || (curr_node != NULL)) {

            if(curr_node) {


                unsigned long destination;
                // unsigned long edge_entry_index = ((id) % thread_multiplier);
                // unsigned long start_index = edge_entry_index * 10;
                // unsigned long end_index = start_index + 10;

                // printf("ID is %lu, curr_node is %p\n", id, curr_node);

                for(unsigned long index = 0; index < EDGE_BLOCK_SIZE ; index++) {

                    // printf("%lu for ID %lu\n", root->edge_block_entry[index].destination_vertex, id);

                    if(curr_node->edge_block_entry[index].destination_vertex == 0)
                        break;

                    unsigned long destination = curr_node->edge_block_entry[index].destination_vertex - 1;

                    for(unsigned long i = start_index ; i < end_index ; i++) {

                        // if(d_shortest_path[long(d_sssp_queue_1[id])] + 1 < d_shortest_path[d_TC_edge_vector[i]]) {
                        if(destination == (d_csr_edges[i] - 1)) {
                            d_csr_edges[i] = INFTY;
                            d_source_degrees[source]--;
                        }

                    }


                }

                if(curr_node->rptr != NULL) {
                    push(stack, &top, stack_size, curr_node->rptr);
                    top++;
                    // printf("ID %lu pushed %p to stack, top=%lu\n", id, curr_node->rptr, top);
                }
                curr_node = curr_node->lptr;

            }
            else {

                curr_node = pop(stack, &top);
                top--;
                // printf("ID %lu popped %p from the stack\n", id, curr_node);
            }

            // printf("ID %lu, curr_node is now %p\n", id, curr_node);

        }

    }

    // if(id == 0) {

    //     printf("\n\nAfter Inter-batch deduplication\nSource degrees\t\t\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //         printf("%lu ", d_source_degrees[i]);
    //         if((i + 1) % vertex_size == 0)
    //             printf("\n");
    //     }
    //     printf("CSR offset\t\t\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++) {
    //         printf("%lu ", d_csr_offset[i]);
    //         if((i + 1) % (vertex_size + 1) == 0)
    //             printf("\n");
    //     }
    //     printf("CSR edges\t\t\n");
    //     for(unsigned long i = 0 ; i < BATCH_SIZE ; i++) {
    //         printf("%lu ", d_csr_edges[i]);
    //         if((i + 1) % BATCH_SIZE == 0)
    //             printf("\n");
    //     }

    // }

}

__global__ void device_remove_batch_duplicates(unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(!id) {

    //    printf("\n\nPrinting batched CSR before duplicate removal\nSource degrees\t\t\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //         printf("%lu ", d_source_degrees[i]);
    //         if((i + 1) % vertex_size == 0)
    //             printf("\n");
    //     }
    //     printf("\nCSR offset\t\t\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++) {
    //         printf("%lu ", d_csr_offset[i]);
    //         if((i + 1) % (vertex_size + 1) == 0)
    //             printf("\n");
    //     }
    //     printf("\nCSR edges\t\t\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++) {

    //         for(unsigned long j = d_csr_offset[i] ; j < d_csr_offset[i + 1] ; j++)
    //             printf("%lu ", d_csr_edges[j]);
    //         printf("\n");
    //     }
    // }

    if(id < batch_size) {

        // thrust::host_vector <unsigned long> h_csr_offset(vertex_size + 1);
        // thrust::host_vector <unsigned long> h_csr_edges(batch_size);

        // h_csr_offset[0] = 0;

        // unsigned long current_length = 0;

        // for(unsigned long i = 0  ; i < vertex_size ; i++) {

        unsigned long source = device_binary_search((unsigned long long*)d_csr_offset, id, vertex_size + 1);
        unsigned long index_counter = id - d_csr_offset[source];


        unsigned long start_index = id;
        unsigned long end_index = d_csr_offset[source + 1];
        unsigned long index = start_index;
        unsigned long prev_value = d_csr_edges[index++];
        // current_length = 0;
        // printf("ID %lu has source %lu, index_counter %lu, start_index is %lu, end_index is %lu, and destination is %lu\n", id, source, index_counter, start_index + 1, end_index, d_csr_edges[id]);

        // if(start_index < end_index) {
        //     h_csr_edges[index++] = h_csr_edges_new[start_index];
        //     // h_csr_edges_new[index++] = h_csr_edges_new[start_index];
        //     prev_value = h_csr_edges_new[start_index];
        //     // index++;
        //     // current_length++;
        // }

        // removing self-loops and duplicate edges
        for(unsigned long i = start_index + 1 ; i < end_index ; i++) {

            if((d_csr_edges[i] == d_csr_edges[id]) || (d_csr_edges[i] == (source + 1))) {
                // if((d_csr_edges[i] == d_csr_edges[id])) {

                d_csr_edges[i] = INFTY;
                // d_source_degrees[source]--;

                // atomicDec(&(d_source_degrees[source]), INFTY);
                // atomicAdd((unsigned long long*)&(d_source_degrees[source]), (unsigned long long int)(-1LL));
                // h_csr_edges_new[index++] = h_csr_edges_new[j];
                // prev_value = d_csr_edges[i];
                // current_length++;

            }

            // index++;

        }

        // d_source_degrees[id] = index - d_csr_offset[id];

        // h_csr_offset[i + 1] = index;
        // h_csr_offset_new[i + 1] = index;

        // }

        // for(unsigned long i = 0 ; i < (vertex_size + 1) ; i++)
        //     h_csr_offset_new[i] = h_csr_offset[i];
        // for(unsigned long i = 0 ; i < batch_size ; i++)
        //     h_csr_edges_new[i] = h_csr_edges[i];

    }

    // if(id == 0) {

    //     printf("\n\nPrinting batched CSR after duplicate removal\nSource degrees\t\t\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //         printf("%lu ", d_source_degrees[i]);
    //         if((i + 1) % vertex_size == 0)
    //             printf("\n");
    //     }
    //     printf("\nCSR offset\t\t\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++) {
    //         printf("%lu ", d_csr_offset[i]);
    //         if((i + 1) % (vertex_size + 1) == 0)
    //             printf("\n");
    //     }
    //     printf("\nCSR edges\t\t\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++) {

    //         for(unsigned long j = d_csr_offset[i] ; j < d_csr_offset[i + 1] ; j++)
    //             printf("%lu ", d_csr_edges[j]);
    //         printf("\n");

    //     }
    //     // for(unsigned long i = 0 ; i < batch_size ; i++) {
    //     //     printf("%lu ", d_csr_edges[i]);
    //     //     if((i + 1) % batch_size == 0)
    //     //         printf("\n");
    //     // }

    // }

}

__global__ void device_update_source_degrees(unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        unsigned long start_index = d_csr_offset[id];
        unsigned long end_index = d_csr_offset[id + 1];

        if(start_index < end_index) {

            for(unsigned long i = start_index ; i < end_index ; i++) {

                if(d_csr_edges[i] == INFTY)
                    d_source_degrees[id]--;

            }

            unsigned long l_index = start_index;
            unsigned long r_index = end_index - 1;

            while(l_index < r_index) {

                while((d_csr_edges[l_index] != INFTY) && (l_index < r_index))
                    l_index++;
                while((d_csr_edges[r_index] == INFTY) && (l_index < r_index))
                    r_index--;

                // printf("ID=%lu, l_index=%lu, r_index=%lu\n", id, l_index, r_index);

                if(l_index < r_index) {
                    // printf("ID=%lu, l_index=%lu, r_index=%lu\n", id, l_index, r_index);

                    unsigned long temp = d_csr_edges[l_index];
                    d_csr_edges[l_index] = d_csr_edges[r_index];
                    d_csr_edges[r_index] = temp;
                    // d_source_degrees[id]--;

                }

            }

        }

    }

    // if(id == 0) {

    //     printf("\n\nPrinting batched CSR after source degrees update\nSource degrees\t\t\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //         printf("%lu ", d_source_degrees[i]);
    //         if((i + 1) % vertex_size == 0)
    //             printf("\n");
    //     }
    //     printf("\nCSR offset\t\t\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++) {
    //         printf("%lu ", d_csr_offset[i]);
    //         if((i + 1) % (vertex_size + 1) == 0)
    //             printf("\n");
    //     }
    //     printf("\nCSR edges\t\t\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++) {

    //         for(unsigned long j = d_csr_offset[i] ; j < d_csr_offset[i + 1] ; j++)
    //             printf("%lu ", d_csr_edges[j]);
    //         printf("\n");

    //     }
    //     // for(unsigned long i = 0 ; i < batch_size ; i++) {
    //     //     printf("%lu ", d_csr_edges[i]);
    //     //     if((i + 1) % batch_size == 0)
    //     //         printf("\n");
    //     // }

    // }

}

__global__ void device_reconstruct_csr(unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        unsigned long start_index = d_csr_offset[id];
        unsigned long end_index = d_csr_offset[id + 1];
        unsigned long index = start_index;
        unsigned long prev_value = d_csr_edges[index++];



        for(unsigned long i = start_index + 1 ; i < end_index ; i++) {

            if(d_csr_edges[i] != prev_value) {

                d_csr_edges[index++] = d_csr_edges[i];
                // h_csr_edges_new[index++] = h_csr_edges_new[j];
                prev_value = d_csr_edges[i];
                // current_length++;

            }

            // index++;

        }

        d_source_degrees[id] = index - d_csr_offset[id];



    }

    // if(id == 0) {

    //     printf("\n\nPrinting batched CSR\nSource degrees\t\t\n");
    //     for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //         printf("%lu ", d_source_degrees[i]);
    //         if((i + 1) % vertex_size == 0)
    //             printf("\n");
    //     }
    //     printf("\nCSR offset\t\t\n");
    //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++) {
    //         printf("%lu ", d_csr_offset[i]);
    //         if((i + 1) % (vertex_size + 1) == 0)
    //             printf("\n");
    //     }
    //     printf("\nCSR edges\t\t\n");
    //     for(unsigned long i = 0 ; i < batch_size ; i++) {
    //         printf("%lu ", d_csr_edges[i]);
    //         if((i + 1) % batch_size == 0)
    //             printf("\n");
    //     }

    // }

}

__global__ void device_generate_csr_batch(unsigned long vertex_size, unsigned long batch_size, unsigned long *d_source, unsigned long *d_destination, unsigned long *d_source_degrees, unsigned long *d_csr_offset, unsigned long *d_csr_edges) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id < vertex_size) {

    //     thrust::host_vector <unsigned long> index(vertex_size);
    //     thrust::fill(h_source_degrees_new.begin(), h_source_degrees_new.end(), 0);

    //     // calculating start and end index of this batch, for use with h_source and h_destination
    //     // unsigned long start_index = batch_number * batch_size;
    //     // unsigned long end_index = start_index + batch_size;
    //     unsigned long start_index = 0;
    //     unsigned long end_index = start_index + batch_size;
    //     // std::cout << "At csr generation, start_index is " << start_index << ", end_index is " << end_index << std::endl;

    //     if(end_index > edge_size)
    //         end_index = edge_size;

    //     // thrust::host_vector <unsigned long> h_source_degrees_new_prev(vertex_size);
    //     // thrust::copy(h_source_degrees_new.begin(), h_source_degrees_new.end(), h_source_degrees_new_prev.begin());

    //     // unsigned long max = h_source_degrees_new[0];
    //     // unsigned long min = h_source_degrees_new[0];
    //     // unsigned long sum = h_source_degrees_new[0];
    //     // unsigned long non_zero_count = 0;

    //     // if(h_source_degrees_new[0])
    //     //     non_zero_count++;

    //     // calculating source degrees of this batch
    //     for(unsigned long i = start_index ; i < end_index ; i++) {

    //         h_source_degrees_new[h_source[i] - 1]++;

    //     }

    //     // std::cout << "Checkpoint 1" << std::endl;

    //     // for(unsigned long i = 1 ; i < vertex_size ; i++) {

    //     //     if(h_source_degrees_new[i] > max)
    //     //         max = h_source_degrees_new[i];

    //     //     if(h_source_degrees_new[i] < min)
    //     //         min = h_source_degrees_new[i];

    //     //     sum += h_source_degrees_new[i];

    //     //     if(h_source_degrees_new[i])
    //     //         non_zero_count++;

    //     // }



    //     // calculating csr offset of this batch
    //     h_csr_offset_new[0] = 0;
    //     h_csr_offset_new[1] = h_source_degrees_new[0];
    //     // h_batch_update_data[0] = 0;
    //     // h_batch_update_data[1] = h_source_degrees_new[0];
    //     for(unsigned long j = 2 ; j < (vertex_size + 1) ; j++) {
    //         h_csr_offset_new[j] = h_csr_offset_new[j - 1] + h_source_degrees_new[j - 1];
    //         // h_batch_update_data[j] = h_batch_update_data[j - 1] + h_source_degrees_new[j - 1];
    //     }

    //     // std::cout << "Checkpoint 2 , start_index is " << start_index << " and end_index is " << end_index << std::endl;


    //     // unsigned long offset = vertex_size + 1;
    //     unsigned long offset = 0;

    //     // calculating csr edges of this batch
    //     for(unsigned long i = start_index ; i < end_index ; i++) {

    //         if(h_source[i] == 1) {
    //             // h_csr_edges_new[index[h_source[i] - 1]++] = h_destination[i];
    //             h_csr_edges_new[index[h_source[i] - 1]] = h_destination[i];
    //             h_batch_update_data[offset + index[h_source[i] - 1]++] = h_destination[i];
    //         }
    //         else {
    //             // h_csr_edges_new[h_csr_offset_new[h_source[i] - 1] + index[h_source[i] - 1]++] = h_destination[i];
    //             h_csr_edges_new[h_csr_offset_new[h_source[i] - 1] + index[h_source[i] - 1]] = h_destination[i];
    //             h_batch_update_data[offset + h_csr_offset_new[h_source[i] - 1] + index[h_source[i] - 1]++] = h_destination[i];
    //         }
    //     }

    //     // comment below section for not sorting each adjacency in the CSR
    //     // unsigned long start_index_edges;
    //     // unsigned long end_index_edges;
    //     // for(unsigned long i = 0 ; i < vertex_size ; i++) {

    //     //     start_index_edges = h_csr_offset_new[i];
    //     //     end_index_edges = h_csr_offset_new[i + 1];

    //     //     if(start_index_edges < end_index_edges)
    //     //         thrust::sort(thrust::host, thrust::raw_pointer_cast(h_csr_edges_new.data()) + start_index_edges, thrust::raw_pointer_cast(h_csr_edges_new.data()) + end_index_edges);

    //     // }

    // }

    if(id == 0) {

        printf("\n\nPrinting batched CSR\nSource degrees\t\t\n");
        for(unsigned long i = 0 ; i < vertex_size ; i++) {
            printf("%lu ", d_source_degrees[i]);
            if((i + 1) % vertex_size == 0)
                printf("\n");
        }
        printf("\nCSR offset\t\t\n");
        for(unsigned long i = 0 ; i < vertex_size + 1 ; i++) {
            printf("%lu ", d_csr_offset[i]);
            if((i + 1) % (vertex_size + 1) == 0)
                printf("\n");
        }
        printf("\nCSR edges\t\t\n");
        for(unsigned long i = 0 ; i < batch_size ; i++) {
            printf("%lu ", d_csr_edges[i]);
            if((i + 1) % batch_size == 0)
                printf("\n");
        }

    }
    // std::cout << std::endl << std::endl << "Printing batched CSR" << std::endl << "Source degrees\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //     std::cout << h_source_degrees_new[i] << " ";
    //     if((i + 1) % vertex_size == 0)
    //         std::cout << std::endl;
    // }


    // std::cout << std::endl << "CSR offset\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < (vertex_size + 1) ; i++) {
    //     std::cout << h_csr_offset_new[i] << " ";
    //     if(((i + 1) % (vertex_size + 1)) == 0)
    //         std::cout << std::endl;
    // }
    // std::cout << std::endl << "CSR edges\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < batch_size ; i++) {
    //     std::cout << h_csr_edges_new[i] << " ";
    //     if((i + 1) % batch_size == 0)
    //         std::cout << std::endl;
    // }

    // std::cout << std::endl << std::endl << std::endl;

}

void generateBatch(unsigned long vertex_size, unsigned long edge_size, thrust::host_vector <unsigned long> &h_csr_offset, thrust::host_vector <unsigned long> &h_csr_edges, unsigned long* h_source_degree, unsigned long* h_prefix_sum_vertex_degrees, unsigned long* h_source, unsigned long* h_destination, unsigned long start_index, unsigned long end_index, unsigned long batch_size, unsigned long current_batch) {

    unsigned long start_index_csr_offset = 0;
    unsigned long end_index_csr_offset = 0;

    for(unsigned long i = 0 ; i < vertex_size ; i++) {

        if(h_csr_offset[i] > start_index) {
            if(i != 0)
                start_index_csr_offset = i - 1;
            else
                start_index_csr_offset = -1;
            break;
        }

    }



    unsigned long current_vertex = start_index_csr_offset;
    unsigned long index = 0;

    for(unsigned long i = start_index ; i < end_index ; i++) {

        while(i >= h_csr_offset[current_vertex + 1]) {
            current_vertex++;
            h_prefix_sum_vertex_degrees[current_vertex + 1] += h_prefix_sum_vertex_degrees[current_vertex];
        }

        h_source[index] = current_vertex + 2;
        h_destination[index++] = h_csr_edges[i];

        h_source_degree[current_vertex + 1]++;

        h_prefix_sum_vertex_degrees[current_vertex + 1]++;

    }



}

// void insert_processing(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long max_degree, thrust::host_vector <unsigned long> &h_source, thrust::host_vector <unsigned long> &h_destination, thrust::host_vector <unsigned long> &h_csr_offset_new, thrust::host_vector <unsigned long> &h_csr_edges_new, thrust::host_vector <unsigned long> &h_source_degrees_new,  thrust::host_vector <unsigned long> &h_source_degrees, unsigned long *d_csr_offset_new_pointer, unsigned long *d_csr_edges_new_pointer, unsigned long *d_source_degrees_new_pointer, unsigned long *d_prefix_sum_edge_blocks_new_pointer) {

//     unsigned long total_edge_blocks_count_batch = 0;

//     thrust::exclusive_scan(h_source_degrees.begin(), h_source_degrees.begin() + vertex_size + 1, h_csr_offset_new.begin());
//     cudaDeviceSynchronize();
//     generate_csr_batch_tester(vertex_size, batch_size, max_degree, h_source, h_destination, h_csr_offset_new, h_csr_edges_new, h_source_degrees_new, h_edge_blocks_count, h_prefix_sum_edge_blocks_new, h_batch_update_data, batch_size, total_batches, kk, space_remaining, &total_edge_blocks_count_batch, &init_time);
//     std::cout << "Generated CSR of random batch" << std::endl;
//     cudaDeviceSynchronize();
//     // unsigned long thread_divider = 10;
//     // thread_blocks = ceil((double(vertex_size / thread_divider)) / THREADS_PER_BLOCK);
//     // device_coo_to_csr<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, d_source_pointer, d_destination_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, edge_size, d_source_degrees_new_pointer, thread_divider);
//     // thrust::copy(h_csr_offset_new.begin(), h_csr_offset_new.end(), d_csr_offset_new.begin());
//     // thrust::copy(h_csr_edges_new.begin(), h_csr_edges_new.end(), d_csr_edges_new.begin());
//     // thrust::copy(h_source_degrees.begin(), h_source_degrees.end(), d_source_degrees_new.begin());
//     thrust::copy(h_csr_offset_new.begin(), h_csr_offset_new.end(), d_csr_offset_new_pointer);
//     thrust::copy(h_csr_edges_new.begin(), h_csr_edges_new.end(), d_csr_edges_new_pointer);
//     thrust::copy(h_source_degrees.begin(), h_source_degrees.end(), d_source_degrees_new_pointer);
//     // thrust::copy(h_destination.begin(), h_destination.begin() + edge_size, d_csr_edges_new.begin());
//     cudaDeviceSynchronize();

//     clock_t deduplication_time;

//     unsigned long thread_blocks;
//     thread_blocks = ceil(double(batch_size) / THREADS_PER_BLOCK);
//     device_remove_batch_duplicates<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);
//     cudaDeviceSynchronize();
//     std::cout << "Removed intra-batch duplicates" << std::endl;
//     // thrust::exclusive_scan(d_source_degrees_new.begin(), d_source_degrees_new.begin() + vertex_size + 1, d_csr_offset_new.begin());
//     // cudaDeviceSynchronize();
//     thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
//     device_update_source_degrees<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);

//     deduplication_time = clock();


//     if(EDGE_BLOCK_SIZE > 100) {

//         unsigned long thread_count_EBC;


//         thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
//         // clock_t preprocessing, kernel_delete;
//         // preprocessing = clock();
//         batched_delete_preprocessing_EC_LD<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer, d_source_vector_pointer);
//         cudaDeviceSynchronize();

//         thrust::exclusive_scan(d_source_vector.begin(), d_source_vector.begin() + vertex_size + 1, d_source_vector.begin());
//         cudaDeviceSynchronize();
//         cudaMemcpy(&thread_count_EBC, d_source_vector_pointer + vertex_size, sizeof(unsigned long),cudaMemcpyDeviceToHost);
//         cudaDeviceSynchronize();
//         // std::cout << "Threads launched is " << thread_count_EBC << std::endl;
//         // thread_blocks = ceil(double(thread_count_EBC) / THREADS_PER_BLOCK);
//         // batched_delete_kernel_EC_LD<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);

//         unsigned long thread_multiplier = EDGE_BLOCK_SIZE / 10;
//         thread_blocks = ceil(double(thread_count_EBC) * thread_multiplier / THREADS_PER_BLOCK);
//         batched_delete_kernel_EC_LD_LB<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, thread_multiplier);


//     }

//     else {

//         // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
//         // batched_delete_kernel_VC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);

//         // unsigned long vertex_LB_factor = 5;
//         // thread_blocks = ceil(double(vertex_size) / vertex_LB_factor / THREADS_PER_BLOCK);
//         // batched_delete_kernel_VC_LB<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, vertex_LB_factor);

//         thread_blocks = ceil(double(batch_size) / THREADS_PER_BLOCK);
//         batched_delete_kernel_EC_HD<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);

//     }

//     std::cout << std::endl << "After batch-insert deduplication" << std::endl << std::endl;
//     printKernelmodded_v2<<< 1, 1>>>(device_vertex_dictionary, vertex_size);


//     // device_inter_batch_deduplication_preprocessing_VC<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);
//     cudaDeviceSynchronize();
//     deduplication_time = clock() - deduplication_time;
//     std::cout << "Removed inter-batch duplicates" << std::endl;

//     thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
//     device_insert_preprocessing<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, batch_size, d_source_degrees_new_pointer, 1, d_edge_blocks_count_pointer);
//     cudaDeviceSynchronize();
//     thrust::exclusive_scan(d_edge_blocks_count.begin(), d_edge_blocks_count.begin() + vertex_size + 1, d_prefix_sum_edge_blocks_new.begin());
//     cudaDeviceSynchronize();

//     cudaMemcpy(&total_edge_blocks_count_batch, d_prefix_sum_edge_blocks_new_pointer + vertex_size, sizeof(unsigned long),cudaMemcpyDeviceToHost);


//     thrust::copy(d_source_degrees_new.begin(), d_source_degrees_new.end(), h_source_degrees_new.begin());
//     cudaDeviceSynchronize();
//     // for(unsigned long i = 0 ; i < vertex_size ; i++) {
//     //     std::thread h_i_p(host_insert_preprocessing, i, vertex_size, thrust::raw_pointer_cast(h_source_degrees_new.data()), thrust::raw_pointer_cast(space_remaining.data()), 0, thrust::raw_pointer_cast(h_edge_blocks_count.data()));

//     //     h_i_p.join();
//     // }
//     // thrust::exclusive_scan(h_edge_blocks_count.begin(), h_edge_blocks_count.begin() + vertex_size + 1, h_prefix_sum_edge_blocks_new.begin());

//     // calculating max and avg degree of batch
//     max_degree_batch = h_source_degrees_new[0];
//     sum_degree_batch = h_source_degrees_new[0];
//     for(unsigned long j = 1 ; j < vertex_size ; j++) {

//         if(h_source_degrees_new[j] > max_degree_batch)
//             max_degree_batch = h_source_degrees_new[j];
//         sum_degree_batch += h_source_degrees_new[j];
//     }

//     average_degree_batch = sum_degree_batch / vertex_size ;

//     std::cout << std::endl << "Max degree of batch is " << max_degree_batch << std::endl;
//     std::cout << "Average degree of batch is " << sum_degree_batch / vertex_size << std::endl << std::endl;
//     std::cout << "Deduplication time (inter-batch) : " <<  (float)deduplication_time/CLOCKS_PER_SEC << " seconds" << std::endl;


//     temp_time = clock();


//     // temp_time = clock();

//     start_index = 0;

//     remaining_edges = edge_size - start_index;

//     if(remaining_edges <= batch_size)
//         end_index = edge_size;
//     else
//         end_index = batch_size;

//     current_batch = end_index - start_index;

//     std::cout << "Current batch is " << current_batch << std::endl;

//     // cudaDeviceSynchronize();
//     // vd_time = clock() - vd_time;
//     std::cout << "Checkpoint" << std::endl;

//     al_time = clock();

//     thread_blocks = ceil(double(h_graph_prop->xDim) / THREADS_PER_BLOCK);
//     // std::cout << "Thread blocks edge init = " << thread_blocks << std::endl;

//     // adjacency_list_init_modded_v5<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, device_vertex_dictionary, kk, current_batch, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);

//     thread_blocks = ceil(double(batch_size) / THREADS_PER_BLOCK);
//     batched_edge_inserts_EC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, device_vertex_dictionary, 1, batch_size, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);
//     batched_edge_inserts_EC_postprocessing<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer, d_prefix_sum_edge_blocks_new_pointer);

//     update_edge_queue<<< 1, 1>>>(total_edge_blocks_count_batch);

// }

// void delete_processing() {

//     thrust::device_vector <unsigned long> d_source(BATCH_SIZE);
//     thrust::device_vector <unsigned long> d_destination(BATCH_SIZE);
//     unsigned long* d_source_pointer = thrust::raw_pointer_cast(d_source.data());
//     unsigned long* d_destination_pointer = thrust::raw_pointer_cast(d_destination.data());
//     cudaDeviceSynchronize();
//     // delete_time = clock();
//     // thrust::copy(h_source.begin(), h_source.begin() + BATCH_SIZE, d_source.begin());
//     // thrust::copy(h_destination.begin(), h_destination.begin() + BATCH_SIZE, d_destination.begin());

//     unsigned long h_data_structure_edge_block_count;
//     cudaDeviceSynchronize();

//     device_prefix_sum_calculation_preprocessing<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_prefix_sum_edge_blocks_new_pointer);
//     cudaDeviceSynchronize();
//     thrust::exclusive_scan(thrust::device, d_prefix_sum_edge_blocks_new_pointer, d_prefix_sum_edge_blocks_new_pointer + vertex_size + 1, d_prefix_sum_edge_blocks_new_pointer);
//     cudaDeviceSynchronize();
//     cudaMemcpy(&h_data_structure_edge_block_count, d_prefix_sum_edge_blocks_new_pointer + vertex_size, sizeof(unsigned long),cudaMemcpyDeviceToHost);


//     // std::cout << "Checkpoint delete" << std::endl;
//     unsigned long* d_thread_count_vector_pointer = thrust::raw_pointer_cast(d_thread_count_vector.data());

//     unsigned long h_total_threads, *d_total_threads;
//     d_total_threads = d_thread_count_vector_pointer + vertex_size;



//     delete_time = clock();
//     // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
//     // batched_delete_preprocessing<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer);
//     // thread_blocks = ceil(double(total_edge_blocks_count_batch) / THREADS_PER_BLOCK);
//     // batched_delete_kernel_v1<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, total_edge_blocks_count_batch, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer);

//     unsigned long decision_tree_true = (average_degree_batch < (max_degree_batch / 20)) && (BATCH_SIZE <= 10000000);

//     if((decision_tree_true)) {

//         thrust::copy(h_source.begin(), h_source.begin() + BATCH_SIZE, d_source.begin());
//         thrust::copy(h_destination.begin(), h_destination.begin() + BATCH_SIZE, d_destination.begin());

//         // Below is the code for edge-centric batch deletes
//         thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
//         batched_delete_preprocessing_edge_centric<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer);
//         thread_blocks = ceil(double(batch_size) / THREADS_PER_BLOCK);
//         batched_delete_kernel_edge_centric<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_pointer, d_destination_pointer);

//     }

//     // Below is the test code for the new bit string based parallel edge block approach

//     // remove this stupid else if and make it else later
//     else {

//         // cudaMemcpy(d_batch_update_data, h_batch_update_data, (batch_size) * sizeof(unsigned long), cudaMemcpyHostToDevice);

//         // // launch with number of threads equal to the vertices in the graph
//         // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
//         // // clock_t preprocessing, kernel_delete;
//         // // preprocessing = clock();
//         // batched_delete_preprocessing_edge_block_centric<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer, d_source_vector_pointer);
//         // // cudaDeviceSynchronize();
//         // // preprocessing = clock() - preprocessing;

//         // // launch with number of threads equal to the edge blocks used by the data structure
//         // // thread_blocks = ceil(double(h_prefix_sum_edge_blocks_new[vertex_size]) / THREADS_PER_BLOCK);
//         // thread_blocks = ceil(double(h_data_structure_edge_block_count) / THREADS_PER_BLOCK);
//         // // kernel_delete = clock();
//         // batched_delete_kernel_edge_block_centric<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);
//         // // cudaDeviceSynchronize();
//         // // kernel_delete = clock() - kernel_delete;

//         if(EDGE_BLOCK_SIZE > 100) {

//             unsigned long thread_count_EBC;

//             // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
//             // // clock_t preprocessing, kernel_delete;
//             // // preprocessing = clock();
//             // batched_delete_preprocessing_edge_block_centric_v4<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer, d_source_vector_pointer);
//             // cudaDeviceSynchronize();
//             // thrust::exclusive_scan(d_source_vector.begin(), d_source_vector.begin() + vertex_size + 1, d_source_vector.begin());
//             // cudaDeviceSynchronize();

//             // cudaMemcpy(&thread_count_EBC, d_source_vector_pointer + vertex_size, sizeof(unsigned long),cudaMemcpyDeviceToHost);

//             // // new edge_block_centric deletes
//             // thread_blocks = ceil(double(thread_count_EBC) / THREADS_PER_BLOCK);
//             // // kernel_delete = clock();
//             // batched_delete_kernel_edge_block_centric_v4<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);

//             thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
//             // clock_t preprocessing, kernel_delete;
//             // preprocessing = clock();
//             batched_delete_preprocessing_EC_LD<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer, d_source_vector_pointer);
//             cudaDeviceSynchronize();

//             thrust::exclusive_scan(d_source_vector.begin(), d_source_vector.begin() + vertex_size + 1, d_source_vector.begin());
//             cudaDeviceSynchronize();
//             cudaMemcpy(&thread_count_EBC, d_source_vector_pointer + vertex_size, sizeof(unsigned long),cudaMemcpyDeviceToHost);
//             cudaDeviceSynchronize();
//             // std::cout << "Threads launched is " << thread_count_EBC << std::endl;
//             // thread_blocks = ceil(double(thread_count_EBC) / THREADS_PER_BLOCK);
//             // batched_delete_kernel_EC_LD<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);
//             unsigned long thread_multiplier = EDGE_BLOCK_SIZE / 10;
//             thread_blocks = ceil(double(thread_count_EBC) * thread_multiplier / THREADS_PER_BLOCK);
//             batched_delete_kernel_EC_LD_LB<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, thread_multiplier);

//         }

//         else {

//             // basic vertex_centric
//             // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
//             // batched_delete_kernel_VC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);

//             // unsigned long vertex_LB_factor = 5;
//             // thread_blocks = ceil(double(vertex_size) / vertex_LB_factor / THREADS_PER_BLOCK);
//             // batched_delete_kernel_VC_LB<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, vertex_LB_factor);

//             thread_blocks = ceil(double(batch_size) / THREADS_PER_BLOCK);
//             batched_delete_kernel_EC_HD<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);


//             // load-balanced vertex_centric
//             // unsigned long thread_count_VC_LB;
//             // unsigned long thread_divider = 1;
//             // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
//             // batched_delete_preprocessing_LB<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer, d_source_vector_pointer);
//             // cudaDeviceSynchronize();
//             // thrust::exclusive_scan(d_source_vector.begin(), d_source_vector.begin() + vertex_size + 1, d_source_vector.begin());
//             // cudaDeviceSynchronize();
//             // cudaMemcpy(&thread_count_VC_LB, d_source_vector_pointer + vertex_size, sizeof(unsigned long),cudaMemcpyDeviceToHost);
//             // cudaDeviceSynchronize();
//             // std::cout << "Thread count for VC_LB is " << double(thread_count_VC_LB) / thread_divider << std::endl;
//             // thread_blocks = ceil((double(thread_count_VC_LB) / thread_divider) / THREADS_PER_BLOCK);
//             // // kernel_delete = clock();
//             // // batched_delete_kernel_edge_block_centric_v4<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);
//             // batched_delete_kernel_LB<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, thread_divider);


//         }
//     }


//     cudaDeviceSynchronize();
//     delete_time = clock() - delete_time;

// }

void memory_usage() {

    // show memory usage of GPU

    size_t free_byte ;

    size_t total_byte ;

    cudaMemGetInfo( &free_byte, &total_byte ) ;
    // cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

    // if ( cudaSuccess != cuda_status ){

    //     printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );

    //     exit(1);

    // }

    double free_db = (double)free_byte ;

    double total_db = (double)total_byte ;

    double used_db = total_db - free_db ;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

           used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

}

unsigned long calculate_log_base2(unsigned long value) {

    while(value > 1)
        return calculate_log_base2(value / 2) + 1;

    return 0;
}

unsigned long calculate_power_base2(unsigned long value) {

    while(value > 0)
        return calculate_power_base2(--value) * 2;

    return 1;
}

__global__ void compactionVertexCentric(unsigned long totalvertices, struct vertex_dictionary_structure *device_vertex_dictionary){
    unsigned long id = blockDim.x * blockIdx.x + threadIdx.x;

    if(id >= totalvertices) return;
    // if(id != 0) return;
//    printf("%ld\n", totalvertices);

//    printf("start FOR VERTEX %lu\n", id);


    struct edge_block *curr = device_vertex_dictionary->edge_block_address[id];
    struct edge_block *root = device_vertex_dictionary->edge_block_address[id];
    unsigned long bitString;
    unsigned long parent_bit_string;
    struct edge_block *parent = NULL;
    unsigned int push_index_for_edge_queue;

//    printf("%p\n", curr);
//    printf("%p\n", root);

    if(!curr) return;

    struct edge_block *swapping_block = device_vertex_dictionary->last_insert_edge_block[id];
//    printf("%p\n", swapping_block);

    unsigned long total_edge_blocks = device_vertex_dictionary->edge_block_count[id];
//    printf("%lu\n", total_edge_blocks);

    long curr_edge_block_index = 0;
    long last_edge_block_index = total_edge_blocks - 1;
    long last_swap_offset = EDGE_BLOCK_SIZE - 1;
    while(last_swap_offset >= 0 && (swapping_block->edge_block_entry[last_swap_offset].destination_vertex == INFTY || swapping_block->edge_block_entry[last_swap_offset].destination_vertex == 0)){
        --last_swap_offset;
    }
//    printf("For %lu last swap offset is: %ld\n", id, last_swap_offset);

//    if(id == 0) printf("%ld\n", curr_edge_block_index);
//    if(id == 0) printf("%ld\n", last_edge_block_index);
//    if(id == 0) printf("%ld\n", last_swap_offset);


    while(curr_edge_block_index < last_edge_block_index){
//        if(id == 0) printf("current edge block index for vertex %lu: %lu\n", id, curr_edge_block_index);
//        if(id == 0) printf("last edge block index for vertex %lu: %lu\n", id, last_edge_block_index);

        for(unsigned long i = 0; i < EDGE_BLOCK_SIZE; ++i){
            unsigned long e = (curr->edge_block_entry[i]).destination_vertex;

            if(e != INFTY) continue;

//             printf("%lu for vertex %lu\n", e, id);

            // deleted edge found
            int edge_swapped_flag = 0;

            while(edge_swapped_flag == 0){
//                 printf("here FOR VERTEX %lu\n", id);
                if(curr_edge_block_index == last_edge_block_index) break;
//                 printf("here for vertex %lu\n", id);
//                 if(id == 0) {
//                     printf("%ld\n", last_swap_offset);
//                     break;
//                 }
//                 printf("1here for %lu\n", id);
//                 printf("%ld\n", last_swap_offset);
                if(last_swap_offset == -1){
//                     printf("here for %lu\n", id);
                    push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1);
//                     if(d_e_queue.rear >= EDGE_PREALLOCATE_LIST_SIZE) printf("%ld\n", id);
                    push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
                    d_e_queue.edge_block_address[push_index_for_edge_queue] = swapping_block;
                    swapping_block->lptr = NULL;
                    swapping_block->rptr = NULL;
//CHANGE                    //  device_vertex_dictionary->active_edge_count[id] -= swapping_block->active_edge_count;
                    //  swapping_block->active_edge_count = 0;
                    swapping_block = swapping_block->level_order_predecessor;
                    device_vertex_dictionary->last_insert_edge_block[id] = swapping_block;
//                     printf("%p\n", swapping_block);
                    swapping_block->lptr = NULL;
                    swapping_block->rptr = NULL;
                    last_swap_offset = EDGE_BLOCK_SIZE - 1;

                    // freeing the parent to child relation
                    if(last_edge_block_index & 1) parent_bit_string = bit_string_lookup[last_edge_block_index / 2];
                    else parent_bit_string = bit_string_lookup[last_edge_block_index / 2 - 1];

                    parent = traverse_bit_string(root, parent_bit_string);

                    if(last_edge_block_index & 1) parent->lptr = NULL;
                    else parent->rptr = NULL;

                    --last_edge_block_index;
                    device_vertex_dictionary->edge_block_count[id] -= 1;
                    if(curr_edge_block_index == last_edge_block_index) break;
                }
                else{
                    while(last_swap_offset >= 0 && swapping_block->edge_block_entry[last_swap_offset].destination_vertex == INFTY /*&& swapping_block->edge_block_entry[last_swap_offset].destination_vertex != 0*/) {
                        //CHANGE                       //  swapping_block->active_edge_count -= 1;
                        //CHANGE                       //  device_vertex_dictionary->active_edge_count[id] -= 1;
                        --last_swap_offset;
                    }

//                     if(id == 2) printf("%ld\n", last_swap_offset);

                    if(last_swap_offset < 0){
                        push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1);
                        push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
                        d_e_queue.edge_block_address[push_index_for_edge_queue] = swapping_block;
                        swapping_block->lptr = NULL;
                        swapping_block->rptr = NULL;
                        //CHANGE                       //  device_vertex_dictionary->active_edge_count[id] -= swapping_block->active_edge_count;
                        //  swapping_block->active_edge_count = 0;
                        swapping_block = swapping_block->level_order_predecessor;
                        device_vertex_dictionary->last_insert_edge_block[id] = swapping_block;
//                         printf("%p\n", swapping_block);
                        swapping_block->lptr = NULL;
                        swapping_block->rptr = NULL;
                        last_swap_offset = EDGE_BLOCK_SIZE - 1;

                        // freeing the parent to child relation
                        if(last_edge_block_index & 1) parent_bit_string = bit_string_lookup[last_edge_block_index / 2];
                        else parent_bit_string = bit_string_lookup[last_edge_block_index / 2 - 1];

                        parent = traverse_bit_string(root, parent_bit_string);

                        if(last_edge_block_index & 1) parent->lptr = NULL;
                        else parent->rptr = NULL;

                        --last_edge_block_index;
                        device_vertex_dictionary->edge_block_count[id] -= 1;
                    }
                    else{
                        if(curr_edge_block_index == last_edge_block_index) break;

                        curr->edge_block_entry[i].destination_vertex = swapping_block->edge_block_entry[last_swap_offset].destination_vertex;
                        swapping_block->edge_block_entry[last_swap_offset].destination_vertex = 0;
                        --last_swap_offset;
                        //CHANGE                       //  swapping_block->active_edge_count -= 1;
                        //CHANGE                       //  device_vertex_dictionary->active_edge_count[id] -= 1;
                        edge_swapped_flag = 1;

                        if(last_swap_offset == -1){
                            push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1);
                            push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
                            d_e_queue.edge_block_address[push_index_for_edge_queue] = swapping_block;
                            swapping_block->lptr = NULL;
                            swapping_block->rptr = NULL;
                            //CHANGE                           //  device_vertex_dictionary->active_edge_count[id] -= swapping_block->active_edge_count;
                            //  swapping_block->active_edge_count = 0;
                            swapping_block = swapping_block->level_order_predecessor;
                            device_vertex_dictionary->last_insert_edge_block[id] = swapping_block;
//                         printf("%p\n", swapping_block);
                            swapping_block->lptr = NULL;
                            swapping_block->rptr = NULL;
                            last_swap_offset = EDGE_BLOCK_SIZE - 1;

                            // freeing the parent to child relation
                            if(last_edge_block_index & 1) parent_bit_string = bit_string_lookup[last_edge_block_index / 2];
                            else parent_bit_string = bit_string_lookup[last_edge_block_index / 2 - 1];

                            parent = traverse_bit_string(root, parent_bit_string);

                            if(last_edge_block_index & 1) parent->lptr = NULL;
                            else parent->rptr = NULL;

                            --last_edge_block_index;
                            device_vertex_dictionary->edge_block_count[id] -= 1;
                        }
                    }
                }
            }

            if(curr_edge_block_index == last_edge_block_index) break;
        }

        if(curr_edge_block_index == last_edge_block_index) break;

        ++curr_edge_block_index;
        bitString = bit_string_lookup[curr_edge_block_index];
        curr = traverse_bit_string(root, bitString);
    }



//    printf("Reached out of the while loop\n");
    if(curr_edge_block_index == last_edge_block_index){
//        printf("%lu\n", curr_edge_block_index);
//        printf("here for vertex %lu\n", id);
        last_swap_offset = EDGE_BLOCK_SIZE - 1;

//        printf("%lu %lu\n", curr_edge_block_index, last_swap_offset);
        while(last_swap_offset >= 0 && (curr->edge_block_entry[last_swap_offset].destination_vertex == INFTY || curr->edge_block_entry[last_swap_offset].destination_vertex == 0)) {
            if(curr->edge_block_entry[last_swap_offset].destination_vertex == INFTY) {
                //CHANGE              // device_vertex_dictionary->active_edge_count[id] -= 1;
                // curr->active_edge_count -= 1;
                curr->edge_block_entry[last_swap_offset].destination_vertex = 0;
            }
            --last_swap_offset;
        }
        long start = 0;
//        printf("%lu for vertex %lu\n", last_swap_offset, id);
        while(start < EDGE_BLOCK_SIZE && start < last_swap_offset && last_swap_offset >= 0 && last_swap_offset < EDGE_BLOCK_SIZE){
//            printf("%lu, %lu for vertex %lu\n", start, last_swap_offset, id);
            if(curr->edge_block_entry[start].destination_vertex != INFTY) ++start;
            else if(curr->edge_block_entry[last_swap_offset].destination_vertex == INFTY) {
                curr->edge_block_entry[last_swap_offset].destination_vertex = 0;
                //CHANGE               // device_vertex_dictionary->active_edge_count[id] -= 1;
                --last_swap_offset;
                // --curr->active_edge_count;
            }
            else{
                curr->edge_block_entry[start].destination_vertex = curr->edge_block_entry[last_swap_offset].destination_vertex;
                curr->edge_block_entry[last_swap_offset].destination_vertex = 0;
                // curr->active_edge_count -= 1;
                //CHANGE               // device_vertex_dictionary->active_edge_count[id] -= 1;
                --last_swap_offset;
//                printf("%lu\n", last_swap_offset);
            }
        }
    }

    curr->lptr = NULL;
    curr->rptr = NULL;
//    printf("here2 FOR VERTEX %lu\n", id);
    device_vertex_dictionary->last_insert_edge_block[id] = swapping_block;
    if(last_swap_offset >= 0){
        for(unsigned long i = 0; i < EDGE_BLOCK_SIZE - 1; ++i){
            unsigned long e = (curr->edge_block_entry[i]).destination_vertex;
            if(e > 0) ++last_swap_offset;
            else break;
        }
        if(last_swap_offset == EDGE_BLOCK_SIZE) last_swap_offset = 0;
    }
    device_vertex_dictionary->last_insert_edge_offset[id] = last_swap_offset;

    if(last_swap_offset < 0){
        push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1);
        push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
        d_e_queue.edge_block_address[push_index_for_edge_queue] = curr;
        device_vertex_dictionary->last_insert_edge_block[id] = curr->level_order_predecessor;
        device_vertex_dictionary->last_insert_edge_offset[id] = EDGE_BLOCK_SIZE - 1;
        device_vertex_dictionary->edge_block_count[id] -= 1;
//        printf("here3 FOR VERTEX %lu\n", id);
        if(curr == root){
//            printf("here4 FOR VERTEX %lu\n", id);
            curr->level_order_predecessor = NULL;
            device_vertex_dictionary->edge_block_address[id] = NULL;
            device_vertex_dictionary->active_edge_count[id] = 0;
            device_vertex_dictionary->last_insert_edge_offset[id] = 0;
            device_vertex_dictionary->edge_block_count[id] = 0;
            device_vertex_dictionary->last_insert_edge_block[id] = NULL;
        }
    }

//    printf("IN THE KERNEL FOR VERTEX %lu\n", id);
}

__global__ void printQueuePtrs(){
    d_e_queue.rear %= EDGE_PREALLOCATE_LIST_SIZE;
    d_e_queue.front %= EDGE_PREALLOCATE_LIST_SIZE;
    printf("%u\n", d_e_queue.rear);
    printf("%u\n", d_e_queue.front);
}

__global__ void recoveredMemoryKernel1(unsigned int* num_blocks_before, double *totalMemoryBeforeCompaction, unsigned int *totalEdgeBlocks,
                                       struct vertex_dictionary_structure *device_vertex_dictionary){
    d_e_queue.rear %= EDGE_PREALLOCATE_LIST_SIZE;
    d_e_queue.front %= EDGE_PREALLOCATE_LIST_SIZE;
    printf("Rear Before: %u\n", d_e_queue.rear);
    printf("Front Before: %u\n", d_e_queue.front);
    if (d_e_queue.rear >= d_e_queue.front) *num_blocks_before = d_e_queue.rear - d_e_queue.front;
    else *num_blocks_before = EDGE_PREALLOCATE_LIST_SIZE - d_e_queue.front + d_e_queue.rear;

    printf("Number of blocks in queue before compaction: %u\n", *num_blocks_before);
    *totalMemoryBeforeCompaction = (double)sizeof(device_vertex_dictionary) + ((double)*totalEdgeBlocks * ((double)sizeof(struct edge_block)));
    printf("Total Memory of GraphVine Before Compaction: %f\n", *totalMemoryBeforeCompaction);
}

__global__ void recoveredMemoryKernel2(unsigned int* num_blocks_before, unsigned int* num_blocks_after){
    d_e_queue.rear %= EDGE_PREALLOCATE_LIST_SIZE;
    d_e_queue.front %= EDGE_PREALLOCATE_LIST_SIZE;
    printf("Rear After: %u\n", d_e_queue.rear);
    printf("Front After: %u\n", d_e_queue.front);
    if (d_e_queue.rear >= d_e_queue.front) *num_blocks_after = d_e_queue.rear - d_e_queue.front;
    else *num_blocks_after = EDGE_PREALLOCATE_LIST_SIZE - d_e_queue.front + d_e_queue.rear;

    printf("Number of blocks in queue after compaction: %u\n", *num_blocks_after);

    unsigned int num_compacted_blocks = *num_blocks_after - *num_blocks_before;
    printf("Number of compacted blocks: %u\n", num_compacted_blocks);
    unsigned int recovered_memory = (sizeof(struct edge_block) * num_compacted_blocks) / (1024 * 1024);
    printf("Recovered Memory: %u\n", recovered_memory);
    printf("\n");
}

__global__ void recoveredMemoryPercentage(unsigned int* num_blocks_before, unsigned int* num_blocks_after, unsigned int* totalEdgeBlocks,
                                          struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long totalvertices, double *totalMemoryBeforeCompaction, unsigned int* initial_fragmentation, unsigned int* final_fragmentation){
    d_e_queue.rear %= EDGE_PREALLOCATE_LIST_SIZE;
    d_e_queue.front %= EDGE_PREALLOCATE_LIST_SIZE;
    printf("Rear After: %u\n", d_e_queue.rear);
    printf("Front After: %u\n", d_e_queue.front);
    if (d_e_queue.rear >= d_e_queue.front) *num_blocks_after = d_e_queue.rear - d_e_queue.front;
    else *num_blocks_after = EDGE_PREALLOCATE_LIST_SIZE - d_e_queue.front + d_e_queue.rear;

    printf("Number of blocks in queue after compaction: %u\n", *num_blocks_after);
    unsigned int num_compacted_blocks = *num_blocks_after - *num_blocks_before;
    printf("Number of compacted blocks: %u\n", num_compacted_blocks);
    unsigned int recovered_memory = (sizeof(struct edge_block) * num_compacted_blocks) / (1024 * 1024);
    printf("Recovered Memory: %u\n", recovered_memory);
    printf("\n");

    double totalMemoryAfterCompaction = 0.0;

    for(unsigned long i = 0; i < totalvertices; ++i)
        totalMemoryAfterCompaction += (double)device_vertex_dictionary->edge_block_count[i];

    totalMemoryAfterCompaction *= (double) sizeof(struct edge_block);
    totalMemoryAfterCompaction += (double) sizeof(device_vertex_dictionary);
    double freedMemory = *totalMemoryBeforeCompaction - totalMemoryAfterCompaction;

    printf("Total Memory After Compaction: %f\n", totalMemoryAfterCompaction);

    double memory_reclaimed = freedMemory / *totalMemoryBeforeCompaction * 100;
    printf("Reclaimed Memory: %f %%\n", memory_reclaimed);

    double fragmentation_reduction = ((double)( *initial_fragmentation - *final_fragmentation) / *initial_fragmentation) * 100;
    printf("Initial Fragmentation: %u\n", *initial_fragmentation);
    printf("Final Fragmentation: %u\n", *final_fragmentation);
    printf("Fragmentation Reduction: %f %%\n", fragmentation_reduction);
    printf("\n");
}

__global__ void helper1(unsigned int i, unsigned int *edge_block_count_per_vertex, unsigned int *totalEdgeBlocks){
    *totalEdgeBlocks = edge_block_count_per_vertex[i];
}

__global__ void findBlocksPerVertex(unsigned long totalvertices,
                                    struct vertex_dictionary_structure *device_vertex_dictionary,
                                    unsigned int *blocks_per_vertex){
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= totalvertices) return;

    blocks_per_vertex[id] = (unsigned int) device_vertex_dictionary->edge_block_count[id];
}

__global__ void findTotalEdgeBlocks(unsigned int totalvertices,
                                    unsigned int *total_edge_blocks,
                                    unsigned int *blocks_per_vertex_prefixSum,
                                    unsigned int *blocks_per_vertex){
    *total_edge_blocks = blocks_per_vertex_prefixSum[totalvertices - 1] + blocks_per_vertex[totalvertices - 1];
    blocks_per_vertex_prefixSum[totalvertices]  = blocks_per_vertex_prefixSum[totalvertices - 1] + blocks_per_vertex[totalvertices - 1];

//    for(unsigned int i = 0; i <= totalvertices; ++i){
//        printf("%u ", blocks_per_vertex_prefixSum[i]);
//    }

    // printf("\n%u\n", *total_edge_blocks);
}

__global__ void compactionVertexCentricLevelOrderQueue(unsigned long totalvertices,
                                                       struct vertex_dictionary_structure *device_vertex_dictionary,
                                                       unsigned int *blocks_per_vertex_start,
                                                       struct edge_block **level_order_queue){
    unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

    if(id >= totalvertices) return;

//    if(id != 0) return;
//    printf("start FOR VERTEX %lu\n", id);

    struct edge_block *curr = device_vertex_dictionary->edge_block_address[id];
    struct edge_block *root = device_vertex_dictionary->edge_block_address[id];
    unsigned long bitString;
    unsigned long parent_bit_string;
    struct edge_block *parent = NULL;
    unsigned int push_index_for_edge_queue;
    unsigned int start_for_level_queue = blocks_per_vertex_start[id];
//    printf("%ld\n", start_for_level_queue);

//    printf("%p\n", curr);
//    printf("%p\n", root);

    if(!curr) return;

    struct edge_block *swapping_block = device_vertex_dictionary->last_insert_edge_block[id];
//    printf("%p\n", swapping_block);

    unsigned long total_edge_blocks = device_vertex_dictionary->edge_block_count[id];
//    printf("%lu\n", total_edge_blocks);

    long curr_edge_block_index = 0;
    long last_edge_block_index = total_edge_blocks - 1;
    long last_swap_offset = EDGE_BLOCK_SIZE - 1;
    while(last_swap_offset >= 0 && (swapping_block->edge_block_entry[last_swap_offset].destination_vertex == INFTY || swapping_block->edge_block_entry[last_swap_offset].destination_vertex == 0)){
        swapping_block->edge_block_entry[last_swap_offset].destination_vertex = 0;
        --last_swap_offset;
    }
//    printf("For %lu last swap offset is: %ld\n", id, last_swap_offset);


//    if(id == 0) printf("%ld\n", curr_edge_block_index);
//    if(id == 0) printf("%ld\n", last_edge_block_index);
//    if(id == 0) printf("%ld\n", last_swap_offset);

    while(curr_edge_block_index < last_edge_block_index){
//        if(id == 2) printf("current edge block index for vertex %lu: %lu\n", id, curr_edge_block_index);
//        if(id == 2) printf("last edge block index for vertex %lu: %lu\n", id, last_edge_block_index);

        for(unsigned long i = 0; i < EDGE_BLOCK_SIZE; ++i){
            unsigned long e = (curr->edge_block_entry[i]).destination_vertex;

            if(e != INFTY) continue;

//             printf("%lu for vertex %lu\n", e, id);

            // deleted edge found
            int edge_swapped_flag = 0;

            while(edge_swapped_flag == 0){
//                 printf("here FOR VERTEX %lu\n", id);
                if(curr_edge_block_index == last_edge_block_index) break;
//                 printf("here for vertex %lu\n", id);
//                 if(id == 0) {
//                     printf("%ld\n", last_swap_offset);
//                     break;
//                 }
//                 printf("1here for %lu\n", id);
//                 printf("%ld\n", last_swap_offset);
                if(last_swap_offset == -1){
//                  printf("here for %lu\n", id);
                    push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1);
//                  if(d_e_queue.rear >= EDGE_PREALLOCATE_LIST_SIZE) printf("%ld\n", id);
                    push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
                    d_e_queue.edge_block_address[push_index_for_edge_queue] = swapping_block;
                    swapping_block->lptr = NULL;
                    swapping_block->rptr = NULL;
                    // device_vertex_dictionary->active_edge_count[id] -= swapping_block->active_edge_count;
                    // swapping_block->active_edge_count = 0;
                    swapping_block = swapping_block->level_order_predecessor;
//                  printf("%p\n", swapping_block);
                    swapping_block->lptr = NULL;
                    swapping_block->rptr = NULL;
                    last_swap_offset = EDGE_BLOCK_SIZE - 1;


                    // freeing the parent to child relation
                    if(last_edge_block_index & 1) parent_bit_string = bit_string_lookup[last_edge_block_index / 2];
                    else parent_bit_string = bit_string_lookup[last_edge_block_index / 2 - 1];

                    parent = traverse_bit_string(root, parent_bit_string);

                    if(last_edge_block_index & 1) parent->lptr = NULL;
                    else parent->rptr = NULL;


                    --last_edge_block_index;
                    device_vertex_dictionary->edge_block_count[id] -= 1;
                    if(curr_edge_block_index == last_edge_block_index) break;
                }
                else{
                    while(last_swap_offset >= 0 && swapping_block->edge_block_entry[last_swap_offset].destination_vertex == INFTY /*&& swapping_block->edge_block_entry[last_swap_offset].destination_vertex != 0*/) {
                        // swapping_block->active_edge_count -= 1;
                        // device_vertex_dictionary->active_edge_count[id] -= 1;
                        swapping_block->edge_block_entry[last_swap_offset].destination_vertex = 0;
                        --last_swap_offset;
                    }

//                     if(id == 2) printf("%ld\n", last_swap_offset);

                    if(last_swap_offset < 0){
                        push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1);
                        push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
                        d_e_queue.edge_block_address[push_index_for_edge_queue] = swapping_block;
                        swapping_block->lptr = NULL;
                        swapping_block->rptr = NULL;
                        // device_vertex_dictionary->active_edge_count[id] -= swapping_block->active_edge_count;
                        // swapping_block->active_edge_count = 0;
                        swapping_block = swapping_block->level_order_predecessor;
                        device_vertex_dictionary->last_insert_edge_block[id] = swapping_block;
//                         printf("%p\n", swapping_block);
                        swapping_block->lptr = NULL;
                        swapping_block->rptr = NULL;
                        last_swap_offset = EDGE_BLOCK_SIZE - 1;


                        // freeing the parent to child relation
                        if(last_edge_block_index & 1) parent_bit_string = bit_string_lookup[last_edge_block_index / 2];
                        else parent_bit_string = bit_string_lookup[last_edge_block_index / 2 - 1];

                        parent = traverse_bit_string(root, parent_bit_string);

                        if(last_edge_block_index & 1) parent->lptr = NULL;
                        else parent->rptr = NULL;


                        --last_edge_block_index;
                        device_vertex_dictionary->edge_block_count[id] -= 1;
                    }
                    else{
                        if(curr_edge_block_index == last_edge_block_index) break;

                        curr->edge_block_entry[i].destination_vertex = swapping_block->edge_block_entry[last_swap_offset].destination_vertex;
                        swapping_block->edge_block_entry[last_swap_offset].destination_vertex = 0;
                        --last_swap_offset;
                        // swapping_block->active_edge_count -= 1;
                        // device_vertex_dictionary->active_edge_count[id] -= 1;
                        edge_swapped_flag = 1;

                        if(last_swap_offset == -1){
                            push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1);
                            push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
                            d_e_queue.edge_block_address[push_index_for_edge_queue] = swapping_block;
//                            swapping_block->lptr = NULL;
//                            swapping_block->rptr = NULL;
                            // device_vertex_dictionary->active_edge_count[id] -= swapping_block->active_edge_count;
                            // swapping_block->active_edge_count = 0;
                            swapping_block = swapping_block->level_order_predecessor;
                            device_vertex_dictionary->last_insert_edge_block[id] = swapping_block;
//                         printf("%p\n", swapping_block);
                            swapping_block->lptr = NULL;
                            swapping_block->rptr = NULL;
                            last_swap_offset = EDGE_BLOCK_SIZE - 1;


                            // freeing the parent to child relation
                            if(last_edge_block_index & 1) parent_bit_string = bit_string_lookup[last_edge_block_index / 2];
                            else parent_bit_string = bit_string_lookup[last_edge_block_index / 2 - 1];

                            parent = traverse_bit_string(root, parent_bit_string);

                            if(last_edge_block_index & 1) parent->lptr = NULL;
                            else parent->rptr = NULL;


                            --last_edge_block_index;
                            device_vertex_dictionary->edge_block_count[id] -= 1;
                        }
                    }
                }
            }

            if(curr_edge_block_index == last_edge_block_index) break;
        }

        if(curr_edge_block_index == last_edge_block_index) break;

        unsigned int idx1 = start_for_level_queue + 2 * ((unsigned int) curr_edge_block_index);
        unsigned int idx2 = start_for_level_queue + 2 * ((unsigned int) curr_edge_block_index) + 1;
//        printf("idx1: %u idx2: %u\n", idx1, idx2);
        if(curr->lptr != NULL) level_order_queue[idx1] = curr->lptr;
        if(curr->rptr != NULL) level_order_queue[idx2] = curr->rptr;

//        ++curr_edge_block_index;
//        bitString = bit_string_lookup[curr_edge_block_index];
//        curr = traverse_bit_string(root, bitString);
        unsigned int idx3 = start_for_level_queue + curr_edge_block_index;
//        printf("idx3: %u\n", idx3);
        curr = level_order_queue[idx3];
        ++curr_edge_block_index;
    }
//    printf("Reached out of the while loop\n");
    if(curr_edge_block_index == last_edge_block_index){
//        printf("%lu\n", curr_edge_block_index);
//        printf("here for vertex %lu\n", id);
        last_swap_offset = EDGE_BLOCK_SIZE - 1;

//        printf("%lu %lu\n", curr_edge_block_index, last_swap_offset);
        while(last_swap_offset >= 0 && (curr->edge_block_entry[last_swap_offset].destination_vertex == INFTY || curr->edge_block_entry[last_swap_offset].destination_vertex == 0)) {
            if(curr->edge_block_entry[last_swap_offset].destination_vertex == INFTY) {
                // device_vertex_dictionary->active_edge_count[id] -= 1;
                // curr->active_edge_count -= 1;
                curr->edge_block_entry[last_swap_offset].destination_vertex = 0;
            }
            --last_swap_offset;
        }
        long start = 0;
//        printf("%lu for vertex %lu\n", last_swap_offset, id);
        while(start < EDGE_BLOCK_SIZE && start < last_swap_offset && last_swap_offset >= 0 && last_swap_offset < EDGE_BLOCK_SIZE){
//            printf("%lu, %lu for vertex %lu\n", start, last_swap_offset, id);
            if(curr->edge_block_entry[start].destination_vertex != INFTY) ++start;
            else if(curr->edge_block_entry[last_swap_offset].destination_vertex == INFTY) {
                curr->edge_block_entry[last_swap_offset].destination_vertex = 0;
                // device_vertex_dictionary->active_edge_count[id] -= 1;
                --last_swap_offset;
                // --curr->active_edge_count;
            }
            else{
                curr->edge_block_entry[start].destination_vertex = curr->edge_block_entry[last_swap_offset].destination_vertex;
                curr->edge_block_entry[last_swap_offset].destination_vertex = 0;
                // curr->active_edge_count -= 1;
                // device_vertex_dictionary->active_edge_count[id] -= 1;
                --last_swap_offset;
//                printf("%lu\n", last_swap_offset);
            }
        }
    }

//    curr->lptr = NULL;
//    curr->rptr = NULL;
//    printf("here2 FOR VERTEX %lu\n", id);
    device_vertex_dictionary->last_insert_edge_block[id] = swapping_block;
    if(last_swap_offset >= 0){
        for(unsigned long i = 0; i < EDGE_BLOCK_SIZE - 1; ++i){
            unsigned long e = (curr->edge_block_entry[i]).destination_vertex;
            if(e > 0) ++last_swap_offset;
            else break;
        }
        if(last_swap_offset == EDGE_BLOCK_SIZE) last_swap_offset = 0;
    }
    device_vertex_dictionary->last_insert_edge_offset[id] = last_swap_offset;

    if(last_swap_offset < 0){
        push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1);
        push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
        d_e_queue.edge_block_address[push_index_for_edge_queue] = curr;
        device_vertex_dictionary->last_insert_edge_block[id] = curr->level_order_predecessor;
        device_vertex_dictionary->last_insert_edge_offset[id] = EDGE_BLOCK_SIZE - 1;
        device_vertex_dictionary->edge_block_count[id] -= 1;
//        printf("here3 FOR VERTEX %lu\n", id);
        if(curr == root){
//            printf("here4 FOR VERTEX %lu\n", id);
            curr->level_order_predecessor = NULL;
            device_vertex_dictionary->edge_block_address[id] = NULL;
            device_vertex_dictionary->active_edge_count[id] = 0;
            device_vertex_dictionary->last_insert_edge_offset[id] = 0;
            device_vertex_dictionary->edge_block_count[id] = 0;
            device_vertex_dictionary->last_insert_edge_block[id] = NULL;
        }
    }

//    printf("IN THE KERNEL FOR VERTEX %lu\n", id);
}

__device__ void preorderTraversalForFindingHoles(unsigned int vertex, struct edge_block *root, unsigned int *arr){
    if(!root) return;

    for(unsigned long j = 0 ; j < EDGE_BLOCK_SIZE ; j++) {
        if(root->edge_block_entry[j].destination_vertex == INFTY)
            ++arr[vertex];
    }

    preorderTraversalForFindingHoles(vertex, root->lptr, arr);
    preorderTraversalForFindingHoles(vertex, root->rptr, arr);
}

__global__ void findHolesPerVertex(unsigned int totalvertices, struct vertex_dictionary_structure *device_vertex_dictionary,
                                   unsigned int *arr){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= totalvertices) return;

    if((device_vertex_dictionary->edge_block_address[id] != NULL) && (device_vertex_dictionary->vertex_id[id] != 0)){
        preorderTraversalForFindingHoles(id, device_vertex_dictionary->edge_block_address[id], arr);
    }
}

__global__ void doPrefixSum(unsigned int* arr, unsigned int n, unsigned int *sum){
    for(unsigned int i = 0; i < n; ++i){
        *sum += arr[i];
    }

    printf("Prefix Sum: %u", *sum);
}

__global__ void helper(unsigned int *blocks_per_vertex_prefixSum, unsigned int* blocks_per_vertex, unsigned int n){
    blocks_per_vertex_prefixSum[n] = blocks_per_vertex_prefixSum[n - 1] + blocks_per_vertex[n - 1];
}

__global__ void printarr(unsigned int *blocks_per_vertex_prefixSum, unsigned int n){
    for(unsigned int i = 0; i < n + 1; ++i){
        printf("%u ", blocks_per_vertex_prefixSum[i]);
    }
    printf("\n");
}

// __global__ void compactionVertexCentricLevelOrderQueue(unsigned long totalvertices,
//                                                        struct vertex_dictionary_structure *device_vertex_dictionary,
//                                                        unsigned int *blocks_per_vertex_start,
//                                                        struct edge_block **level_order_queue){
//     unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

//     if(id >= totalvertices) return;

// //    if(id != 0) return;
// //    printf("start FOR VERTEX %lu\n", id);

//     struct edge_block *curr = device_vertex_dictionary->edge_block_address[id];
//     struct edge_block *root = device_vertex_dictionary->edge_block_address[id];
//     unsigned long bitString;
//     unsigned long parent_bit_string;
//     struct edge_block *parent = NULL;
//     unsigned int push_index_for_edge_queue;
//     unsigned int start_for_level_queue = blocks_per_vertex_start[id];
// //    printf("%ld\n", start_for_level_queue);

// //    printf("%p\n", curr);
// //    printf("%p\n", root);

//     if(!curr) return;

//     struct edge_block *swapping_block = device_vertex_dictionary->last_insert_edge_block[id];
// //    printf("%p\n", swapping_block);

//     unsigned long total_edge_blocks = device_vertex_dictionary->edge_block_count[id];
// //    printf("%lu\n", total_edge_blocks);

//     long curr_edge_block_index = 0;     // Keeps track of the current edge block
//     long last_edge_block_index = total_edge_blocks - 1;     // Keeps track of the swapping or last edge block
//     long last_swap_offset = EDGE_BLOCK_SIZE - 1;        // The last entry in the swapping block

//     // Finding the first non Infinity entry in the swapping edge block (last edge block)
//     while(last_swap_offset >= 0 && (swapping_block->edge_block_entry[last_swap_offset].destination_vertex == INFTY
//     || swapping_block->edge_block_entry[last_swap_offset].destination_vertex == 0)){
//         swapping_block->edge_block_entry[last_swap_offset].destination_vertex = 0;
//         --last_swap_offset;
//     }
// //    printf("For %lu last swap offset is: %ld\n", id, last_swap_offset);


// //    if(id == 0) printf("%ld\n", curr_edge_block_index);
// //    if(id == 0) printf("%ld\n", last_edge_block_index);
// //    if(id == 0) printf("%ld\n", last_swap_offset);

//     // Loop till our current edge block and last edge block are not same
//     while(curr_edge_block_index < last_edge_block_index){
// //        if(id == 2) printf("current edge block index for vertex %lu: %lu\n", id, curr_edge_block_index);
// //        if(id == 2) printf("last edge block index for vertex %lu: %lu\n", id, last_edge_block_index);

//         // Loop through each edge of the current edge block
//         for(unsigned long i = 0; i < EDGE_BLOCK_SIZE; ++i){
//             unsigned long e = (curr->edge_block_entry[i]).destination_vertex;

//             if(e != INFTY) continue;        // If it is a non-Infinity (valid) edge then continue, no need to swap

// //             printf("%lu for vertex %lu\n", e, id);

//             // deleted edge found
//             int edge_swapped_flag = 0;     // Flag to keep track whether the deleted edge is swapped with a non-Infinity edge from the last edge block or not

//             // Loop till the deleted edge in current edge block is not swapped with non Infinity edge from the last edge block
//             while(edge_swapped_flag == 0){
// //                 printf("here FOR VERTEX %lu\n", id);

//                 // If curr edge block and swapping edge block are same then break
//                 if(curr_edge_block_index == last_edge_block_index) break;

//                 // printf("here for vertex %lu\n", id);
// //                 if(id == 0) {
// //                     printf("%ld\n", last_swap_offset);
// //                     break;
// //                 }
// //                 printf("1here for %lu\n", id);
// //                 printf("%ld\n", last_swap_offset);

//                 /* If last swap offset is negative then it means the last edge block does not contain any non-Infinity value,
//                  * and we need to free the last edge block and update the swapping edge block */
//                 if(last_swap_offset == -1){
// //                  printf("here for %lu\n", id);

//                     // Get the index at which we need to push the freed edge block into the edge queue
//                     push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1);

//                     // if(d_e_queue.rear >= EDGE_PREALLOCATE_LIST_SIZE) printf("%ld\n", id);

//                     // wrap around
//                     push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
//                     d_e_queue.edge_block_address[push_index_for_edge_queue] = swapping_block;

//                     // Remove the child ptrs of the freed edge block
//                     swapping_block->lptr = NULL;
//                     swapping_block->rptr = NULL;

//                     // device_vertex_dictionary->active_edge_count[id] -= swapping_block->active_edge_count;
//                     // swapping_block->active_edge_count = 0;

//                    // Get the new swapping block
//                     swapping_block = swapping_block->level_order_predecessor;

// //                  printf("%p\n", swapping_block);

//                    // Remove the child ptrs of the new swapping block
//                    swapping_block->lptr = NULL;
//                    swapping_block->rptr = NULL;
//                     last_swap_offset = EDGE_BLOCK_SIZE - 1;


//                     /* freeing the parent to child relation for the old swapping block (freed edge block).
//                      * Note that last_edge_block_index is still not updated,
//                      * and it still points to the old swapping block (freed edge block). */
//                     if(last_edge_block_index & 1) parent_bit_string = bit_string_lookup[last_edge_block_index / 2];     // Get parent bit string of old swapping block
//                     else parent_bit_string = bit_string_lookup[last_edge_block_index / 2 - 1];                          // Get parent bit string of old swapping block

//                     parent = traverse_bit_string(root, parent_bit_string);   // Get parent ptr

//                     // Remove parent to child relation
//                     if(last_edge_block_index & 1) parent->lptr = NULL;
//                     else parent->rptr = NULL;

//                     // Update last edge block index, decrement it, pointing to new swapping block
//                     --last_edge_block_index;

//                     // Decrement edge block count for the vertex
//                     device_vertex_dictionary->edge_block_count[id] -= 1;

//                     if(curr_edge_block_index == last_edge_block_index) break;
//                 }
//                 // ELSE: Means that the last edge block might contain some non infinity edges
//                 else{
//                     // Loop till you find a non infinity edge in the swapping block
//                     while(last_swap_offset >= 0 && swapping_block->edge_block_entry[last_swap_offset].destination_vertex == INFTY) {
//                         // swapping_block->active_edge_count -= 1;
//                         // device_vertex_dictionary->active_edge_count[id] -= 1;

//                         swapping_block->edge_block_entry[last_swap_offset].destination_vertex = 0;
//                         --last_swap_offset;
//                     }

// //                     if(id == 2) printf("%ld\n", last_swap_offset);

//                     /* If last swap offset is negative then it means the last edge block does not contain any non-Infinity value,
//                      * and we need to free the last edge block and update the swapping edge block */
//                     if(last_swap_offset < 0){
//                         // Get the index at which we need to push the freed edge block into the edge queue
//                         push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1);

//                         // wrap around
//                         push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
//                         d_e_queue.edge_block_address[push_index_for_edge_queue] = swapping_block;

//                         // Remove the child ptrs of the freed edge block
//                        swapping_block->lptr = NULL;
//                        swapping_block->rptr = NULL;

//                         // device_vertex_dictionary->active_edge_count[id] -= swapping_block->active_edge_count;
//                         // swapping_block->active_edge_count = 0;

//                         // Get the new swapping block
//                         swapping_block = swapping_block->level_order_predecessor;

//                         // Update the last insert edge block for the current vertex
//                         device_vertex_dictionary->last_insert_edge_block[id] = swapping_block;

// //                         printf("%p\n", swapping_block);

//                         // Remove the child ptrs of the new swapping block
//                        swapping_block->lptr = NULL;
//                        swapping_block->rptr = NULL;
//                         last_swap_offset = EDGE_BLOCK_SIZE - 1;


//                         /* freeing the parent to child relation for the old swapping block (freed edge block).
//                          * Note that last_edge_block_index is still not updated,
//                          * and it still points to the old swapping block (freed edge block). */
//                         if(last_edge_block_index & 1) parent_bit_string = bit_string_lookup[last_edge_block_index / 2]; // Get parent bit string of old swapping block
//                         else parent_bit_string = bit_string_lookup[last_edge_block_index / 2 - 1];                      // Get parent bit string of old swapping block

//                         parent = traverse_bit_string(root, parent_bit_string);      // Get parent ptr

//                         // Remove parent to child relation
//                         if(last_edge_block_index & 1) parent->lptr = NULL;
//                         else parent->rptr = NULL;

//                         // Update last edge block index, decrement it, pointing to new swapping block
//                         --last_edge_block_index;

//                         // Decrement edge block count for the vertex
//                         device_vertex_dictionary->edge_block_count[id] -= 1;
//                     }
//                     // ELSE: Means we've found a non infinity edge in the swapping block
//                     else{
//                         if(curr_edge_block_index == last_edge_block_index) break;

//                         // Swap the deleted edge of current edge block with the non infinity edge of swapping block
//                         curr->edge_block_entry[i].destination_vertex = swapping_block->edge_block_entry[last_swap_offset].destination_vertex;
//                         swapping_block->edge_block_entry[last_swap_offset].destination_vertex = 0;
//                         --last_swap_offset;

//                         // swapping_block->active_edge_count -= 1;
//                         // device_vertex_dictionary->active_edge_count[id] -= 1;

//                         // Update edge swapped flag
//                         edge_swapped_flag = 1;

//                         /* If last swap offset is negative then it means the last edge block does not contain any non-Infinity value,
//                          * and we need to free the last edge block and update the swapping edge block */
//                         if(last_swap_offset == -1){
//                             // Get the index at which we need to push the freed edge block into the edge queue
//                             push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1);

//                             // wrap around
//                             push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
//                             d_e_queue.edge_block_address[push_index_for_edge_queue] = swapping_block;

//                             // Remove the child ptrs of the freed edge block
//                             swapping_block->lptr = NULL;
//                             swapping_block->rptr = NULL;

//                             // device_vertex_dictionary->active_edge_count[id] -= swapping_block->active_edge_count;
//                             // swapping_block->active_edge_count = 0;

//                             // Get the new swapping block
//                             swapping_block = swapping_block->level_order_predecessor;

//                             // Update the last insert edge block for the current vertex
//                             device_vertex_dictionary->last_insert_edge_block[id] = swapping_block;
// //                         printf("%p\n", swapping_block);

//                             // Remove the child ptrs of the new swapping block
//                            swapping_block->lptr = NULL;
//                            swapping_block->rptr = NULL;
//                             last_swap_offset = EDGE_BLOCK_SIZE - 1;


//                             /* freeing the parent to child relation for the old swapping block (freed edge block).
//                             Note that last_edge_block_index is still not updated, and it still points to the old swapping block (freed edge block). */
//                             if(last_edge_block_index & 1) parent_bit_string = bit_string_lookup[last_edge_block_index / 2];    // Get parent bit string of old swapping block
//                             else parent_bit_string = bit_string_lookup[last_edge_block_index / 2 - 1];                          // Get parent bit string of old swapping block

//                             parent = traverse_bit_string(root, parent_bit_string);      // Get parent ptr

//                             // Remove parent to child relation
//                             if(last_edge_block_index & 1) parent->lptr = NULL;
//                             else parent->rptr = NULL;

//                             // Update last edge block index, decrement it, pointing to new swapping block
//                             --last_edge_block_index;

//                             // Decrement edge block count for the vertex
//                             device_vertex_dictionary->edge_block_count[id] -= 1;
//                         }
//                     }
//                 }
//             }

//             if(curr_edge_block_index == last_edge_block_index) break;
//         }

//         if(curr_edge_block_index == last_edge_block_index) break;

//         /* Get the indices to push the left and right child edge blocks
//          * to the level order queue for level order traversal */
//         unsigned int idx1 = start_for_level_queue + 2 * ((unsigned int) curr_edge_block_index);
//         unsigned int idx2 = start_for_level_queue + 2 * ((unsigned int) curr_edge_block_index) + 1;

// //        printf("idx1: %u idx2: %u\n", idx1, idx2);

//         // Push the child blocks to the level order queue
//         if(curr->lptr != NULL) level_order_queue[idx1] = curr->lptr;
//         if(curr->rptr != NULL) level_order_queue[idx2] = curr->rptr;

// //        ++curr_edge_block_index;
// //        bitString = bit_string_lookup[curr_edge_block_index];
// //        curr = traverse_bit_string(root, bitString);

//         // Get the index for the next edge block in level order traversal
//         unsigned int idx3 = start_for_level_queue + curr_edge_block_index;

// //        printf("idx3: %u\n", idx3);

//         // Get the pointer to the next edge block. Update it to current.
//         curr = level_order_queue[idx3];

//         // Update the curr edge block index
//         ++curr_edge_block_index;
//     }

// //    printf("Reached out of the while loop\n");

//     /* If the current edge block and swapping edge block are same,
//      * then move the non Infinity edges to beginning and the deleted (inifinity marked) edges
//      * to the end of the edge block. */
//     if(curr_edge_block_index == last_edge_block_index){
// //        printf("%lu\n", curr_edge_block_index);
// //        printf("here for vertex %lu\n", id);

//         last_swap_offset = EDGE_BLOCK_SIZE - 1;

// //        printf("%lu %lu\n", curr_edge_block_index, last_swap_offset);

//         // Find the first non infinity edge from the end of the edge block
//         while(last_swap_offset >= 0 && (curr->edge_block_entry[last_swap_offset].destination_vertex == INFTY
//         || curr->edge_block_entry[last_swap_offset].destination_vertex == 0)) {
//             if(curr->edge_block_entry[last_swap_offset].destination_vertex == INFTY) {
//                 // device_vertex_dictionary->active_edge_count[id] -= 1;
//                 // curr->active_edge_count -= 1;

//                 curr->edge_block_entry[last_swap_offset].destination_vertex = 0;
//             }
//             --last_swap_offset;
//         }

//         long start = 0;

// //        printf("%lu for vertex %lu\n", last_swap_offset, id);

//         // Do the swapping procedure
//         while(start < EDGE_BLOCK_SIZE && start < last_swap_offset && last_swap_offset >= 0 &&
//         last_swap_offset < EDGE_BLOCK_SIZE){
// //            printf("%lu, %lu for vertex %lu\n", start, last_swap_offset, id);

//             if(curr->edge_block_entry[start].destination_vertex != INFTY) ++start;      // Non infinity edge, increment start
//             else if(curr->edge_block_entry[last_swap_offset].destination_vertex == INFTY) {     // Infinity edge from last, decrement last ptr
//                 curr->edge_block_entry[last_swap_offset].destination_vertex = 0;

//                 // device_vertex_dictionary->active_edge_count[id] -= 1;

//                 --last_swap_offset;

//                 // --curr->active_edge_count;
//             }
//             else{ // Got infinity edge from start and non infinity edge from end, Swap both, update ptrs
//                 curr->edge_block_entry[start].destination_vertex = curr->edge_block_entry[last_swap_offset].destination_vertex;
//                 curr->edge_block_entry[last_swap_offset].destination_vertex = 0;

//                 // curr->active_edge_count -= 1;
//                 // device_vertex_dictionary->active_edge_count[id] -= 1;

//                 --last_swap_offset;

// //                printf("%lu\n", last_swap_offset);
//             }
//         }
//     }

// //    curr->lptr = NULL;
// //    curr->rptr = NULL;
// //    printf("here2 FOR VERTEX %lu\n", id);

//     // Update the last insert edge block for the current vertex
//     device_vertex_dictionary->last_insert_edge_block[id] = swapping_block;

//     // Check whether last insert edge block of the adjacency is full and update the last swap offset.
//     if(last_swap_offset >= 0){
//         for(unsigned long i = 0; i < EDGE_BLOCK_SIZE - 1; ++i){
//             unsigned long e = (curr->edge_block_entry[i]).destination_vertex;
//             if(e > 0) ++last_swap_offset;
//             else break;
//         }
//         if(last_swap_offset == EDGE_BLOCK_SIZE) last_swap_offset = 0;
//     }

//     device_vertex_dictionary->last_insert_edge_offset[id] = last_swap_offset;

//     /* If last swap offset is negative then it means the last edge block does not contain any non-Infinity value,
//      * and we need to free the last edge block and update the swapping edge block */
//     if(last_swap_offset < 0){
//         // Get the index at which we need to push the freed edge block into the edge queue
//         push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1);

//         // wrap around
//         push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
//         d_e_queue.edge_block_address[push_index_for_edge_queue] = curr;

//         // Update last insert edge block for the adjacency
//         device_vertex_dictionary->last_insert_edge_block[id] = curr->level_order_predecessor;

//         // Update the last insert edge offset for the adjacency
//         device_vertex_dictionary->last_insert_edge_offset[id] = EDGE_BLOCK_SIZE - 1;

//         // Decrement the edge block count for the vertex
//         device_vertex_dictionary->edge_block_count[id] -= 1;

// //        printf("here3 FOR VERTEX %lu\n", id);

//         // If the freed edge block is root, the free the adjacency of the vertex
//         if(curr == root){
// //            printf("here4 FOR VERTEX %lu\n", id);

//             curr->level_order_predecessor = NULL;
//             device_vertex_dictionary->edge_block_address[id] = NULL;
//             device_vertex_dictionary->active_edge_count[id] = 0;
//             device_vertex_dictionary->last_insert_edge_offset[id] = 0;
//             device_vertex_dictionary->edge_block_count[id] = 0;
//             device_vertex_dictionary->last_insert_edge_block[id] = NULL;
//         }
//     }

// //    printf("IN THE KERNEL FOR VERTEX %lu\n", id);
// }


__global__ void compactionVertexCentricLevelOrderQueueOptimized(unsigned long totalvertices,
                                                                struct vertex_dictionary_structure *device_vertex_dictionary,
                                                                unsigned int *blocks_per_vertex_start,
                                                                struct edge_block **level_order_queue){
    unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

    if(id >= totalvertices) return;
    // if(id != 0) return;

//    if(id != 0) return;
//    printf("start FOR VERTEX %lu\n", id);

    struct edge_block *curr = device_vertex_dictionary->edge_block_address[id];
    struct edge_block *root = device_vertex_dictionary->edge_block_address[id];
    unsigned long bitString;
    unsigned long parent_bit_string;
    struct edge_block *parent = NULL;
    unsigned int push_index_for_edge_queue;
    unsigned int start_for_level_queue = blocks_per_vertex_start[id];
//    printf("%ld\n", start_for_level_queue);

//    printf("%p\n", curr);
//    printf("%p\n", root);

    if(!curr) return;

    struct edge_block *swapping_block = device_vertex_dictionary->last_insert_edge_block[id];
//    printf("%p\n", swapping_block);

    unsigned long total_edge_blocks = device_vertex_dictionary->edge_block_count[id];
//    printf("%lu\n", total_edge_blocks);

    /* FIRST LEVEL ORDER TRAVERSAL TO TAKE THE EDGE BLOCKS */

    unsigned int i = 0;
    unsigned int idx1;
    unsigned int idx2;
    unsigned int idx3;
    while(i < total_edge_blocks){
        /* Get the indices to push the left and right child edge blocks
         * to the level order queue for level order traversal */
        level_order_queue[start_for_level_queue + i] = curr;

        idx1 = start_for_level_queue + 2 * ((unsigned int) i) + 1;
        idx2 = start_for_level_queue + 2 * ((unsigned int) i) + 2;

        // Push the child blocks to the level order queue
        if(curr->lptr != NULL) level_order_queue[idx1] = curr->lptr;
        if(curr->rptr != NULL) level_order_queue[idx2] = curr->rptr;

        ++i;
        idx3 = start_for_level_queue + i;
        curr = level_order_queue[idx3];
    }
    // printf("Here1\n");

    /* SECOND LEVEL ORDER TRAVERSAL TO DO THE COMPACTION */

    long curr_edge_block_index = 0;     // Keeps track of the current edge block
    long last_edge_block_index = total_edge_blocks - 1;     // Keeps track of the swapping or last edge block
    long last_swap_offset = EDGE_BLOCK_SIZE - 1;        // The last entry in the swapping block

    // Finding the first non Infinity entry in the swapping edge block (last edge block)
    while(last_swap_offset >= 0 && (swapping_block->edge_block_entry[last_swap_offset].destination_vertex == INFTY
                                    || swapping_block->edge_block_entry[last_swap_offset].destination_vertex == 0)){
        swapping_block->edge_block_entry[last_swap_offset].destination_vertex = 0;
        --last_swap_offset;
    }
//    printf("For %lu last swap offset is: %ld\n", id, last_swap_offset);


//    if(id == 0) printf("%ld\n", curr_edge_block_index);
//    if(id == 0) printf("%ld\n", last_edge_block_index);
//    if(id == 0) printf("%ld\n", last_swap_offset);

    // Loop till our current edge block and last edge block are not same
    curr = level_order_queue[start_for_level_queue];
    while(curr_edge_block_index < last_edge_block_index){
//        if(id == 2) printf("current edge block index for vertex %lu: %lu\n", id, curr_edge_block_index);
//        if(id == 2) printf("last edge block index for vertex %lu: %lu\n", id, last_edge_block_index);

        // Loop through each edge of the current edge block
        for(unsigned long i = 0; i < EDGE_BLOCK_SIZE; ++i){
            unsigned long e = (curr->edge_block_entry[i]).destination_vertex;

            if(e != INFTY) continue;        // If it is a non-Infinity (valid) edge then continue, no need to swap

//             printf("%lu for vertex %lu\n", e, id);

            // deleted edge found
            int edge_swapped_flag = 0;     // Flag to keep track whether the deleted edge is swapped with a non-Infinity edge from the last edge block or not

            // Loop till the deleted edge in current edge block is not swapped with non Infinity edge from the last edge block
            while(edge_swapped_flag == 0){
//                 printf("here FOR VERTEX %lu\n", id);

                // If curr edge block and swapping edge block are same then break
                if(curr_edge_block_index == last_edge_block_index) break;

                // printf("here for vertex %lu\n", id);
//                 if(id == 0) {
//                     printf("%ld\n", last_swap_offset);
//                     break;
//                 }
//                 printf("1here for %lu\n", id);
//                 printf("%ld\n", last_swap_offset);

                /* If last swap offset is negative then it means the last edge block does not contain any non-Infinity value,
                 * and we need to free the last edge block and update the swapping edge block */
                if(last_swap_offset == -1){
//                  printf("here for %lu\n", id);

                    // Get the index at which we need to push the freed edge block into the edge queue
                    push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1);

                    // if(d_e_queue.rear >= EDGE_PREALLOCATE_LIST_SIZE) printf("%ld\n", id);

                    // wrap around
                    push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
                    d_e_queue.edge_block_address[push_index_for_edge_queue] = swapping_block;

                    // Remove the child ptrs of the freed edge block
                    swapping_block->lptr = NULL;
                    swapping_block->rptr = NULL;

                    // device_vertex_dictionary->active_edge_count[id] -= swapping_block->active_edge_count;
                    // swapping_block->active_edge_count = 0;

                    // Get the new swapping block
                    swapping_block = swapping_block->level_order_predecessor;

//                  printf("%p\n", swapping_block);

                    // Remove the child ptrs of the new swapping block
                    swapping_block->lptr = NULL;
                    swapping_block->rptr = NULL;
                    last_swap_offset = EDGE_BLOCK_SIZE - 1;


                    /* freeing the parent to child relation for the old swapping block (freed edge block).
                     * Note that last_edge_block_index is still not updated,
                     * and it still points to the old swapping block (freed edge block). */
                    if(last_edge_block_index & 1) {
                        parent = level_order_queue[start_for_level_queue + last_edge_block_index / 2];
                        parent->lptr = NULL;
                    }
                    else {
                        parent = level_order_queue[start_for_level_queue + last_edge_block_index / 2 - 1];
                        parent->rptr = NULL;
                    }

                    // Update last edge block index, decrement it, pointing to new swapping block
                    --last_edge_block_index;

                    // Decrement edge block count for the vertex
                    device_vertex_dictionary->edge_block_count[id] -= 1;

                    if(curr_edge_block_index == last_edge_block_index) break;
                }
                    // ELSE: Means that the last edge block might contain some non infinity edges
                else{
                    // Loop till you find a non infinity edge in the swapping block
                    while(last_swap_offset >= 0 && swapping_block->edge_block_entry[last_swap_offset].destination_vertex == INFTY) {
                        // swapping_block->active_edge_count -= 1;
                        // device_vertex_dictionary->active_edge_count[id] -= 1;

                        swapping_block->edge_block_entry[last_swap_offset].destination_vertex = 0;
                        --last_swap_offset;
                    }

//                     if(id == 2) printf("%ld\n", last_swap_offset);

                    /* If last swap offset is negative then it means the last edge block does not contain any non-Infinity value,
                     * and we need to free the last edge block and update the swapping edge block */
                    if(last_swap_offset < 0){
                        // Get the index at which we need to push the freed edge block into the edge queue
                        push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1);

                        // wrap around
                        push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
                        d_e_queue.edge_block_address[push_index_for_edge_queue] = swapping_block;

                        // Remove the child ptrs of the freed edge block
                        swapping_block->lptr = NULL;
                        swapping_block->rptr = NULL;

                        // device_vertex_dictionary->active_edge_count[id] -= swapping_block->active_edge_count;
                        // swapping_block->active_edge_count = 0;

                        // Get the new swapping block
                        swapping_block = swapping_block->level_order_predecessor;

                        // Update the last insert edge block for the current vertex
                        device_vertex_dictionary->last_insert_edge_block[id] = swapping_block;

//                         printf("%p\n", swapping_block);

                        // Remove the child ptrs of the new swapping block
                        swapping_block->lptr = NULL;
                        swapping_block->rptr = NULL;
                        last_swap_offset = EDGE_BLOCK_SIZE - 1;


                        /* freeing the parent to child relation for the old swapping block (freed edge block).
                         * Note that last_edge_block_index is still not updated,
                         * and it still points to the old swapping block (freed edge block). */
                        if(last_edge_block_index & 1) {
                            parent = level_order_queue[start_for_level_queue + last_edge_block_index / 2];
                            parent->lptr = NULL;
                        }
                        else {
                            parent = level_order_queue[start_for_level_queue + last_edge_block_index / 2 - 1];
                            parent->rptr = NULL;
                        }

                        // Update last edge block index, decrement it, pointing to new swapping block
                        --last_edge_block_index;

                        // Decrement edge block count for the vertex
                        device_vertex_dictionary->edge_block_count[id] -= 1;
                    }
                        // ELSE: Means we've found a non infinity edge in the swapping block
                    else{
                        if(curr_edge_block_index == last_edge_block_index) break;

                        // Swap the deleted edge of current edge block with the non infinity edge of swapping block
                        curr->edge_block_entry[i].destination_vertex = swapping_block->edge_block_entry[last_swap_offset].destination_vertex;
                        swapping_block->edge_block_entry[last_swap_offset].destination_vertex = 0;
                        --last_swap_offset;

                        // swapping_block->active_edge_count -= 1;
                        // device_vertex_dictionary->active_edge_count[id] -= 1;

                        // Update edge swapped flag
                        edge_swapped_flag = 1;

                        /* If last swap offset is negative then it means the last edge block does not contain any non-Infinity value,
                         * and we need to free the last edge block and update the swapping edge block */
                        if(last_swap_offset == -1){
                            // Get the index at which we need to push the freed edge block into the edge queue
                            push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1);

                            // wrap around
                            push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
                            d_e_queue.edge_block_address[push_index_for_edge_queue] = swapping_block;

                            // Remove the child ptrs of the freed edge block
                            swapping_block->lptr = NULL;
                            swapping_block->rptr = NULL;

                            // device_vertex_dictionary->active_edge_count[id] -= swapping_block->active_edge_count;
                            // swapping_block->active_edge_count = 0;

                            // Get the new swapping block
                            swapping_block = swapping_block->level_order_predecessor;

                            // Update the last insert edge block for the current vertex
                            device_vertex_dictionary->last_insert_edge_block[id] = swapping_block;
//                         printf("%p\n", swapping_block);

                            // Remove the child ptrs of the new swapping block
                            swapping_block->lptr = NULL;
                            swapping_block->rptr = NULL;
                            last_swap_offset = EDGE_BLOCK_SIZE - 1;


                            /* freeing the parent to child relation for the old swapping block (freed edge block).
                            Note that last_edge_block_index is still not updated, and it still points to the old swapping block (freed edge block). */
                            if(last_edge_block_index & 1) {
                                parent = level_order_queue[start_for_level_queue + last_edge_block_index / 2];
                                parent->lptr = NULL;
                            }
                            else {
                                parent = level_order_queue[start_for_level_queue + last_edge_block_index / 2 - 1];
                                parent->rptr = NULL;
                            }

                            // Update last edge block index, decrement it, pointing to new swapping block
                            --last_edge_block_index;

                            // Decrement edge block count for the vertex
                            device_vertex_dictionary->edge_block_count[id] -= 1;
                        }
                    }
                }
            }

            if(curr_edge_block_index == last_edge_block_index) break;
        }

        if(curr_edge_block_index == last_edge_block_index) break;

//         /* Get the indices to push the left and right child edge blocks
//          * to the level order queue for level order traversal */
//         unsigned int idx1 = start_for_level_queue + 2 * ((unsigned int) curr_edge_block_index);
//         unsigned int idx2 = start_for_level_queue + 2 * ((unsigned int) curr_edge_block_index) + 1;

// //        printf("idx1: %u idx2: %u\n", idx1, idx2);

//         // Push the child blocks to the level order queue
//         if(curr->lptr != NULL) level_order_queue[idx1] = curr->lptr;
//         if(curr->rptr != NULL) level_order_queue[idx2] = curr->rptr;

// //        ++curr_edge_block_index;
// //        bitString = bit_string_lookup[curr_edge_block_index];
// //        curr = traverse_bit_string(root, bitString);

        // Get the index for the next edge block in level order traversal
        ++curr_edge_block_index;
        //  if(curr_edge_block_index >= last_edge_block_index) break;
        idx3 = start_for_level_queue + curr_edge_block_index;
        // printf("%u, %u\n", id, idx3);

//        printf("idx3: %u\n", idx3);

        // Get the pointer to the next edge block. Update it to current.
        curr = level_order_queue[idx3];

        // Update the curr edge block index
    }

//    printf("%u\n", id);
//    printf("Reached out of the while loop\n");

    /* If the current edge block and swapping edge block are same,
     * then move the non Infinity edges to beginning and the deleted (inifinity marked) edges
     * to the end of the edge block. */
    if(curr_edge_block_index == last_edge_block_index){
//        printf("%lu\n", curr_edge_block_index);
//        printf("here for vertex %lu\n", id);

        last_swap_offset = EDGE_BLOCK_SIZE - 1;

//        printf("%lu %lu\n", curr_edge_block_index, last_swap_offset);

        // Find the first non infinity edge from the end of the edge block
        while(last_swap_offset >= 0 && (curr->edge_block_entry[last_swap_offset].destination_vertex == INFTY
                                        || curr->edge_block_entry[last_swap_offset].destination_vertex == 0)) {
            if(curr->edge_block_entry[last_swap_offset].destination_vertex == INFTY) {
                // device_vertex_dictionary->active_edge_count[id] -= 1;
                // curr->active_edge_count -= 1;

                curr->edge_block_entry[last_swap_offset].destination_vertex = 0;
            }
            --last_swap_offset;
        }

        long start = 0;

//        printf("%lu for vertex %lu\n", last_swap_offset, id);

        // Do the swapping procedure
        while(start < EDGE_BLOCK_SIZE && start < last_swap_offset && last_swap_offset >= 0 &&
              last_swap_offset < EDGE_BLOCK_SIZE){
//            printf("%lu, %lu for vertex %lu\n", start, last_swap_offset, id);

            if(curr->edge_block_entry[start].destination_vertex != INFTY) ++start;      // Non infinity edge, increment start
            else if(curr->edge_block_entry[last_swap_offset].destination_vertex == INFTY) {     // Infinity edge from last, decrement last ptr
                curr->edge_block_entry[last_swap_offset].destination_vertex = 0;

                // device_vertex_dictionary->active_edge_count[id] -= 1;

                --last_swap_offset;

                // --curr->active_edge_count;
            }
            else{ // Got infinity edge from start and non infinity edge from end, Swap both, update ptrs
                curr->edge_block_entry[start].destination_vertex = curr->edge_block_entry[last_swap_offset].destination_vertex;
                curr->edge_block_entry[last_swap_offset].destination_vertex = 0;

                // curr->active_edge_count -= 1;
                // device_vertex_dictionary->active_edge_count[id] -= 1;

                --last_swap_offset;

//                printf("%lu\n", last_swap_offset);
            }
        }
    }

//    curr->lptr = NULL;
//    curr->rptr = NULL;
//    printf("here2 FOR VERTEX %lu\n", id);

    // Update the last insert edge block for the current vertex
    device_vertex_dictionary->last_insert_edge_block[id] = swapping_block;

    // Check whether last insert edge block of the adjacency is full and update the last swap offset.
    if(last_swap_offset >= 0){
        for(unsigned long i = 0; i < EDGE_BLOCK_SIZE - 1; ++i){
            unsigned long e = (curr->edge_block_entry[i]).destination_vertex;
            if(e > 0) ++last_swap_offset;
            else break;
        }
        if(last_swap_offset == EDGE_BLOCK_SIZE) last_swap_offset = 0;
    }

    device_vertex_dictionary->last_insert_edge_offset[id] = last_swap_offset;

    /* If last swap offset is negative then it means the last edge block does not contain any non-Infinity value,
     * and we need to free the last edge block and update the swapping edge block */
    if(last_swap_offset < 0){
        // Get the index at which we need to push the freed edge block into the edge queue
        push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1);

        // wrap around
        push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
        d_e_queue.edge_block_address[push_index_for_edge_queue] = curr;

        // Update last insert edge block for the adjacency
        device_vertex_dictionary->last_insert_edge_block[id] = curr->level_order_predecessor;

        // Update the last insert edge offset for the adjacency
        device_vertex_dictionary->last_insert_edge_offset[id] = EDGE_BLOCK_SIZE - 1;

        // Decrement the edge block count for the vertex
        device_vertex_dictionary->edge_block_count[id] -= 1;

//        printf("here3 FOR VERTEX %lu\n", id);

        // If the freed edge block is root, the free the adjacency of the vertex
        if(curr == root){
//            printf("here4 FOR VERTEX %lu\n", id);

            curr->level_order_predecessor = NULL;
            device_vertex_dictionary->edge_block_address[id] = NULL;
            device_vertex_dictionary->active_edge_count[id] = 0;
            device_vertex_dictionary->last_insert_edge_offset[id] = 0;
            device_vertex_dictionary->edge_block_count[id] = 0;
            device_vertex_dictionary->last_insert_edge_block[id] = NULL;
        }
    }

//    printf("IN THE KERNEL FOR VERTEX %lu\n", id);
}


void compactionFunction(struct vertex_dictionary_structure *device_vertex_dictionary, struct graph_properties *h_graph_prop, double fullTime)
{
    std::cout << std::endl << "PREPROCESSING BEFORE COMPACTION FOR LEVEL ORDER TRAVERSAL STARTING!" << std::endl;
    // Find total edge blocks per vertex
    clock_t level_order_queue_start;
    clock_t level_order_queue_end;
    float total_level_order_queue_time = 0.0;
    float compactTime = 0.0;
    float totalTime = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned int *blocks_per_vertex;
    unsigned int vertex_size = h_graph_prop->xDim;
    cudaMalloc(&blocks_per_vertex, ((unsigned int) vertex_size) * sizeof (unsigned int));

    unsigned int thread_blocks = ceil((double) vertex_size / THREADS_PER_BLOCK);
    level_order_queue_start = clock();
    findBlocksPerVertex<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, device_vertex_dictionary, blocks_per_vertex);
    cudaDeviceSynchronize();
    level_order_queue_end = clock();

    total_level_order_queue_time += ((double) (level_order_queue_end - level_order_queue_start)) / CLOCKS_PER_SEC * 1000;

    // Do prefix sum
    unsigned int *blocks_per_vertex_prefixSum;
    cudaMalloc(&blocks_per_vertex_prefixSum, ((unsigned int) vertex_size + 1) * sizeof(unsigned int));

    thrust::device_ptr<unsigned int> blocks_per_vertex_ptr = thrust::device_pointer_cast(blocks_per_vertex);
    thrust::device_ptr<unsigned int> blocks_per_vertex_prefixSum_ptr = thrust::device_pointer_cast(blocks_per_vertex_prefixSum);
    cudaDeviceSynchronize();
    level_order_queue_start = clock();
    thrust::exclusive_scan(thrust::device, blocks_per_vertex_ptr, blocks_per_vertex_ptr + ((unsigned int) vertex_size), blocks_per_vertex_prefixSum_ptr);
    cudaDeviceSynchronize();
    level_order_queue_end = clock();

    total_level_order_queue_time += ((double) (level_order_queue_end - level_order_queue_start)) / CLOCKS_PER_SEC * 1000;
    unsigned int *total_edge_blocks;
    cudaMalloc(&total_edge_blocks, sizeof(unsigned int));
    cudaDeviceSynchronize();

    findTotalEdgeBlocks<<<1,1>>>((unsigned int)vertex_size, total_edge_blocks, blocks_per_vertex_prefixSum, blocks_per_vertex);
    cudaDeviceSynchronize();

    // helper<<<1,1>>>(blocks_per_vertex_prefixSum, blocks_per_vertex, (unsigned int)vertex_size);
    // cudaDeviceSynchronize();

    unsigned int *host_total_edge_blocks;
    host_total_edge_blocks = (unsigned int *)malloc(sizeof(unsigned int));

    cudaMemcpy(host_total_edge_blocks, total_edge_blocks, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
//        std::cout << std::endl << *host_total_edge_blocks << std::endl;

    struct edge_block **level_order_queue;
    cudaMalloc(&level_order_queue, (*host_total_edge_blocks + 3) * ((unsigned int) sizeof (struct edge_block *)));
    cudaDeviceSynchronize();

    unsigned int* holesPerVertexBefore;
    cudaMalloc(&holesPerVertexBefore, (unsigned int)vertex_size * sizeof(unsigned int));
    thread_blocks = ceil((double)(vertex_size) / THREADS_PER_BLOCK);
    findHolesPerVertex<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, device_vertex_dictionary, holesPerVertexBefore);
    cudaDeviceSynchronize();

    unsigned int* total_holes_before_compaction;
    cudaMalloc(&total_holes_before_compaction, sizeof(unsigned int));
    doPrefixSum<<<1,1>>>(holesPerVertexBefore, vertex_size, total_holes_before_compaction);
    cudaDeviceSynchronize();

    std::cout << std::endl << "PREPROCESSING BEFORE COMPACTION FOR LEVEL ORDER TRAVERSAL DONE!" << std::endl;
    CUDA_CHECK_ERROR();

    // printarr<<<1,1>>>(blocks_per_vertex_prefixSum, (unsigned int)vertex_size);
    cudaDeviceSynchronize();
    std::cout << std::endl << "STARTING COMPACTION!" << std::endl;
    thread_blocks = ceil((double)(vertex_size) / THREADS_PER_BLOCK);
//        std::cout << thread_blocks << std::endl;

    unsigned int *num_blocks_before;
    double *totalMemoryBeforeCompaction;
    cudaMalloc(&num_blocks_before, sizeof(unsigned int));
    cudaMalloc(&totalMemoryBeforeCompaction, sizeof(double));
    recoveredMemoryKernel1<<<1,1>>>(num_blocks_before, totalMemoryBeforeCompaction, total_edge_blocks, device_vertex_dictionary);
    cudaDeviceSynchronize();

    compactTime = 0.0;
    cudaEventRecord(start);
    //    compactionVertexCentric<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, device_vertex_dictionary);
//        compactionVertexCentricParallel<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, device_vertex_dictionary, push_area_prefixSum);
//        compactionVertexCentricPostOrder<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, device_vertex_dictionary);
    compactionVertexCentricLevelOrderQueue<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, device_vertex_dictionary, blocks_per_vertex_prefixSum, level_order_queue);
    // compactionVertexCentricLevelOrderQueueOptimized<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, device_vertex_dictionary, blocks_per_vertex_prefixSum, level_order_queue);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    cudaEventElapsedTime(&compactTime, start, stop);
    totalTime += compactTime;
    fullTime += totalTime;
    CUDA_CHECK_ERROR();
    std::cout << std::endl << "COMPACTION DONE!" << std::endl;
//        printKernelmodded_v2<<< 1, 1>>>(device_vertex_dictionary, vertex_size);
    cudaDeviceSynchronize();
    std::cout << std::endl << "COMPACTION DONE!" << std::endl;

    std::cout << std::endl << "Total Time: " << fullTime << " ms" << std::endl;
    std::cout << "Compaction Kernel Time: " << compactTime << " ms" << std::endl;
//        std::cout << "Push Area Computation Time: " << total_push_area_time << " ms" << std::endl;
//        std::cout << "Total compaction Time: " << total_push_area_time + compactTime << " ms" << std::endl;
    std::cout << "Level Order Queue Preprocessing Time: " << total_level_order_queue_time << " ms" << std::endl;
    std::cout << "Total compaction Time: " << total_level_order_queue_time + compactTime << " ms" << std::endl;
    std::cout << "Compaction Overhead: " << (total_level_order_queue_time + compactTime) / fullTime << std::endl;

    unsigned int* holesPerVertexAfter;
    cudaMalloc(&holesPerVertexAfter, (unsigned int)vertex_size * sizeof(unsigned int));
    thread_blocks = ceil((double)(vertex_size) / THREADS_PER_BLOCK);
    findHolesPerVertex<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, device_vertex_dictionary, holesPerVertexAfter);
    cudaDeviceSynchronize();

    unsigned int* total_holes_after_compaction;
    cudaMalloc(&total_holes_after_compaction, sizeof(unsigned int));
    doPrefixSum<<<1,1>>>(holesPerVertexAfter, vertex_size, total_holes_after_compaction);
    cudaDeviceSynchronize();

    unsigned int *num_blocks_after;
    cudaMalloc(&num_blocks_after, sizeof(unsigned int));
    recoveredMemoryPercentage<<<1,1>>>(num_blocks_before, num_blocks_after, total_edge_blocks, device_vertex_dictionary, vertex_size, totalMemoryBeforeCompaction, total_holes_before_compaction, total_holes_after_compaction);
    cudaDeviceSynchronize();
}

__global__ void printDist(unsigned int vertices, unsigned int *dist){
    unsigned int limit;
    if(vertices >= 40) limit = 40;
    else limit = vertices;
    for(unsigned int u = 0; u < limit; ++u) printf("%u ", dist[u]);
    printf("\n");
}

__global__ void countPerVertexEdge(unsigned int vertices, unsigned int *edge_count_per_vertex,
                                   struct vertex_dictionary_structure *device_vertex_dictionary){
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= vertices) return;

    edge_count_per_vertex[id] = device_vertex_dictionary->active_edge_count[id];
}

__global__ void triangleCountingEdgeCentricNaive(unsigned int total_edges, unsigned int total_vertices, unsigned int *edge_count_per_vertex_prefixSum, unsigned int* tc_per_vertex,
                                                 struct vertex_dictionary_structure *device_vertex_dictionary){
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= total_edges) return;

    // 1. Find source vertex for the edge using binary search
    unsigned int start = 0;
    unsigned int end = total_vertices;
    unsigned int src = -1;

    while(start <= end){
        unsigned int mid = start + (end - start) / 2;

        if(edge_count_per_vertex_prefixSum[mid] == id){
            src = mid;
            break;
        }
        else if(edge_count_per_vertex_prefixSum[mid] < id){
            src = mid;
            start = mid + 1;
        }
        else end = mid - 1;
    }

    struct edge_block *root = device_vertex_dictionary->edge_block_address[src];

//    printf("id: %u, Edge src: %u\n", id, src);

    // 2. Find edge block corresponding to the given edge
    unsigned int edge_offset_in_adjacency = id - edge_count_per_vertex_prefixSum[src];
//    printf("Edge offset in adjacency: %u\n", edge_offset_in_adjacency);

    unsigned int edge_block_index = edge_offset_in_adjacency / EDGE_BLOCK_SIZE;
//    printf("Edge block index: %u\n", edge_block_index);
    unsigned long bitString = bit_string_lookup[edge_block_index];
    struct edge_block *req_edge_block = traverse_bit_string(root, bitString);

    // 3. Find destination for the edge
    unsigned int edge_offset_in_edge_block = edge_offset_in_adjacency % EDGE_BLOCK_SIZE;
    unsigned int dest = (unsigned int) req_edge_block->edge_block_entry[edge_offset_in_edge_block].destination_vertex - 1;

//    printf("id: %u, Edge destination: %u\n", id, dest);

    // 4. TC logic
    unsigned int third_vertex;
    unsigned long dest_bitString;

    struct edge_block *dest_root = device_vertex_dictionary->edge_block_address[dest];
    struct edge_block *temp;
    unsigned int dest_edge_block_index;

    ++edge_offset_in_edge_block;
    while (req_edge_block){
        while (edge_offset_in_edge_block < EDGE_BLOCK_SIZE){
            third_vertex = (unsigned int) req_edge_block->edge_block_entry[edge_offset_in_edge_block].destination_vertex;

            if(third_vertex == INFTY) continue;
            if(third_vertex == 0) break;

            --third_vertex;

            // Find whether this vertex is adjacent to dest or not
            temp = dest_root;
            dest_edge_block_index = 0;
            while(temp){
                unsigned int v;
                int endFlag = 0;

                for(int e = 0; e < EDGE_BLOCK_SIZE; ++e){
                    v = (unsigned int) temp->edge_block_entry[e].destination_vertex;

                    if(v == 0) {
                        endFlag = 1;
                        break;
                    }

                    --v;

                    if(v == third_vertex){
                        atomicAdd(&tc_per_vertex[src], 1);
                        endFlag = 1;
                        break;
                    }
                }
                if(endFlag) break;

                ++dest_edge_block_index;
                dest_bitString = bit_string_lookup[dest_edge_block_index];
                temp = traverse_bit_string(dest_root, dest_bitString);
            }

            ++edge_offset_in_edge_block;
        }

        ++edge_block_index;
        bitString = bit_string_lookup[edge_block_index];
        req_edge_block = traverse_bit_string(root, bitString);
        edge_offset_in_edge_block = 0;
    }
}

void triangleCount(struct vertex_dictionary_structure *device_vertex_dictionary, struct graph_properties *h_graph_prop){
    clock_t start, end;
    clock_t prefix_start, prefix_end;
    float total_preprocess_time = 0.0;
    float tc_kernel_time = 0.0;
    float total_time = 0.0;

    // 1. Find total edges per vertex in an array
    std::cout << "PREPROCESSING BEFORE LAUNCHING TC" << std::endl;

    unsigned int total_vertices = h_graph_prop->xDim;

    unsigned int *edge_count_per_vertex;
    cudaMalloc(&edge_count_per_vertex, total_vertices * sizeof(unsigned int));

    unsigned int thread_blocks = ceil((double) total_vertices / THREADS_PER_BLOCK);
    prefix_start = clock();
    countPerVertexEdge<<<thread_blocks, THREADS_PER_BLOCK>>>(total_vertices, edge_count_per_vertex, device_vertex_dictionary);
    cudaDeviceSynchronize();
    prefix_end = clock();

    total_preprocess_time += ((double) (prefix_end - prefix_start)) / CLOCKS_PER_SEC * 1000;

    // 2. Do prefix Sum of the above array
    unsigned int *edge_count_per_vertex_prefixSum;
    cudaMalloc(&edge_count_per_vertex_prefixSum, (total_vertices + 1) * sizeof(unsigned int));

    thrust::device_ptr<unsigned int> edge_count_per_vertex_ptr = thrust::device_pointer_cast(edge_count_per_vertex);
    thrust::device_ptr<unsigned int> edge_count_per_vertex_prefixSum_ptr = thrust::device_pointer_cast(edge_count_per_vertex_prefixSum);
    cudaDeviceSynchronize();

    prefix_start = clock();
    thrust::exclusive_scan(thrust::device, edge_count_per_vertex_ptr, edge_count_per_vertex_ptr + total_vertices, edge_count_per_vertex_prefixSum_ptr);
    cudaDeviceSynchronize();
    prefix_end = clock();

    total_preprocess_time += ((double) (prefix_end - prefix_start)) / CLOCKS_PER_SEC * 1000;

    unsigned int *total_edges;
    cudaMalloc(&total_edges, sizeof(unsigned int));
    cudaDeviceSynchronize();

    findTotalEdgeBlocks<<<1,1>>>(total_vertices, total_edges, edge_count_per_vertex_prefixSum, edge_count_per_vertex);
    cudaDeviceSynchronize();

    unsigned int *host_total_edges;
    host_total_edges = (unsigned int *)malloc(sizeof(unsigned int));

    cudaMemcpy(host_total_edges, total_edges, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // 3. Allocate space for tc
    unsigned int *tc_per_vertex;
    cudaMalloc(&tc_per_vertex, total_vertices * sizeof(tc_per_vertex));

    std::cout << "PREPROCESSING FOR TC DONE!" << std::endl;

    std::cout << "LAUNCHING EDGE-CENTRIC TC!!!" << std::endl;

    // 4. Launch edge centric kernel to calculate TC
    thread_blocks = ceil((double) *host_total_edges / THREADS_PER_BLOCK);
    start = clock();
    triangleCountingEdgeCentricNaive<<<thread_blocks, THREADS_PER_BLOCK>>>(*host_total_edges, total_vertices, edge_count_per_vertex_prefixSum, tc_per_vertex, device_vertex_dictionary);
    cudaDeviceSynchronize();
    end = clock();

    tc_kernel_time += ((double) (end - start)) / CLOCKS_PER_SEC * 1000;
    total_time = total_preprocess_time + tc_kernel_time;

    std::cout << "EDGE-CENTRIC TC KERNEL FINISHED!" << std::endl;
    std::cout << "PRINTING TC VALUES-->" << std::endl;
    printDist<<<1,1>>>(total_vertices, tc_per_vertex);
    cudaDeviceSynchronize();

    std::cout << "Total Time for Preprocessing: " << total_preprocess_time << std::endl;
    std::cout << "Total Time for TC Kernel: " << tc_kernel_time << std::endl;
    std::cout << "Total Time for TC: " << total_time << std::endl;
}

int main(void) {
    double fullTime = 0.0;

//    char fileLoc[30] = "../../Graphs/input.mtx";
    // char fileLoc[20] = "input1.mtx";
    // char fileLoc[20] = "inputSSSP.mtx";
    // char fileLoc[20] = "chesapeake_mod.mtx";
    // char fileLoc[30] = "klein-b1.mtx";
//     char fileLoc[30] = "../../Graphs/chesapeake.mtx";
    // char fileLoc[30] = "bio-celegansneural.mtx";
    // char fileLoc[30] = "inf-luxembourg_osm.mtx";
    // char fileLoc[30] = "rgg_n_2_16_s0.mtx";
    // char fileLoc[30] = "delaunay_n10.mtx";
    // char fileLoc[30] = "delaunay_n12.mtx";
    // char fileLoc[30] = "delaunay_n13.mtx";
    // char fileLoc[30] = "delaunay_n16.mtx";
    // char fileLoc[30] = "delaunay_n17.mtx";
    // char fileLoc[30] = "fe-ocean.mtx";
//     char fileLoc[40] = "../../Graphs/co-papers-dblp.mtx";
//     char fileLoc[40] = "../../Graphs/co-papers-citeseer.mtx";
//     char fileLoc[40] = "../../Graphs/hugetrace-00020.mtx";
//     char fileLoc[50] = "../../Graphs/channel-500x100x100-b050.mtx";
    // char fileLoc[30] = "kron_g500-logn16.mtx";
    // char fileLoc[30] = "kron_g500-logn17.mtx";
//     char fileLoc[40] = "../../Graphs/kron_g500-logn21.mtx";
    // char fileLoc[30] = "delaunay_n22.mtx";
    // char fileLoc[30] = "delaunay_n23.mtx";
//     char fileLoc[40] = "../../Graphs/delaunay_n24.mtx";
//     char fileLoc[40] = "../../Graphs/inf-europe_osm.mtx";
    // char fileLoc[30] = "rgg_n_2_23_s0.mtx";
     char fileLoc[40] = "../../Graphs/rgg_n_2_24_s0.mtx";
    // char fileLoc[30] = "nlpkkt240.mtx";
    // char fileLoc[30] = "uk-2005.mtx";
    // char fileLoc[30] = "twitter7.mtx";
    // char fileLoc[30] = "sk-2005.mtx";

    memory_usage();


    // some random inits
    unsigned long choice = 1;
    // printf("Please enter structure of edge blocks\n1. Unsorted\n2. Sorted\n");
    // scanf("%lu", &choice);

    clock_t section1, section2, section2a, section3, search_times, al_time, vd_time, time_req, push_to_queues_time, temp_time, delete_time, init_time, pageRank_time, triangleCounting_time, sssp_time, vertex_insert_time;

    struct graph_properties *h_graph_prop = (struct graph_properties*)malloc(sizeof(struct graph_properties));
    thrust::host_vector <unsigned long> h_source(1);
    thrust::host_vector <unsigned long> h_destination(1);
    // thrust::host_vector <unsigned long> h_source_degree(1);
    // thrust::host_vector <unsigned long> h_prefix_sum_vertex_degrees(1);
    thrust::host_vector <unsigned long> h_prefix_sum_edge_blocks(1);
    thrust::host_vector <unsigned long> h_edge_blocks_count_init(1);

    thrust::host_vector <unsigned long> h_source_degrees(1);
    thrust::host_vector <unsigned long> h_csr_offset(1);
    thrust::host_vector <unsigned long> h_csr_edges(1);

    thrust::host_vector <unsigned long> h_source_degrees_new(1);
    thrust::host_vector <unsigned long> h_csr_offset_new(1);
    thrust::host_vector <unsigned long> h_csr_edges_new(1);
    thrust::host_vector <unsigned long> h_edge_blocks_count(1);
    thrust::host_vector <unsigned long> h_prefix_sum_edge_blocks_new(1);

    // thrust::host_vector <unsigned long> h_batch_update_data(1);
    // initialize bit string lookup
    unsigned long thread_blocks = ceil(double(BIT_STRING_LOOKUP_SIZE) / THREADS_PER_BLOCK);

    build_bit_string_lookup<<< thread_blocks, THREADS_PER_BLOCK>>>();
    // print_bit_string<<< 1, 1>>>();

    // unsigned long graph_choice;
    // std::cout << std::endl << "Enter input type" << std::endl << "1. Real Graph" << std::endl << "2. Random Graph" << std::endl;
    // std::cin >> graph_choice;

    // if(graph_choice ==1) {
    // reading file, after function call h_source has data on source vertex and h_destination on destination vertex
    // both represent the edge data on host
    section1 = clock();
    readFile(fileLoc, h_graph_prop, h_source, h_destination, h_source_degrees, 0);
    section1 = clock() - section1;

    std::cout << "File read complete" << std::endl;
    // }
    // else {

    //     section1 = clock();
    //     readFile(fileLoc, h_graph_prop, h_source, h_destination, h_source_degrees, 1);

    //     // h_graph_prop->total_edges = BATCH_SIZE;

    //     generate_random_batch(h_graph_prop->xDim, h_graph_prop->total_edges, h_source, h_destination, h_source_degrees_new, h_source_degrees);
    //     std::cout << "Generated random batch" << std::endl;
    //     // std::cout << "Vertex size : " << h_graph_prop->xDim << ", Edge size : " << h_graph_prop->total_edges << std::endl;

    //     section1 = clock() - section1;

    // }

    // below device vectors are for correctness check of the data structure at the end
    // thrust::device_vector <unsigned long> d_c_source(h_graph_prop->total_edges);
    // thrust::device_vector <unsigned long> d_c_destination(h_graph_prop->total_edges);
    // thrust::copy(h_source.begin(), h_source.end(), d_c_source.begin());
    // thrust::copy(h_destination.begin(), h_destination.end(), d_c_destination.begin());
    // unsigned long* c_source = thrust::raw_pointer_cast(d_c_source.data());
    // unsigned long* c_destination = thrust::raw_pointer_cast(d_c_destination.data());

    section2 = clock();

    unsigned long vertex_size = h_graph_prop->xDim;
    unsigned long edge_size = h_graph_prop->total_edges;

    unsigned long max_degree = h_source_degrees[0];
    unsigned long sum_degree = h_source_degrees[0];
    for(unsigned long i = 1 ; i < vertex_size ; i++) {

        if(h_source_degrees[i] > max_degree)
            max_degree = h_source_degrees[i];
        sum_degree += h_source_degrees[i];
    }

    std::cout << std::endl << "Max degree of the graph is " << max_degree << std::endl;
    std::cout << "Average degree of the graph is " << sum_degree / vertex_size << std::endl << std::endl;

    unsigned long batch_size = BATCH_SIZE;
    unsigned long total_batches = ceil(double(edge_size) / batch_size);
    // unsigned long batch_number = 0;

    std::cout << "Batches required is " << total_batches << std::endl;

    // h_source_degree.resize(vertex_size);
    // h_prefix_sum_vertex_degrees.resize(vertex_size);
    // h_prefix_sum_edge_blocks.resize(vertex_size);
    h_edge_blocks_count_init.resize(vertex_size);

    // below one is suspected redundant op
    h_source_degrees.resize(vertex_size);
    // h_csr_offset.resize(vertex_size + 1);
    // h_csr_edges.resize(edge_size);

    h_source_degrees_new.resize(vertex_size);
    h_csr_offset_new.resize((vertex_size + 1));
    // h_csr_edges_new.resize(batch_size);
    h_edge_blocks_count.resize(vertex_size);
    h_prefix_sum_edge_blocks_new.resize(vertex_size + 1);

    unsigned long *h_batch_update_data;

    if(edge_size > BATCH_SIZE) {
        h_csr_edges_new.resize(edge_size);
        // h_batch_update_data = (unsigned long *)malloc((edge_size) * (sizeof(unsigned long)));
    }
    else {
        h_csr_edges_new.resize(BATCH_SIZE);
        // h_batch_update_data = (unsigned long *)malloc((BATCH_SIZE) * (sizeof(unsigned long)));
    }

    // h_batch_update_data.resize(vertex_size + 1 + batch_size + vertex_size);
    // unsigned long *h_batch_update_data = (unsigned long *)malloc((batch_size) * (sizeof(unsigned long)));

    // generateCSR(vertex_size, edge_size, h_source, h_destination, h_csr_offset, h_csr_edges, h_source_degrees);



    // generateCSRnew(vertex_size, edge_size, h_source, h_destination, h_csr_offset_new, h_csr_edges_new, h_source_degrees_new, h_edge_blocks_count, BATCH_SIZE, total_batches);


    section2 = clock() - section2;

    std::cout << "CSR Generation complete" << std::endl;

    // std::cout << "Generated CSR" << std::endl;
    // for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //     std::cout << h_csr_offset[i] << " ";
    // std::cout << std::endl;
    // for(unsigned long i = 0 ; i < edge_size ; i++)
    //     std::cout << h_csr_edges[i] << " ";
    // std::cout << std::endl;

    // thrust::fill(h_source_degree.begin(), h_source_degree.end(), 0);

    // std::cout << "Check, " << h_source.size() << " and " << h_destination.size() << std::endl;

    // sleep(5);

    section2a = clock();
    vertex_insert_time = clock();

    // unsigned long vertex_dictionary_size = 8;
    // std::cout << "Power of 5167 is " << calculate_log_base2(5167) << std::endl;
    unsigned long vertex_dictionary_size = calculate_power_base2(calculate_log_base2(vertex_size) + 1);
    std::cout << "VD size is " << vertex_dictionary_size << std::endl;

    struct vertex_dictionary_structure *device_vertex_dictionary;
    cudaMalloc(&device_vertex_dictionary, sizeof(struct vertex_dictionary_structure));

    // thrust::device_vector <unsigned long> d_vertex_id(vertex_dictionary_size);
    // thrust::device_vector <unsigned long> d_edge_block_count_VD(vertex_dictionary_size);
    // thrust::device_vector <unsigned int> d_active_edge_count_VD(vertex_dictionary_size);
    // thrust::device_vector <unsigned long> d_last_insert_edge_offset(vertex_dictionary_size);
    // thrust::device_vector <struct edge_block *> d_last_insert_edge_block(vertex_dictionary_size);
    // thrust::device_vector <struct edge_block *> d_edge_block_address(vertex_dictionary_size);


    // unsigned long *d_vertex_id_pointer = thrust::raw_pointer_cast(d_vertex_id.data());
    // unsigned long *d_edge_block_count_VD_pointer = thrust::raw_pointer_cast(d_edge_block_count_VD.data());
    // unsigned int *d_active_edge_count_VD_pointer = thrust::raw_pointer_cast(d_active_edge_count_VD.data());
    // unsigned long *d_last_insert_edge_offset_pointer = thrust::raw_pointer_cast(d_last_insert_edge_offset.data());
    // struct edge_block **d_last_insert_edge_block_pointer = thrust::raw_pointer_cast(d_last_insert_edge_block.data());
    // struct edge_block **d_edge_block_address_pointer = thrust::raw_pointer_cast(d_edge_block_address.data());

    unsigned long *d_vertex_id, *d_edge_block_count_VD, *d_last_insert_edge_offset;
    unsigned int *d_active_edge_count_VD;
    struct edge_block **d_last_insert_edge_block, **d_edge_block_address;

    cudaMalloc(&d_vertex_id, 3 * vertex_dictionary_size * sizeof(unsigned long));
    d_edge_block_count_VD = d_vertex_id + vertex_dictionary_size;
    d_last_insert_edge_offset = d_vertex_id + (2 * vertex_dictionary_size);

    cudaMalloc(&d_active_edge_count_VD, vertex_dictionary_size * sizeof(unsigned int));

    cudaMalloc((struct edge_block **)&d_last_insert_edge_block, 2 * vertex_dictionary_size * sizeof(struct edge_block *));
    d_edge_block_address = d_last_insert_edge_block + vertex_dictionary_size;

    // initializing vertex dictionary
    // unsigned long *vertex_id;
    // struct adjacency_sentinel_new *vertex_adjacency[VERTEX_BLOCK_SIZE];
    // unsigned long *edge_block_count;
    // unsigned long active_vertex_count;

    // // below is directly from edge sentinel node
    // // unsigned long edge_block_count[VERTEX_BLOCK_SIZE];
    // unsigned int *active_edge_count;
    // unsigned long *last_insert_edge_offset;

    // struct edge_block **last_insert_edge_block;
    // // struct edge_block *edge_block_address[100];
    // struct edge_block **edge_block_address;
    // cudaMalloc(device_vertex_dictionary->vertex_id, vertex_dictionary_size * sizeof(unsigned long));

    struct adjacency_sentinel_new *device_adjacency_sentinel;
    // cudaMalloc((struct adjacency_sentinel_new**)&device_adjacency_sentinel, vertex_size * sizeof(struct adjacency_sentinel_new));
    cudaDeviceSynchronize();

    vertex_insert_time = clock() - vertex_insert_time;
    section2a = clock() - section2a;
    // init_time = section2a;

    // below till cudaMalloc is temp code


    unsigned long total_edge_blocks_count_init = 0;
    // std::cout << "Edge blocks calculation" << std::endl << "Source\tEdge block count\tGPU address" << std::endl;
    for(unsigned long i = 0 ; i < h_graph_prop->xDim ; i++) {

        unsigned long edge_blocks = ceil(double(h_source_degrees[i]) / EDGE_BLOCK_SIZE);
        h_edge_blocks_count_init[i] = edge_blocks;
        total_edge_blocks_count_init += edge_blocks;


    }

    total_edge_blocks_count_init = total_edge_blocks_count_init * 2;

    printf("Total edge blocks needed = %lu, %luMB\n", total_edge_blocks_count_init, (total_edge_blocks_count_init * sizeof(struct edge_block)) /1024/1024);
    memory_usage();
    cudaDeviceSynchronize();
    // h_prefix_sum_edge_blocks[0] = h_edge_blocks_count_init[0];
    // // printf("Prefix sum array edge blocks\n%ld ", h_prefix_sum_edge_blocks[0]);
    // for(unsigned long i = 1 ; i < h_graph_prop->xDim ; i++) {

    //     h_prefix_sum_edge_blocks[i] += h_prefix_sum_edge_blocks[i-1] + h_edge_blocks_count_init[i];
    //     // printf("%ld ", h_prefix_sum_edge_blocks[i]);

    // }

    temp_time = clock();
    struct edge_block *device_edge_block;
    cudaMalloc((struct edge_block**)&device_edge_block, total_edge_blocks_count_init * sizeof(struct edge_block));
    // cudaMalloc((struct edge_block**)&device_edge_block, 2 * total_edge_blocks_count_init * sizeof(struct edge_block));
    cudaDeviceSynchronize();
    temp_time = clock() - temp_time;
    section2a += temp_time;
    init_time = section2a;

    // thrust::device_vector <unsigned long> d_edge_blocks_count_init(vertex_size);
    // thrust::copy(h_edge_blocks_count_init.begin(), h_edge_blocks_count_init.end(), d_edge_blocks_count_init.begin());
    // unsigned long* ebci = thrust::raw_pointer_cast(d_edge_blocks_count_init.data());

    // thrust::device_vector <unsigned long> d_prefix_sum_edge_blocks(vertex_size);
    // thrust::copy(h_prefix_sum_edge_blocks.begin(), h_prefix_sum_edge_blocks.end(), d_prefix_sum_edge_blocks.begin());
    // unsigned long* pseb = thrust::raw_pointer_cast(d_prefix_sum_edge_blocks.data());

    // thrust::device_vector <unsigned long> d_source(batch_size);
    // thrust::device_vector <unsigned long> d_destination(batch_size);
    // thrust::device_vector <unsigned long> d_prefix_sum_vertex_degrees(vertex_size);

    // thrust::device_vector <unsigned long> d_csr_offset(vertex_size + 1);
    // thrust::device_vector <unsigned long> d_csr_edges(edge_size);

    // thrust::device_vector <unsigned long> d_source(edge_size);
    // thrust::device_vector <unsigned long> d_destination(edge_size);
    thrust::device_vector <unsigned long> d_csr_edges_new(edge_size);


    thrust::device_vector <unsigned long> d_source_degrees_new(vertex_size);
    thrust::device_vector <unsigned long> d_csr_offset_new(vertex_size + 1);
    // thrust::device_vector <unsigned long> d_csr_edges_new(batch_size);

    // thrust::device_vector <unsigned long> d_edge_blocks_count(vertex_size);
    thrust::device_vector <unsigned long> d_edge_blocks_count(vertex_size + 1);
    thrust::device_vector <unsigned long> d_prefix_sum_edge_blocks_new(vertex_size + 1);

    thrust::device_vector <unsigned long> d_source_vector(EDGE_PREALLOCATE_LIST_SIZE);
    thrust::device_vector <unsigned long> d_source_vector_1(1);
    thrust::device_vector <unsigned long> d_thread_count_vector(vertex_size + 1);

    if(edge_size < batch_size) {
        // d_source.resize(batch_size);
        // d_destination.resize(batch_size);
        d_csr_edges_new.resize(batch_size);
        // d_source_vector_1.resize(batch_size);
    }

    // if(batch_size > edge_size)
    //     d_source_vector_1.resize(batch_size);

    // thrust::copy(h_source.begin(), h_source.end(), d_source.begin());
    // thrust::copy(h_destination.begin(), h_destination.end(), d_destination.begin());

    // thrust::device_vector <unsigned long> d_source(BATCH_SIZE);
    // thrust::device_vector <unsigned long> d_destination(BATCH_SIZE);

    // thrust::device_vector <unsigned long> d_batch_update_data(vertex_size + 1 + batch_size + vertex_size);
    temp_time = clock();
    // unsigned long *d_batch_update_data;
    // cudaMalloc((unsigned long**)&d_batch_update_data, (edge_size) * sizeof(unsigned long));
    cudaDeviceSynchronize();
    temp_time = clock() - temp_time;
    init_time += temp_time;
    // section2a = clock() - section2a;


    // Parallelize this
    // push_preallocate_list_to_device_queue_kernel<<< 1, 1>>>(device_vertex_block, device_edge_block, dapl, vertex_blocks_count_init, ebci, total_edge_blocks_count_init);

    // thread_blocks = ceil(double(vertex_blocks_count_init) / THREADS_PER_BLOCK);

    time_req = clock();
    push_to_queues_time = clock();


    // data_structure_init<<< 1, 1>>>(device_vertex_dictionary, d_vertex_id_pointer, d_edge_block_count_VD_pointer, d_active_edge_count_VD_pointer, d_last_insert_edge_offset_pointer, d_last_insert_edge_block_pointer, d_edge_block_address_pointer);
    data_structure_init<<< 1, 1>>>(device_vertex_dictionary, d_vertex_id, d_edge_block_count_VD, d_active_edge_count_VD, d_last_insert_edge_offset, d_last_insert_edge_block, d_edge_block_address);
    // parallel_push_vertex_preallocate_list_to_device_queue<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_block, dapl, vertex_blocks_count_init);

    thread_blocks = ceil(double(total_edge_blocks_count_init) / THREADS_PER_BLOCK);
    // thread_blocks = ceil(double(2 * total_edge_blocks_count_init) / THREADS_PER_BLOCK);

    // parallel_push_edge_preallocate_list_to_device_queue<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, dapl, ebci, total_edge_blocks_count_init);
    parallel_push_edge_preallocate_list_to_device_queue_v1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, total_edge_blocks_count_init);
    // parallel_push_edge_preallocate_list_to_device_queue_v1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, 2 * total_edge_blocks_count_init);

    parallel_push_queue_update<<< 1, 1>>>(total_edge_blocks_count_init);
    // parallel_push_queue_update<<< 1, 1>>>(2 * total_edge_blocks_count_init);
    cudaDeviceSynchronize();

    push_to_queues_time = clock() - push_to_queues_time;

    init_time += push_to_queues_time;

    // sleep(5);

    // Pass raw array and its size to kernel
    // thread_blocks = ceil(double(vertex_blocks_count_init) / THREADS_PER_BLOCK);
    // std::cout << "Thread blocks vertex init = " << thread_blocks << std::endl;
    // vertex_dictionary_init<<< thread_blocks, THREADS_PER_BLOCK>>>(dvpl, dapl, vertex_blocks_count_init, d_graph_prop, vertex_size, ebci);
    vd_time = clock();
    // vertex_dictionary_init<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_block, device_adjacency_sentinel, vertex_blocks_count_init, d_graph_prop, vertex_size, ebci);
    thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
    // std::cout << "Thread blocks vertex init = " << thread_blocks << std::endl;
    // parallel_vertex_dictionary_init<<< thread_blocks, THREADS_PER_BLOCK>>>(device_adjacency_sentinel, vertex_blocks_count_init, d_graph_prop, vertex_size, ebci, device_vertex_dictionary);
    parallel_vertex_dictionary_init_v1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_adjacency_sentinel, vertex_size, NULL, device_vertex_dictionary);

    vd_time = clock() - vd_time;
    vertex_insert_time += vd_time;

    init_time += vd_time;

    // h_source.resize(batch_size);
    // h_destination.resize(batch_size);
    // h_source.resize(1);
    // h_destination.resize(1);

    // thrust::copy(h_csr_offset.begin(), h_csr_offset.end(), d_csr_offset.begin());
    // thrust::copy(h_csr_edges.begin(), h_csr_edges.end(), d_csr_edges.begin());
    // unsigned long* d_csr_offset_pointer = thrust::raw_pointer_cast(d_csr_offset.data());
    // unsigned long* d_csr_edges_pointer = thrust::raw_pointer_cast(d_csr_edges.data());
    // unsigned long* d_source_pointer = thrust::raw_pointer_cast(d_source.data());
    // unsigned long* d_destination_pointer = thrust::raw_pointer_cast(d_destination.data());

    unsigned long* d_source_degrees_new_pointer = thrust::raw_pointer_cast(d_source_degrees_new.data());
    unsigned long* d_csr_offset_new_pointer = thrust::raw_pointer_cast(d_csr_offset_new.data());
    unsigned long* d_csr_edges_new_pointer = thrust::raw_pointer_cast(d_csr_edges_new.data());
    unsigned long* d_edge_blocks_count_pointer = thrust::raw_pointer_cast(d_edge_blocks_count.data());
    unsigned long* d_prefix_sum_edge_blocks_new_pointer = thrust::raw_pointer_cast(d_prefix_sum_edge_blocks_new.data());

    unsigned long* d_source_vector_pointer = thrust::raw_pointer_cast(d_source_vector.data());
    unsigned long* d_source_vector_1_pointer = thrust::raw_pointer_cast(d_source_vector_1.data());

    // unsigned long* d_source_pointer = thrust::raw_pointer_cast(d_source.data());
    // unsigned long* d_destination_pointer = thrust::raw_pointer_cast(d_destination.data());

    // unsigned long* d_batch_update_data_pointer = thrust::raw_pointer_cast(d_batch_update_data.data());

    // struct batch_update_data batch_update;

    thrust::host_vector <unsigned long> space_remaining(vertex_size);
    unsigned long total_edge_blocks_count_batch;

    // al_time = clock();
    al_time = 0;

    // std::cout << "Enter type of insertion required" << std::endl << "1. Regular batched edge insertion" << std::endl << "2. Edge Insert and Delete performance benchmark" << std::endl << "3. Vertex Insert and Delete performance benchmark" << std::endl;
    // std::cin >> choice;


    // if(choice == 1) {
    //     for(unsigned long i = 0 ; i < 1 ; i++) {
    //     // for(unsigned long i = 0 ; i < 7 ; i++) {

    //         // section2 = clock();

    //         // for(unsigned long i = 0 ; i < h_graph_prop->total_edges ; i++) {

    //         //     // printf("%lu and %lu\n", h_source[i], h_destination[i]);
    //         //     h_source_degree[h_source[i] - 1]++;

    //         // }

    //         std::cout << std::endl << "Iteration " << i << std::endl;

    //         // thrust::copy(h_csr_offset_new.begin(), h_csr_offset_new.end(), d_csr_offset_new.begin());
    //         // thrust::copy(h_csr_edges_new.begin(), h_csr_edges_new.end(), d_csr_edges_new.begin());

    //         // std::cout << std::endl << "Iteration " << i << std::endl;

    //         total_edge_blocks_count_batch = 0;

    //         // std::cout << "COO pairs" << std::endl;
    //         // for(unsigned long z = 0 ; z < edge_size ; z++)
    //         //     std::cout << h_source[z] << " ";
    //         // std::cout << std::endl;
    //         // for(unsigned long z = 0 ; z < edge_size ; z++)
    //         //     std::cout << h_destination[z] << " ";
    //         // std::cout << std::endl;

    //         // generate_csr_batch(vertex_size, edge_size, max_degree, h_source, h_destination, h_csr_offset_new, h_csr_edges_new, h_source_degrees_new, h_edge_blocks_count, h_prefix_sum_edge_blocks_new, h_batch_update_data, edge_size, total_batches, i, space_remaining, &total_edge_blocks_count_batch, &init_time);

    //         thrust::exclusive_scan(h_source_degrees.begin(), h_source_degrees.begin() + vertex_size + 1, h_csr_offset_new.begin());
    //         cudaDeviceSynchronize();
    //         generate_csr_batch_tester(vertex_size, edge_size, max_degree, h_source, h_destination, h_csr_offset_new, h_csr_edges_new, h_source_degrees_new, h_edge_blocks_count, h_prefix_sum_edge_blocks_new, h_batch_update_data, edge_size, total_batches, i, space_remaining, &total_edge_blocks_count_batch, &init_time);
    //         // unsigned long thread_divider = 10;
    //         // thread_blocks = ceil((double(vertex_size / thread_divider)) / THREADS_PER_BLOCK);
    //         // device_coo_to_csr<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, d_source_pointer, d_destination_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, edge_size, d_source_degrees_new_pointer, thread_divider);
    //         thrust::copy(h_csr_offset_new.begin(), h_csr_offset_new.end(), d_csr_offset_new.begin());
    //         thrust::copy(h_csr_edges_new.begin(), h_csr_edges_new.end(), d_csr_edges_new.begin());
    //         thrust::copy(h_source_degrees.begin(), h_source_degrees.end(), d_source_degrees_new.begin());
    //         // thrust::copy(h_destination.begin(), h_destination.begin() + edge_size, d_csr_edges_new.begin());
    //         cudaDeviceSynchronize();


    //         // for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //         //     std::thread sort_thread(csr_sort, i, vertex_size, thrust::raw_pointer_cast(h_csr_offset_new.data()), thrust::raw_pointer_cast(h_csr_edges_new.data()));

    //         //     sort_thread.join();
    //         // }

    //         // for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //         //     std::thread remove_duplicates_thread(csr_remove_duplicates, i, vertex_size, thrust::raw_pointer_cast(h_csr_offset_new.data()), thrust::raw_pointer_cast(h_csr_edges_new.data()), thrust::raw_pointer_cast(h_source_degrees_new.data()));

    //         //     remove_duplicates_thread.join();
    //         // }

    //         // reconstruct_deduplicated_csr(vertex_size, edge_size, thrust::raw_pointer_cast(h_csr_offset_new.data()), thrust::raw_pointer_cast(h_csr_edges_new.data()), thrust::raw_pointer_cast(h_source_degrees_new.data()));
    //         // for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //         //     std::thread h_i_p(host_insert_preprocessing, i, vertex_size, thrust::raw_pointer_cast(h_source_degrees_new.data()), thrust::raw_pointer_cast(space_remaining.data()), 0, thrust::raw_pointer_cast(h_edge_blocks_count.data()));

    //         //     h_i_p.join();
    //         // }

    //         thread_blocks = ceil(double(edge_size) / THREADS_PER_BLOCK);
    //         device_remove_batch_duplicates<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, edge_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);
    //         cudaDeviceSynchronize();
    //         // thrust::exclusive_scan(d_source_degrees_new.begin(), d_source_degrees_new.begin() + vertex_size + 1, d_csr_offset_new.begin());
    //         // cudaDeviceSynchronize();
    //         thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
    //         device_insert_preprocessing<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, edge_size, d_source_degrees_new_pointer, 0, d_edge_blocks_count_pointer);
    //         cudaDeviceSynchronize();
    //         thrust::exclusive_scan(d_edge_blocks_count.begin(), d_edge_blocks_count.begin() + vertex_size + 1, d_prefix_sum_edge_blocks_new.begin());
    //         cudaDeviceSynchronize();

    //         cudaMemcpy(&total_edge_blocks_count_batch, d_prefix_sum_edge_blocks_new_pointer + vertex_size, sizeof(unsigned long),cudaMemcpyDeviceToHost);

    //         //  new code end

    //         // thrust::copy(h_prefix_sum_edge_blocks_new.begin(), h_prefix_sum_edge_blocks_new.end(), d_prefix_sum_edge_blocks_new.begin());

    //         // host_insert_preprocessing(vertex_size, thrust::raw_pointer_cast(h_source_degrees_new.data()), thrust::raw_pointer_cast(space_remaining.data()), 0, thrust::raw_pointer_cast(h_edge_blocks_count.data()));


    //         // device_reconstruct_csr<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, edge_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);
    //         // cudaDeviceSynchronize();

    //         // remove_batch_duplicates(vertex_size, edge_size, h_csr_offset_new, h_csr_edges_new, h_batch_update_data, batch_size);


    //         // std::cout << "Total edge blocks is " << total_edge_blocks_count_batch  << std::endl;

    //         // std::cout << std::endl << "Printing batched CSR after sort CPU threads" << std::endl << "Source degrees\t\t" << std::endl;
    //         // for(unsigned long j = 0 ; j < vertex_size ; j++) {
    //         //     std::cout << h_source_degrees_new[j] << " ";
    //         //     if((j + 1) % vertex_size == 0)
    //         //         std::cout << std::endl;
    //         // }
    //         // std::cout << std::endl << "CSR offset\t\t" << std::endl;
    //         // for(unsigned long j = 0 ; j < (vertex_size + 1) ; j++) {
    //         //     std::cout << h_csr_offset_new[j] << " ";
    //         //     if(((j + 1) % (vertex_size + 1)) == 0)
    //         //         std::cout << std::endl;
    //         // }
    //         // std::cout << std::endl << "CSR edges\t\t" << std::endl;
    //         // for(unsigned long j = 0 ; j < h_csr_offset_new[vertex_size] ; j++) {
    //         //     std::cout << h_csr_edges_new[j] << " ";
    //         //     if((j + 1) % h_csr_offset_new[vertex_size] == 0)
    //         //         std::cout << std::endl;
    //         // }
    //         // std::cout << std::endl << "Edge blocks count\t\t" << std::endl;
    //         // for(unsigned long j = 0 ; j < vertex_size ; j++) {
    //         //     std::cout << h_edge_blocks_count[j] << " ";
    //         //     if((j + 1) % vertex_size == 0)
    //         //         std::cout << std::endl;
    //         // }
    //         // std::cout << std::endl << "Prefix sum edge blocks\t\t" << std::endl;
    //         // for(unsigned long j = 0 ; j < vertex_size ; j++) {
    //         //     std::cout << h_prefix_sum_edge_blocks_new[j] << " ";
    //         //     if((j + 1) % vertex_size == 0)
    //         //         std::cout << std::endl;
    //         // }
    //         // std::cout << std::endl << "Space remaining\t\t" << std::endl;
    //         // for(unsigned long j = 0 ; j < vertex_size ; j++)
    //         //     std::cout << space_remaining[j] << " ";
    //         // std::cout << std::endl;
    //         // std::cout << std::endl << "Batch update vector\t\t" << std::endl;
    //         // for(unsigned long j = 0 ; j < vertex_size + 1 + batch_size + vertex_size ; j++)
    //         //     std::cout << h_batch_update_data[j] << " ";
    //         // std::cout << std::endl;
    //         // std::cout << std::endl << std::endl;

    //         // temp_time = clock();


    //         // thrust::copy(h_source_degrees_new.begin(), h_source_degrees_new.end(), d_source_degrees_new.begin());
    //         // thrust::copy(h_csr_offset_new.begin(), h_csr_offset_new.end(), d_csr_offset_new.begin());
    //         // thrust::copy(h_destination.begin(), h_destination.end(), d_csr_edges_new.begin());
    //         // thrust::copy(h_edge_blocks_count.begin(), h_edge_blocks_count.end(), d_edge_blocks_count.begin());
    //         // thrust::copy(h_prefix_sum_edge_blocks_new.begin(), h_prefix_sum_edge_blocks_new.end(), d_prefix_sum_edge_blocks_new.begin());

    //         // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
    //         // device_generate_csr_batch<<<thread_blocks, THREADS_PER_BLOCK>>>();

    //         temp_time = clock();

    //         // thrust::copy(h_batch_update_data.begin(), h_batch_update_data.end(), d_batch_update_data.begin());
    //         // cudaMemcpy(d_batch_update_data, &h_batch_update_data, (vertex_size + 1 + batch_size + vertex_size) * sizeof(unsigned long), cudaMemcpyHostToDevice);
    //         cudaMemcpy(d_batch_update_data, h_batch_update_data, (batch_size) * sizeof(unsigned long), cudaMemcpyHostToDevice);

    //         // thrust::device_vector <unsigned long> d_source_degrees_new(vertex_size);
    //         // thrust::device_vector <unsigned long> d_csr_offset_new(vertex_size + 1);
    //         // thrust::device_vector <unsigned long> d_csr_edges_new(batch_size);
    //         // thrust::device_vector <unsigned long> d_edge_blocks_count_new(vertex_size);
    //         // thrust::device_vector <unsigned long> d_prefix_sum_edge_blocks_new(vertex_size);

    //         // temp_time = clock();

    //         unsigned long start_index = i * batch_size;
    //         unsigned long end_index;

    //         unsigned long remaining_edges = edge_size - start_index;

    //         if(remaining_edges <= batch_size)
    //             end_index = edge_size;
    //         else
    //             end_index = (i + 1) * batch_size;

    //         unsigned long current_batch = end_index - start_index;

    //         // std::cout << "Current batch is " << current_batch << std::endl;

    //         // cudaDeviceSynchronize();
    //         // vd_time = clock() - vd_time;

    //         // std::cout << "Thread blocks edge init = " << thread_blocks << std::endl;

    //         // sleep(5);

    //         // adjacency_list_init<<< thread_blocks, THREADS_PER_BLOCK>>>(depl, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size);
    //         // al_time = clock();
    //         // adjacency_list_init<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks);
    //         // adjacency_list_init_modded<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks, device_vertex_dictionary);
    //         // adjacency_list_init_modded_v1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, psvd, thread_blocks, device_vertex_dictionary);
    //         // adjacency_list_init_modded_v2<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, psvd, thread_blocks, device_vertex_dictionary, i, current_batch);
    //         // adjacency_list_init_modded_v3<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_pointer, d_csr_edges_pointer);

    //         // adjacency_list_init_modded_v4<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_pointer, d_csr_edges_pointer);

    //         // std::cout << "Checkpoint 1" << std::endl;
    //         // temp_time = clock();

    //         // v5 code below
    //         thread_blocks = ceil(double(h_graph_prop->xDim) / THREADS_PER_BLOCK);
    //         adjacency_list_init_modded_v5<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);
    //         update_edge_queue<<< 1, 1>>>(total_edge_blocks_count_batch);

    //         // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
    //         // batched_edge_inserts_preprocessing_v6<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_vector_pointer);

    //         // thread_blocks = ceil(double(vertex_size + h_prefix_sum_edge_blocks_new[vertex_size]) / THREADS_PER_BLOCK);
    //         // batched_edge_inserts_v6<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_vector_pointer);

    //         // update_edge_queue<<< 1, 1>>>(total_edge_blocks_count_batch);
    //         cudaDeviceSynchronize();
    //         memory_usage();
    //         temp_time = clock() - temp_time;

    //         // std::cout << "Checkpoint 2" << std::endl;

    //         // if(i != 7) {
    //         //     std::cout << "Hit here" << std::endl;
    //         // }
    //         // temp_time = clock() - temp_time;
    //         al_time += temp_time;
    //         // std::cout << "Batch #" << i << " took " << (float)temp_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    //         // Seperate kernel for updating queues due to performance issues for global barriers
    //         // printKernelmodded_v1<<< 1, 1>>>(device_vertex_dictionary, vertex_size);
    //         // if(i < 10)
    //         // printKernelmodded_v2<<< 1, 1>>>(device_vertex_dictionary, vertex_size);
    //         // cbt_stats<<< 1, 1>>>(device_vertex_dictionary, vertex_size);

    //         cudaDeviceSynchronize();
    //         // std::cout << "Outside checkpoint" << std::endl;


    //     }

    //     cudaDeviceSynchronize();

    //     // for(long i = 0 ; i < edge_size ; i++)
    //     //     std::cout << h_source[i] << " and " << h_destination[i] << std::endl;

    //     // printf("\nCorrectness check of data structure\n");
    //     // printf("*---------------*\n");
    //     // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
    //     // // correctness_check_kernel<<< 1, 1>>>(device_vertex_dictionary, vertex_size, edge_size, c_source, c_destination);
    //     // unsigned long h_correctness_flag, *d_correctness_flag;
    //     // cudaMalloc(&d_correctness_flag, sizeof(unsigned long));
    //     // // correctness_check_kernel<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, edge_size, d_csr_offset_pointer, d_csr_edges_pointer, d_correctness_flag);
    //     // correctness_check_kernel_v1<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, edge_size, d_csr_offset_pointer, d_csr_edges_pointer, d_correctness_flag);
    //     // cudaMemcpy(&h_correctness_flag, d_correctness_flag, sizeof(unsigned long),cudaMemcpyDeviceToHost);
    //     // cudaDeviceSynchronize();
    //     // search_times = clock() - search_times;

    //     // if(h_correctness_flag)
    //     //     std::cout << "Data structure corrupted" << std::endl;
    //     // else
    //     //     std::cout << "Data structure uncorrupted" << std::endl;
    //     // printf("*---------------*\n\n");


    //     // sleep(5);

    //     // printKernel<<< 1, 1>>>(device_vertex_block, vertex_size);
    //     // printKernelmodded<<< 1, 1>>>(device_vertex_dictionary, vertex_size);
    //     // printKernelmodded_v1<<< 1, 1>>>(device_vertex_dictionary, vertex_size);

    //     cudaDeviceSynchronize();
    //     delete_time = 0;
    // }

    // else {
    // for(unsigned long i = 0 ; i < 1 ; i++) {
    // for(unsigned long i = 0 ; i < 7 ; i++) {

    // section2 = clock();

    // for(unsigned long i = 0 ; i < h_graph_prop->total_edges ; i++) {

    //     // printf("%lu and %lu\n", h_source[i], h_destination[i]);
    //     h_source_degree[h_source[i] - 1]++;

    // }

    // unsigned long graph_choice;
    // std::cout << std::endl << "Enter input type" << std::endl << "1. Real Graph" << std::endl << "2. Random Graph" << std::endl;
    // std::cin >> graph_choice;

    // test code for actual benchmark start

    // goto gv_end;

    total_edge_blocks_count_batch = 0;

    std::cout << "Sizes of h_csr_edges, d_csr_edges, edge_size, batch_size are " << h_csr_edges_new.size() << ", " << d_csr_edges_new.size() << ", " << edge_size << ", " << batch_size << std::endl;

    // generate_csr_batch(vertex_size, edge_size, max_degree, h_source, h_destination, h_csr_offset_new, h_csr_edges_new, h_source_degrees_new, h_edge_blocks_count, h_prefix_sum_edge_blocks_new, h_batch_update_data, edge_size, total_batches, i, space_remaining, &total_edge_blocks_count_batch, &init_time);
    thrust::exclusive_scan(h_source_degrees.begin(), h_source_degrees.begin() + vertex_size + 1, h_csr_offset_new.begin());
    // for(unsigned long i = 0 ; i < vertex_size ; i++)
    //     h_csr_offset_new[i + 1] = h_csr_offset_new[i] + h_source_degrees[i];

    cudaDeviceSynchronize();

    generate_csr_batch_tester(vertex_size, edge_size, max_degree, h_source, h_destination, h_csr_offset_new, h_csr_edges_new, h_source_degrees_new, h_edge_blocks_count, h_prefix_sum_edge_blocks_new, h_batch_update_data, edge_size, total_batches, 0, space_remaining, &total_edge_blocks_count_batch, &init_time);
    // unsigned long thread_divider = 10;
    // thread_blocks = ceil((double(vertex_size / thread_divider)) / THREADS_PER_BLOCK);
    // device_coo_to_csr<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, d_source_pointer, d_destination_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, edge_size, d_source_degrees_new_pointer, thread_divider);

    thrust::copy(h_csr_offset_new.begin(), h_csr_offset_new.end(), d_csr_offset_new.begin());
    thrust::copy(h_csr_edges_new.begin(), h_csr_edges_new.end(), d_csr_edges_new.begin());
    thrust::copy(h_source_degrees.begin(), h_source_degrees.end(), d_source_degrees_new.begin());
    // thrust::copy(h_destination.begin(), h_destination.begin() + edge_size, d_csr_edges_new.begin());
    cudaDeviceSynchronize();


    // for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //     std::thread sort_thread(csr_sort, i, vertex_size, thrust::raw_pointer_cast(h_csr_offset_new.data()), thrust::raw_pointer_cast(h_csr_edges_new.data()));

    //     sort_thread.join();
    // }

    // for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //     std::thread remove_duplicates_thread(csr_remove_duplicates, i, vertex_size, thrust::raw_pointer_cast(h_csr_offset_new.data()), thrust::raw_pointer_cast(h_csr_edges_new.data()), thrust::raw_pointer_cast(h_source_degrees_new.data()));

    //     remove_duplicates_thread.join();
    // }

    // reconstruct_deduplicated_csr(vertex_size, edge_size, thrust::raw_pointer_cast(h_csr_offset_new.data()), thrust::raw_pointer_cast(h_csr_edges_new.data()), thrust::raw_pointer_cast(h_source_degrees_new.data()));
    // for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //     std::thread h_i_p(host_insert_preprocessing, i, vertex_size, thrust::raw_pointer_cast(h_source_degrees_new.data()), thrust::raw_pointer_cast(space_remaining.data()), 0, thrust::raw_pointer_cast(h_edge_blocks_count.data()));

    //     h_i_p.join();
    // }



    thread_blocks = ceil(double(edge_size) / THREADS_PER_BLOCK);
    device_remove_batch_duplicates<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, edge_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);
    cudaDeviceSynchronize();
    thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
    device_update_source_degrees<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);


    // thrust::exclusive_scan(d_source_degrees_new.begin(), d_source_degrees_new.begin() + vertex_size + 1, d_csr_offset_new.begin());
    // cudaDeviceSynchronize();
    thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
    device_insert_preprocessing<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, edge_size, d_source_degrees_new_pointer, 0, d_edge_blocks_count_pointer);
    cudaDeviceSynchronize();
    std::cout << "Checkpoint, vertex_size is " << vertex_size << ", size of d_edge_blocks_count is " << d_edge_blocks_count.size() << ", size of d_prefix_sum_edge_blocks_new is " << d_prefix_sum_edge_blocks_new.size() << std::endl;



    // tester_exclusive_scan<<<1, 1>>>(vertex_size, d_edge_blocks_count_pointer, d_prefix_sum_edge_blocks_new_pointer);

    // thrust::exclusive_scan(d_edge_blocks_count.begin(), d_edge_blocks_count.begin() + vertex_size, d_prefix_sum_edge_blocks_new.begin());
    thrust::exclusive_scan(d_edge_blocks_count.begin(), d_edge_blocks_count.begin() + vertex_size + 1, d_prefix_sum_edge_blocks_new.begin());
    // device_prefix_sum_calculation<<<1, 1>>>(d_edge_blocks_count_pointer, vertex_size, d_prefix_sum_edge_blocks_new_pointer);
    cudaDeviceSynchronize();

    cudaMemcpy(&total_edge_blocks_count_batch, d_prefix_sum_edge_blocks_new_pointer + vertex_size, sizeof(unsigned long),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // cudaMemcpy(d_batch_update_data, h_batch_update_data, (edge_size) * sizeof(unsigned long), cudaMemcpyHostToDevice);




    unsigned long start_index = 0;
    unsigned long end_index = edge_size;

    unsigned long remaining_edges = edge_size - start_index;

    // if(remaining_edges <= batch_size)
    //     end_index = edge_size;
    // else
    //     end_index = (i + 1) * batch_size;

    unsigned long current_batch = end_index - start_index;
    // std::cout << "Start index is " << start_index << ", End index is " << end_index << ", Current batch is " << current_batch << std::endl;

    // std::cout << "Current batch is " << current_batch << std::endl;

    // cudaDeviceSynchronize();
    // vd_time = clock() - vd_time;

    // std::cout << "Thread blocks edge init = " << thread_blocks << std::endl;

    // sleep(5);

    // adjacency_list_init<<< thread_blocks, THREADS_PER_BLOCK>>>(depl, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size);
    // al_time = clock();
    // adjacency_list_init<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks);
    // adjacency_list_init_modded<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks, device_vertex_dictionary);
    // adjacency_list_init_modded_v1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, psvd, thread_blocks, device_vertex_dictionary);
    // adjacency_list_init_modded_v2<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, psvd, thread_blocks, device_vertex_dictionary, i, current_batch);
    // adjacency_list_init_modded_v3<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_pointer, d_csr_edges_pointer);

    // adjacency_list_init_modded_v4<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_pointer, d_csr_edges_pointer);

    // std::cout << "Checkpoint 1" << std::endl;
    // temp_time = clock();

    // v5 code below
    std::cout << "Bulk build real graph now" << std::endl;
    unsigned long insert_load_factor_VC = 2;
    temp_time = clock();

    // thread_blocks = ceil(double(h_graph_prop->xDim) / THREADS_PER_BLOCK);
    // adjacency_list_init_modded_v5<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, device_vertex_dictionary, 0, current_batch, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);

    thread_blocks = ceil(double(edge_size) / THREADS_PER_BLOCK);
    batched_edge_inserts_EC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, device_vertex_dictionary, 0, edge_size, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);
    // batched_edge_inserts_EC_postprocessing<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer, d_prefix_sum_edge_blocks_new_pointer);

    update_edge_queue<<< 1, 1>>>(total_edge_blocks_count_batch);



    // thread_blocks = ceil(double(vertex_size) / insert_load_factor_VC / THREADS_PER_BLOCK);
    // adjacency_list_kernel_VC_LB<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, device_vertex_dictionary, 0, current_batch, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer, insert_load_factor_VC);
    // update_edge_queue<<< 1, 1>>>(total_edge_blocks_count_batch);

    // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
    // batched_edge_inserts_preprocessing_v6<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_vector_pointer);

    // thread_blocks = ceil(double(vertex_size + h_prefix_sum_edge_blocks_new[vertex_size]) / THREADS_PER_BLOCK);
    // batched_edge_inserts_v6<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_vector_pointer);

    // update_edge_queue<<< 1, 1>>>(total_edge_blocks_count_batch);
    // std::cout << std::endl << "After bulk-build" << std::endl << std::endl;
    // printKernelmodded_v2<<< 1, 1>>>(device_vertex_dictionary, vertex_size);
    cudaDeviceSynchronize();
    temp_time = clock() - temp_time;
    init_time += temp_time;
    // test code for actual benchmark end

    // std::cout << "Checkpoint" << std::endl;

    // sleep(5);

    std::cout << "Enter type of insertion required" << std::endl << "1. Regular batched edge insertion" << std::endl << "2. Edge Insert and Delete performance benchmark" << std::endl << "3. Vertex Insert and Delete performance benchmark" << std::endl;
    std::cin >> choice;

    // choice = 1;
    unsigned long kk = 0;

    // below if is taken for performance benchmark
    // if(choice == 2) {
    if(choice == 2) {


        unsigned long average_degree_batch, max_degree_batch, sum_degree_batch;

        // incremental here start

        // for(unsigned long kk = 1 ; kk < 2 ; kk++) {
        // kk is batch_number

        insert_processing:
        kk++;

        std::cout << std::endl << "Iteration " << kk << std::endl;

        // thrust::copy(h_csr_offset_new.begin(), h_csr_offset_new.end(), d_csr_offset_new.begin());
        // thrust::copy(h_csr_edges_new.begin(), h_csr_edges_new.end(), d_csr_edges_new.begin());

        // std::cout << std::endl << "Iteration " << i << std::endl;

        total_edge_blocks_count_batch = 0;


        // if(graph_choice == 1)
        // generate_csr_batch(vertex_size, edge_size, h_source, h_destination, h_csr_offset_new, h_csr_edges_new, h_source_degrees_new, h_edge_blocks_count, h_prefix_sum_edge_blocks_new, h_batch_update_data, BATCH_SIZE, total_batches, i, space_remaining, &total_edge_blocks_count_batch, &init_time);
        // else {

        // generate_random_batch(vertex_size, BATCH_SIZE, h_source, h_destination, h_source_degrees_new);
        // std::cout << "Generated random batch" << std::endl;

        thrust::fill(h_source_degrees.begin(), h_source_degrees.begin() + vertex_size, 0);

        generate_random_batch(h_graph_prop->xDim, BATCH_SIZE, h_source, h_destination, h_source_degrees_new, h_source_degrees, kk);
        std::cout << "Generated random batch" << std::endl;
        cudaDeviceSynchronize();
        // generate_csr_batch(vertex_size, edge_size, max_degree, h_source, h_destination, h_csr_offset_new, h_csr_edges_new, h_source_degrees_new, h_edge_blocks_count, h_prefix_sum_edge_blocks_new, h_batch_update_data, BATCH_SIZE, total_batches, kk, space_remaining, &total_edge_blocks_count_batch, &init_time);
        thrust::exclusive_scan(h_source_degrees.begin(), h_source_degrees.begin() + vertex_size + 1, h_csr_offset_new.begin());
        cudaDeviceSynchronize();
        generate_csr_batch_tester(vertex_size, edge_size, max_degree, h_source, h_destination, h_csr_offset_new, h_csr_edges_new, h_source_degrees_new, h_edge_blocks_count, h_prefix_sum_edge_blocks_new, h_batch_update_data, batch_size, total_batches, kk, space_remaining, &total_edge_blocks_count_batch, &init_time);
        std::cout << "Generated CSR of random batch" << std::endl;
        cudaDeviceSynchronize();
        // unsigned long thread_divider = 10;
        // thread_blocks = ceil((double(vertex_size / thread_divider)) / THREADS_PER_BLOCK);
        // device_coo_to_csr<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, d_source_pointer, d_destination_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, edge_size, d_source_degrees_new_pointer, thread_divider);
        thrust::copy(h_csr_offset_new.begin(), h_csr_offset_new.end(), d_csr_offset_new.begin());
        thrust::copy(h_csr_edges_new.begin(), h_csr_edges_new.end(), d_csr_edges_new.begin());
        thrust::copy(h_source_degrees.begin(), h_source_degrees.end(), d_source_degrees_new.begin());
        // thrust::copy(h_destination.begin(), h_destination.begin() + edge_size, d_csr_edges_new.begin());
        cudaDeviceSynchronize();


        // for(unsigned long i = 0 ; i < vertex_size ; i++) {
        //     std::thread sort_thread(csr_sort, i, vertex_size, thrust::raw_pointer_cast(h_csr_offset_new.data()), thrust::raw_pointer_cast(h_csr_edges_new.data()));

        //     sort_thread.join();
        // }

        // for(unsigned long i = 0 ; i < vertex_size ; i++) {
        //     std::thread remove_duplicates_thread(csr_remove_duplicates, i, vertex_size, thrust::raw_pointer_cast(h_csr_offset_new.data()), thrust::raw_pointer_cast(h_csr_edges_new.data()), thrust::raw_pointer_cast(h_source_degrees_new.data()));

        //     remove_duplicates_thread.join();
        // }

        // reconstruct_deduplicated_csr(vertex_size, edge_size, thrust::raw_pointer_cast(h_csr_offset_new.data()), thrust::raw_pointer_cast(h_csr_edges_new.data()), thrust::raw_pointer_cast(h_source_degrees_new.data()));
        // for(unsigned long i = 0 ; i < vertex_size ; i++) {
        //     std::thread h_i_p(host_insert_preprocessing, i, vertex_size, thrust::raw_pointer_cast(h_source_degrees_new.data()), thrust::raw_pointer_cast(space_remaining.data()), 0, thrust::raw_pointer_cast(h_edge_blocks_count.data()));

        //     h_i_p.join();
        // }

        clock_t deduplication_time;

        thread_blocks = ceil(double(batch_size) / THREADS_PER_BLOCK);
        device_remove_batch_duplicates<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);
        cudaDeviceSynchronize();
        std::cout << "Removed intra-batch duplicates" << std::endl;
        // thrust::exclusive_scan(d_source_degrees_new.begin(), d_source_degrees_new.begin() + vertex_size + 1, d_csr_offset_new.begin());
        // cudaDeviceSynchronize();
        thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
        device_update_source_degrees<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);

        deduplication_time = clock();


        if((EDGE_BLOCK_SIZE > 35) && (BATCH_SIZE <= 10000000)) {

            unsigned long thread_count_EBC;

            // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
            // // clock_t preprocessing, kernel_delete;
            // // preprocessing = clock();
            // batched_delete_preprocessing_edge_block_centric_v4<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer, d_source_vector_pointer);
            // cudaDeviceSynchronize();
            // thrust::exclusive_scan(d_source_vector.begin(), d_source_vector.begin() + vertex_size + 1, d_source_vector.begin());
            // cudaDeviceSynchronize();

            // cudaMemcpy(&thread_count_EBC, d_source_vector_pointer + vertex_size, sizeof(unsigned long),cudaMemcpyDeviceToHost);

            // // new edge_block_centric deletes
            // thread_blocks = ceil(double(thread_count_EBC) / THREADS_PER_BLOCK);
            // // kernel_delete = clock();
            // batched_delete_kernel_edge_block_centric_v4<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);
            // cudaDeviceSynchronize();

            thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
            // clock_t preprocessing, kernel_delete;
            // preprocessing = clock();
            batched_delete_preprocessing_EC_LD<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer, d_source_vector_pointer);
            cudaDeviceSynchronize();

            thrust::exclusive_scan(d_source_vector.begin(), d_source_vector.begin() + vertex_size + 1, d_source_vector.begin());
            cudaDeviceSynchronize();
            cudaMemcpy(&thread_count_EBC, d_source_vector_pointer + vertex_size, sizeof(unsigned long),cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            // std::cout << "Threads launched is " << thread_count_EBC << std::endl;
            thread_blocks = ceil(double(thread_count_EBC) / THREADS_PER_BLOCK);
            batched_delete_kernel_EC_LD<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);

            // unsigned long thread_multiplier = EDGE_BLOCK_SIZE / 10;
            // thread_blocks = ceil(double(thread_count_EBC) * thread_multiplier / THREADS_PER_BLOCK);
            // batched_delete_kernel_EC_LD_LB<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, thread_multiplier);


        }

        else {

            // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
            // batched_delete_kernel_VC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);

            // unsigned long vertex_LB_factor = 5;
            // thread_blocks = ceil(double(vertex_size) / vertex_LB_factor / THREADS_PER_BLOCK);
            // batched_delete_kernel_VC_LB<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, vertex_LB_factor);

            thread_blocks = ceil(double(batch_size) / THREADS_PER_BLOCK);
            batched_delete_kernel_EC_HD<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);

        }

        // std::cout << std::endl << "After batch-insert deduplication" << std::endl << std::endl;
        // printKernelmodded_v2<<< 1, 1>>>(device_vertex_dictionary, vertex_size);


        // device_inter_batch_deduplication_preprocessing_VC<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);
        cudaDeviceSynchronize();
        deduplication_time = clock() - deduplication_time;
        std::cout << "Removed inter-batch duplicates" << std::endl;

        thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
        device_insert_preprocessing<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, batch_size, d_source_degrees_new_pointer, 1, d_edge_blocks_count_pointer);
        cudaDeviceSynchronize();
        // thrust::exclusive_scan(d_edge_blocks_count.begin(), d_edge_blocks_count.begin() + vertex_size, d_prefix_sum_edge_blocks_new.begin());
        thrust::exclusive_scan(d_edge_blocks_count.begin(), d_edge_blocks_count.begin() + vertex_size + 1, d_prefix_sum_edge_blocks_new.begin());
        cudaDeviceSynchronize();

        cudaMemcpy(&total_edge_blocks_count_batch, d_prefix_sum_edge_blocks_new_pointer + vertex_size, sizeof(unsigned long),cudaMemcpyDeviceToHost);


        thrust::copy(d_source_degrees_new.begin(), d_source_degrees_new.end(), h_source_degrees_new.begin());
        cudaDeviceSynchronize();
        // for(unsigned long i = 0 ; i < vertex_size ; i++) {
        //     std::thread h_i_p(host_insert_preprocessing, i, vertex_size, thrust::raw_pointer_cast(h_source_degrees_new.data()), thrust::raw_pointer_cast(space_remaining.data()), 0, thrust::raw_pointer_cast(h_edge_blocks_count.data()));

        //     h_i_p.join();
        // }
        // thrust::exclusive_scan(h_edge_blocks_count.begin(), h_edge_blocks_count.begin() + vertex_size + 1, h_prefix_sum_edge_blocks_new.begin());

        // calculating max and avg degree of batch
        max_degree_batch = h_source_degrees_new[0];
        sum_degree_batch = h_source_degrees_new[0];
        for(unsigned long j = 1 ; j < vertex_size ; j++) {

            if(h_source_degrees_new[j] > max_degree_batch)
                max_degree_batch = h_source_degrees_new[j];
            sum_degree_batch += h_source_degrees_new[j];
        }

        average_degree_batch = sum_degree_batch / vertex_size ;

        std::cout << std::endl << "Max degree of batch is " << max_degree_batch << std::endl;
        std::cout << "Average degree of batch is " << sum_degree_batch / vertex_size << std::endl << std::endl;
        std::cout << "Deduplication time (inter-batch) : " <<  (float)deduplication_time/CLOCKS_PER_SEC << " seconds" << std::endl;

        // }
        // std::cout << "Total edge blocks is " << total_edge_blocks_count_batch  << std::endl;

        // std::cout << std::endl << "Printing batched CSR" << std::endl << "Source degrees\t\t" << std::endl;
        // for(unsigned long j = 0 ; j < vertex_size ; j++) {
        //     std::cout << h_source_degrees_new[j] << " ";
        //     if((j + 1) % vertex_size == 0)
        //         std::cout << std::endl;
        // }
        // std::cout << std::endl << "CSR offset\t\t" << std::endl;
        // for(unsigned long j = 0 ; j < (vertex_size + 1) ; j++) {
        //     std::cout << h_csr_offset_new[j] << " ";
        //     if(((j + 1) % (vertex_size + 1)) == 0)
        //         std::cout << std::endl;
        // }
        // std::cout << std::endl << "CSR edges\t\t" << std::endl;
        // for(unsigned long j = 0 ; j < batch_size ; j++) {
        //     std::cout << h_csr_edges_new[j] << " ";
        //     if((j + 1) % batch_size == 0)
        //         std::cout << std::endl;
        // }
        // std::cout << std::endl << "Edge blocks count\t\t" << std::endl;
        // for(unsigned long j = 0 ; j < vertex_size ; j++) {
        //     std::cout << h_edge_blocks_count[j] << " ";
        //     if((j + 1) % vertex_size == 0)
        //         std::cout << std::endl;
        // }
        // std::cout << std::endl << "Prefix sum edge blocks\t\t" << std::endl;
        // for(unsigned long j = 0 ; j < vertex_size ; j++) {
        //     std::cout << h_prefix_sum_edge_blocks_new[j] << " ";
        //     if((j + 1) % vertex_size == 0)
        //         std::cout << std::endl;
        // }
        // std::cout << std::endl << "Space remaining\t\t" << std::endl;
        // for(unsigned long j = 0 ; j < vertex_size ; j++)
        //     std::cout << space_remaining[j] << " ";
        // std::cout << std::endl;
        // std::cout << std::endl << "Batch update vector\t\t" << std::endl;
        // for(unsigned long j = 0 ; j < vertex_size + 1 + batch_size + vertex_size ; j++)
        //     std::cout << h_batch_update_data[j] << " ";
        // std::cout << std::endl;
        // std::cout << std::endl << std::endl;

        // temp_time = clock();


        // thrust::copy(h_source_degrees_new.begin(), h_source_degrees_new.end(), d_source_degrees_new.begin());
        // thrust::copy(h_csr_offset_new.begin(), h_csr_offset_new.end(), d_csr_offset_new.begin());
        // thrust::copy(h_csr_edges_new.begin(), h_csr_edges_new.end(), d_csr_edges_new.begin());
        // thrust::copy(h_edge_blocks_count.begin(), h_edge_blocks_count.end(), d_edge_blocks_count.begin());
        // thrust::copy(h_prefix_sum_edge_blocks_new.begin(), h_prefix_sum_edge_blocks_new.end(), d_prefix_sum_edge_blocks_new.begin());
        // thrust::copy(h_source.begin(), h_source.begin() + BATCH_SIZE, d_source.begin());
        // thrust::copy(h_destination.begin(), h_destination.begin() + BATCH_SIZE, d_destination.begin());

        temp_time = clock();

        // thrust::copy(h_batch_update_data.begin(), h_batch_update_data.end(), d_batch_update_data.begin());
        // cudaMemcpy(d_batch_update_data, &h_batch_update_data, (vertex_size + 1 + batch_size + vertex_size) * sizeof(unsigned long), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_batch_update_data, h_batch_update_data, (batch_size) * sizeof(unsigned long), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_batch_update_data, h_batch_update_data, (edge_size) * sizeof(unsigned long), cudaMemcpyHostToDevice);

        // thrust::device_vector <unsigned long> d_source_degrees_new(vertex_size);
        // thrust::device_vector <unsigned long> d_csr_offset_new(vertex_size + 1);
        // thrust::device_vector <unsigned long> d_csr_edges_new(batch_size);
        // thrust::device_vector <unsigned long> d_edge_blocks_count_new(vertex_size);
        // thrust::device_vector <unsigned long> d_prefix_sum_edge_blocks_new(vertex_size);

        // temp_time = clock();

        start_index = 0;

        remaining_edges = edge_size - start_index;

        if(remaining_edges <= batch_size)
            end_index = edge_size;
        else
            end_index = batch_size;

        current_batch = end_index - start_index;

        std::cout << "Current batch is " << current_batch << std::endl;

        // cudaDeviceSynchronize();
        // vd_time = clock() - vd_time;
        std::cout << "Checkpoint" << std::endl;

        al_time = clock();

        thread_blocks = ceil(double(h_graph_prop->xDim) / THREADS_PER_BLOCK);
        // std::cout << "Thread blocks edge init = " << thread_blocks << std::endl;

        // sleep(5);

        // adjacency_list_init<<< thread_blocks, THREADS_PER_BLOCK>>>(depl, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size);
        // al_time = clock();
        // adjacency_list_init<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks);
        // adjacency_list_init_modded<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks, device_vertex_dictionary);
        // adjacency_list_init_modded_v1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, psvd, thread_blocks, device_vertex_dictionary);
        // adjacency_list_init_modded_v2<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, psvd, thread_blocks, device_vertex_dictionary, i, current_batch);
        // adjacency_list_init_modded_v3<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_pointer, d_csr_edges_pointer);

        // adjacency_list_init_modded_v4<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_pointer, d_csr_edges_pointer);

        // std::cout << "Checkpoint 1" << std::endl;
        // temp_time = clock();

        // adjacency_list_init_modded_v5<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, device_vertex_dictionary, kk, current_batch, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);

        thread_blocks = ceil(double(batch_size) / THREADS_PER_BLOCK);
        batched_edge_inserts_EC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, device_vertex_dictionary, kk, batch_size, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);
        batched_edge_inserts_EC_postprocessing<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer, d_prefix_sum_edge_blocks_new_pointer);

        update_edge_queue<<< 1, 1>>>(total_edge_blocks_count_batch);

        // thread_blocks = ceil(double(vertex_size) / insert_load_factor_VC / THREADS_PER_BLOCK);
        // adjacency_list_kernel_VC_LB<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, device_vertex_dictionary, 0, current_batch, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer, insert_load_factor_VC);
        // update_edge_queue<<< 1, 1>>>(total_edge_blocks_count_batch);

        cudaDeviceSynchronize();
        std::cout << "Batched insert done" << std::endl;

        // std::cout << std::endl << "After batch-insert" << std::endl << std::endl;
        // printKernelmodded_v2<<< 1, 1>>>(device_vertex_dictionary, vertex_size);

        // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
        // batched_edge_inserts_preprocessing_v6<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_vector_pointer);

        // thread_blocks = ceil(double(vertex_size + h_prefix_sum_edge_blocks_new[vertex_size]) / THREADS_PER_BLOCK);
        // batched_edge_inserts_v6<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_vector_pointer);
        // update_edge_queue<<< 1, 1>>>(total_edge_blocks_count_batch);


        cudaDeviceSynchronize();
        memory_usage();
        al_time = clock() - al_time;
        // temp_time = clock() - temp_time;
        // al_time += deduplication_time;

        std::cout << "Insert done" << std::endl;
        std::cout << "Insert time   : " << (float)al_time/CLOCKS_PER_SEC << " seconds" << std::endl;
        fullTime += ((double) al_time/CLOCKS_PER_SEC * 1000);


        cudaDeviceSynchronize();
        // goto main_menu;

        // }

        // goto end_for_now;

        // incremental end here

        // goto end_for_now;

        // cbt_stats<<< 1, 1>>>(device_vertex_dictionary, vertex_size);

        // if(i != 7) {
        //     std::cout << "Hit here" << std::endl;
        // }
        // temp_time = clock() - temp_time;

        // d_csr_edges_new.resize(1);
        // d_csr_offset_new.resize(1);

        // d_csr_offset_new.clear();
        // device_vector<T>().swap(d_csr_offset_new);
        // d_csr_edges_new.clear();
        // device_vector<T>().swap(d_csr_edges_new);
        // d_csr_offset_new.shrink_to_fit();
        // d_csr_edges_new.shrink_to_fit();

        // std::cout << "Checkpoint delete" << std::endl;

        delete_processing:

        // uncomment below section for edge-centric deletes
        thrust::device_vector <unsigned long> d_source(1);
        thrust::device_vector <unsigned long> d_destination(1);
        unsigned long* d_source_pointer = thrust::raw_pointer_cast(d_source.data());
        unsigned long* d_destination_pointer = thrust::raw_pointer_cast(d_destination.data());
        cudaDeviceSynchronize();
        // delete_time = clock();
        // thrust::copy(h_source.begin(), h_source.begin() + BATCH_SIZE, d_source.begin());
        // thrust::copy(h_destination.begin(), h_destination.begin() + BATCH_SIZE, d_destination.begin());

        unsigned long h_data_structure_edge_block_count;
        cudaDeviceSynchronize();

        thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
        device_prefix_sum_calculation_preprocessing<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_prefix_sum_edge_blocks_new_pointer);
        cudaDeviceSynchronize();
        thrust::exclusive_scan(thrust::device, d_prefix_sum_edge_blocks_new_pointer, d_prefix_sum_edge_blocks_new_pointer + vertex_size + 1, d_prefix_sum_edge_blocks_new_pointer);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_data_structure_edge_block_count, d_prefix_sum_edge_blocks_new_pointer + vertex_size, sizeof(unsigned long),cudaMemcpyDeviceToHost);


        // std::cout << "Checkpoint delete" << std::endl;
        unsigned long* d_thread_count_vector_pointer = thrust::raw_pointer_cast(d_thread_count_vector.data());

        unsigned long h_total_threads, *d_total_threads;
        d_total_threads = d_thread_count_vector_pointer + vertex_size;



        delete_time = clock();
        // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
        // batched_delete_preprocessing<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer);
        // thread_blocks = ceil(double(total_edge_blocks_count_batch) / THREADS_PER_BLOCK);
        // batched_delete_kernel_v1<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, total_edge_blocks_count_batch, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer);

        unsigned long decision_tree_true = (average_degree_batch < (max_degree_batch / 20)) && (BATCH_SIZE <= 10000000);

        // if((decision_tree_true)) {
        if(0) {

            thrust::copy(h_source.begin(), h_source.begin() + BATCH_SIZE, d_source.begin());
            thrust::copy(h_destination.begin(), h_destination.begin() + BATCH_SIZE, d_destination.begin());

            // Below is the code for edge-centric batch deletes
            thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
            batched_delete_preprocessing_edge_centric<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer);
            thread_blocks = ceil(double(batch_size) / THREADS_PER_BLOCK);
            batched_delete_kernel_edge_centric<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_pointer, d_destination_pointer);

        }

            // Below is the test code for the new bit string based parallel edge block approach

            // remove this stupid else if and make it else later
        else {

            // cudaMemcpy(d_batch_update_data, h_batch_update_data, (batch_size) * sizeof(unsigned long), cudaMemcpyHostToDevice);

            // // launch with number of threads equal to the vertices in the graph
            // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
            // // clock_t preprocessing, kernel_delete;
            // // preprocessing = clock();
            // batched_delete_preprocessing_edge_block_centric<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer, d_source_vector_pointer);
            // // cudaDeviceSynchronize();
            // // preprocessing = clock() - preprocessing;

            // // launch with number of threads equal to the edge blocks used by the data structure
            // // thread_blocks = ceil(double(h_prefix_sum_edge_blocks_new[vertex_size]) / THREADS_PER_BLOCK);
            // thread_blocks = ceil(double(h_data_structure_edge_block_count) / THREADS_PER_BLOCK);
            // // kernel_delete = clock();
            // batched_delete_kernel_edge_block_centric<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);
            // // cudaDeviceSynchronize();
            // // kernel_delete = clock() - kernel_delete;

            // if(EDGE_BLOCK_SIZE > 100) {
            if((EDGE_BLOCK_SIZE > 35) && (BATCH_SIZE <= 10000000)) {
                unsigned long thread_count_EBC;

                // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
                // // clock_t preprocessing, kernel_delete;
                // // preprocessing = clock();
                // batched_delete_preprocessing_edge_block_centric_v4<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer, d_source_vector_pointer);
                // cudaDeviceSynchronize();
                // thrust::exclusive_scan(d_source_vector.begin(), d_source_vector.begin() + vertex_size + 1, d_source_vector.begin());
                // cudaDeviceSynchronize();

                // cudaMemcpy(&thread_count_EBC, d_source_vector_pointer + vertex_size, sizeof(unsigned long),cudaMemcpyDeviceToHost);

                // // new edge_block_centric deletes
                // thread_blocks = ceil(double(thread_count_EBC) / THREADS_PER_BLOCK);
                // // kernel_delete = clock();
                // batched_delete_kernel_edge_block_centric_v4<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);

                thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
                // clock_t preprocessing, kernel_delete;
                // preprocessing = clock();
                batched_delete_preprocessing_EC_LD<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer, d_source_vector_pointer);
                cudaDeviceSynchronize();

                thrust::exclusive_scan(d_source_vector.begin(), d_source_vector.begin() + vertex_size + 1, d_source_vector.begin());
                cudaDeviceSynchronize();
                cudaMemcpy(&thread_count_EBC, d_source_vector_pointer + vertex_size, sizeof(unsigned long),cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                // std::cout << "Threads launched is " << thread_count_EBC << std::endl;
                thread_blocks = ceil(double(thread_count_EBC) / THREADS_PER_BLOCK);
                batched_delete_kernel_EC_LD<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);
                // unsigned long thread_multiplier = EDGE_BLOCK_SIZE / 10;
                // thread_blocks = ceil(double(thread_count_EBC) * thread_multiplier / THREADS_PER_BLOCK);
                // batched_delete_kernel_EC_LD_LB<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, thread_multiplier);

            }

            else {

                // basic vertex_centric
                // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
                // batched_delete_kernel_VC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);

                // unsigned long vertex_LB_factor = 5;
                // thread_blocks = ceil(double(vertex_size) / vertex_LB_factor / THREADS_PER_BLOCK);
                // batched_delete_kernel_VC_LB<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, vertex_LB_factor);

                thread_blocks = ceil(double(batch_size) / THREADS_PER_BLOCK);
                batched_delete_kernel_EC_HD<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);


                // load-balanced vertex_centric
                // unsigned long thread_count_VC_LB;
                // unsigned long thread_divider = 1;
                // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
                // batched_delete_preprocessing_LB<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer, d_source_vector_pointer);
                // cudaDeviceSynchronize();
                // thrust::exclusive_scan(d_source_vector.begin(), d_source_vector.begin() + vertex_size + 1, d_source_vector.begin());
                // cudaDeviceSynchronize();
                // cudaMemcpy(&thread_count_VC_LB, d_source_vector_pointer + vertex_size, sizeof(unsigned long),cudaMemcpyDeviceToHost);
                // cudaDeviceSynchronize();
                // std::cout << "Thread count for VC_LB is " << double(thread_count_VC_LB) / thread_divider << std::endl;
                // thread_blocks = ceil((double(thread_count_VC_LB) / thread_divider) / THREADS_PER_BLOCK);
                // // kernel_delete = clock();
                // // batched_delete_kernel_edge_block_centric_v4<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);
                // batched_delete_kernel_LB<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, thread_divider);


            }
        }




        // Below is the test code for the update of Approach #1
        // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
        // batched_delete_preprocessing_v3_1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_thread_count_vector_pointer);

        // cudaDeviceSynchronize();
        // // delete_time = clock() - delete_time;

        // batched_delete_preprocessing_v3_prefix_sum<<< 1, 1>>>(d_thread_count_vector_pointer, vertex_size);

        // // clock_t temp;
        // // temp = clock();
        // batched_delete_preprocessing_v3_2<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer, d_source_vector_pointer, d_thread_count_vector_pointer);

        // cudaMemcpy(&h_total_threads, d_total_threads, sizeof(unsigned long), cudaMemcpyDeviceToHost);
        // d_source_vector.resize(h_total_threads);
        // // std::cout << "Number of threads needed is " << h_total_threads << std::endl;

        // thread_blocks = ceil(double(h_total_threads) / THREADS_PER_BLOCK);
        // batched_delete_kernel_edge_block_centric_v3<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_thread_count_vector_pointer);

        // cudaDeviceSynchronize();
        // temp = clock() - temp;
        // delete_time += temp;

        // thrust::exclusive_scan(d_thread_count_vector_pointer, d_thread_count_vector_pointer + vertex_size, d_thread_count_vector_pointer);

        // Below is the test code for Approach #2, where only the adjacencies that have deletes are targeted
        // unsigned long h_non_zero_vertices[vertex_size];
        // thrust::host_vector <unsigned long> h_non_zero_vertices(vertex_size);
        // thrust::device_vector <unsigned long> d_non_zero_vertices(vertex_size);
        // thrust::device_vector <unsigned long> d_index_counter(EDGE_PREALLOCATE_LIST_SIZE);

        // unsigned long* d_non_zero_vertices_pointer = thrust::raw_pointer_cast(d_non_zero_vertices.data());
        // unsigned long* d_index_counter_pointer = thrust::raw_pointer_cast(d_index_counter.data());

        // unsigned long h_non_zero_vertices_count = 0;

        // for(unsigned long i = 0 ; i < vertex_size ; i++) {

        //     if(h_csr_offset_new[i] != h_csr_offset_new[i + 1]) {

        //         h_non_zero_vertices[h_non_zero_vertices_count++] = i;

        //     }

        // }

        // // std::cout << std::endl << "Non-zero vertices count is " << h_non_zero_vertices_count << std::endl;
        // // for(unsigned long i = 0 ; i < h_non_zero_vertices_count ; i++)
        // //     std::cout << h_non_zero_vertices[i] << " ";
        // // std::cout << std::endl;

        // thrust::copy(h_non_zero_vertices.begin(), h_non_zero_vertices.end(), d_non_zero_vertices.begin());
        // thread_blocks = ceil(double(h_non_zero_vertices_count) / THREADS_PER_BLOCK);
        // batched_delete_preprocessing_edge_block_centric_v2<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer, d_source_vector_pointer, d_non_zero_vertices_pointer, d_index_counter_pointer, h_non_zero_vertices_count);

        // // // launch with number of threads equal to the edge blocks used by the data structure
        // // thread_blocks = ceil(double(h_prefix_sum_edge_blocks_new[vertex_size]) / THREADS_PER_BLOCK);
        // // batched_delete_kernel_edge_block_centric<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);



        // Below is the test code for parallelized edge-centric batch deletes
        // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
        // batched_delete_preprocessing<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer);
        // thread_blocks = ceil(double(batch_size * EDGE_BLOCK_SIZE) / THREADS_PER_BLOCK);
        // batched_delete_kernel_edge_centric_parallelized<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, batch_size * EDGE_BLOCK_SIZE, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_pointer, d_destination_pointer);


        // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
        // batched_delete_kernel<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer);
        cudaDeviceSynchronize();
        delete_time = clock() - delete_time;

        std::cout << "Delete done" << std::endl;
        std::cout << "Delete batch   : " << (float)delete_time/CLOCKS_PER_SEC << " seconds" << std::endl;
        fullTime += ((double) delete_time / CLOCKS_PER_SEC * 1000);

        // std::cout << "Delete Preprocessing     : " << (float)preprocessing/CLOCKS_PER_SEC << " seconds" << std::endl;
        // std::cout << "Delete Kernel            : " << (float)kernel_delete/CLOCKS_PER_SEC << " seconds" << std::endl;

        // std::cout << "Batch #" << i << " took " << (float)temp_time/CLOCKS_PER_SEC << " seconds" << std::endl;
        // Seperate kernel for updating queues due to performance issues for global barriers
        // printKernelmodded_v1<<< 1, 1>>>(device_vertex_dictionary, vertex_size);
        // if(i < 10)

        // std::cout << std::endl << "After batch-delete" << std::endl << std::endl;

        // printKernelmodded_v2<<< 1, 1>>>(device_vertex_dictionary, vertex_size);

        // std::cout << "Outside checkpoint" << std::endl;

        if(decision_tree_true)
            std::cout << "Edge-Centric deletes taken" << std::endl << std::endl;
        else
        if(EDGE_BLOCK_SIZE > 35)
            std::cout << "Edge-Block-Centric deletes taken" << std::endl << std::endl;
        else
            std::cout << "Vertex-Centric deletes taken" << std::endl << std::endl;
    }

    else if(choice == 3) {

        // Vertex dictionary benchmarking here
        // Insert code here
        clock_t vertex_insert_time, vertex_delete_time;
        unsigned long vertex_batch_size = VERTEX_BATCH_SIZE;

        thrust::host_vector <unsigned long> h_vertex_batch(vertex_batch_size);
        for(unsigned long l = 0 ; l < vertex_batch_size ; l++)
            h_vertex_batch[l] = vertex_size + l + 1;

        thrust::device_vector <unsigned long> d_vertex_batch(vertex_batch_size);
        unsigned long *d_vertex_batch_pointer = thrust::raw_pointer_cast(d_vertex_batch.data());
        thrust::copy(h_vertex_batch.begin(), h_vertex_batch.end(), d_vertex_batch.begin());

        cudaDeviceSynchronize();

        // if((vertex_size + vertex_batch_size) < vertex_dictionary_size) {

        //     // simple insert
        //     thread_blocks = ceil(double(vertex_batch_size) / THREADS_PER_BLOCK);
        //     vertex_insert_kernel<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, vertex_batch_size, d_vertex_batch_pointer);
        //     vertex_insert_kernel_postprocessing<<< 1, 1>>>(device_vertex_dictionary, vertex_batch_size);

        // }

        vertex_insert_time = clock();

        if((vertex_size + vertex_batch_size) > vertex_dictionary_size) {

            // resize every vector

            while(vertex_dictionary_size < (vertex_size + vertex_batch_size)) {
                vertex_dictionary_size *= 2;
                // std::cout << "New VD size is " << vertex_dictionary_size << std::endl;
            }
            // std::cout << "New VD size is " << vertex_dictionary_size << std::endl;

            // d_vertex_id.resize(vertex_dictionary_size);
            // d_edge_block_count_VD.resize(vertex_dictionary_size);
            // d_active_edge_count_VD.resize(vertex_dictionary_size);
            // d_last_insert_edge_offset.resize(vertex_dictionary_size);
            // d_last_insert_edge_block.resize(vertex_dictionary_size);
            // d_edge_block_address.resize(vertex_dictionary_size);

            // unsigned long *d_vertex_id, *d_edge_block_count_VD, *d_last_insert_edge_offset;
            // unsigned int *d_active_edge_count_VD;
            // struct edge_block **d_last_insert_edge_block, **d_edge_block_address;

            cudaMalloc(&d_vertex_id, 3 * vertex_dictionary_size * sizeof(unsigned long));
            d_edge_block_count_VD = d_vertex_id + vertex_dictionary_size;
            d_last_insert_edge_offset = d_vertex_id + (2 * vertex_dictionary_size);

            cudaMalloc(&d_active_edge_count_VD, vertex_dictionary_size * sizeof(unsigned int));

            cudaMalloc((struct edge_block **)&d_last_insert_edge_block, 2 * vertex_dictionary_size * sizeof(struct edge_block *));
            d_edge_block_address = d_last_insert_edge_block + vertex_dictionary_size;
            cudaDeviceSynchronize();

            thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
            copy_vertex_dictionary<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, d_vertex_id, d_edge_block_count_VD, d_active_edge_count_VD, d_last_insert_edge_offset, d_last_insert_edge_block, d_edge_block_address, vertex_size);

            // no reclamation now, memory leak is there
            copy_vertex_dictionary_postprocessing<<<1, 1>>>(device_vertex_dictionary, d_vertex_id, d_edge_block_count_VD, d_active_edge_count_VD, d_last_insert_edge_offset, d_last_insert_edge_block, d_edge_block_address, vertex_size);

            cudaDeviceSynchronize();

            // simple insert?

        }

        thread_blocks = ceil(double(vertex_batch_size) / THREADS_PER_BLOCK);
        vertex_insert_kernel<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, vertex_batch_size, d_vertex_batch_pointer);
        vertex_insert_kernel_postprocessing<<< 1, 1>>>(device_vertex_dictionary, vertex_batch_size);
        vertex_size += vertex_batch_size;

        cudaDeviceSynchronize();
        vertex_insert_time = clock() - vertex_insert_time;
        // Delete code here

        srand(1 + 1);
        for(unsigned long l = 0 ; l < vertex_batch_size ; l++) {
            h_vertex_batch[l] = (rand() % vertex_size) + 1;
            // std::cout << h_vertex_batch[l] << " ";
        }
        thrust::sort(thrust::host, h_vertex_batch.begin(), h_vertex_batch.end());
        cudaDeviceSynchronize();
        for(unsigned long l = 0 ; l < vertex_batch_size ; l++) {
            // h_vertex_batch[l] = (rand() % vertex_size) + 1;
            std::cout << h_vertex_batch[l] << " ";
        }

        std::cout << std::endl;
        thrust::copy(h_vertex_batch.begin(), h_vertex_batch.end(), d_vertex_batch.begin());
        thrust::device_vector <unsigned long> d_edge_block_count_free_vector(vertex_batch_size + 1);
        unsigned long *d_edge_block_count_free_vector_pointer = thrust::raw_pointer_cast(d_edge_block_count_free_vector.data());
        cudaDeviceSynchronize();
        // unsigned long freed_edge_block_count;

        vertex_delete_time = clock();
        if(DIRECTED) {
            thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
            vertex_delete_kernel_directed<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, vertex_batch_size, d_vertex_batch_pointer);
        }
        else
            vertex_delete_kernel_stage_1<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, vertex_batch_size, d_vertex_batch_pointer);

        thread_blocks = ceil(double(vertex_batch_size) / THREADS_PER_BLOCK);
        vertex_delete_kernel_stage_2_preprocessing<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, vertex_batch_size, d_vertex_batch_pointer, d_edge_block_count_free_vector_pointer);
        cudaDeviceSynchronize();
        thrust::exclusive_scan(d_edge_block_count_free_vector.begin(), d_edge_block_count_free_vector.begin() + vertex_batch_size + 1, d_edge_block_count_free_vector.begin());
        // cudaMemcpy(&freed_edge_block_count, d_edge_block_count_free_vector_pointer + vertex_batch_size + 1, sizeof(unsigned long),cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        vertex_delete_kernel_stage_2<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, vertex_batch_size, d_vertex_batch_pointer, d_edge_block_count_free_vector_pointer);
        vertex_delete_kernel_postprocessing<<< 1, 1>>>(device_vertex_dictionary, vertex_batch_size, d_edge_block_count_free_vector_pointer);
        // update_edge_queue<<< 1, 1>>>(freed_edge_block_count);
        // thread_blocks = ceil(double(vertex_batch_size * vertex_size) / THREADS_PER_BLOCK);
        // batched_delete_kernel_EC_HD_vertex<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_vertex_batch_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, vertex_batch_size);
        // vertex_size -= vertex_batch_size;
        cudaDeviceSynchronize();
        vertex_delete_time = clock() - vertex_delete_time;

        std::cout << "Vertex Inserts   : " << (float)vertex_insert_time/CLOCKS_PER_SEC << " seconds" << std::endl;
        std::cout << "Vertex Deletes   : " << (float)vertex_delete_time/CLOCKS_PER_SEC << " seconds" << std::endl;

    }

    // }
    // }

    end_for_now:

    // update_edge_queue<<< 1, 1>>>(total_edge_blocks_count_init);

    // al_time = clock() - al_time;
    time_req = clock() - time_req;

    // memory_usage();

    // printKernelmodded_v2<<< 1, 1>>>(device_vertex_dictionary, vertex_size);

    // cudaDeviceSynchronize();

    // // for(long i = 0 ; i < edge_size ; i++)
    // //     std::cout << h_source[i] << " and " << h_destination[i] << std::endl;

    // printf("\nCorrectness check of data structure\n");
    // printf("*---------------*\n");
    // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
    // // correctness_check_kernel<<< 1, 1>>>(device_vertex_dictionary, vertex_size, edge_size, c_source, c_destination);
    // unsigned long h_correctness_flag, *d_correctness_flag;
    // cudaMalloc(&d_correctness_flag, sizeof(unsigned long));
    // // correctness_check_kernel<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, edge_size, d_csr_offset_pointer, d_csr_edges_pointer, d_correctness_flag);
    // correctness_check_kernel_v1<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, edge_size, d_csr_offset_pointer, d_csr_edges_pointer, d_correctness_flag);
    // cudaMemcpy(&h_correctness_flag, d_correctness_flag, sizeof(unsigned long),cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // // search_times = clock() - search_times;

    // if(h_correctness_flag)
    //     std::cout << "Data structure corrupted" << std::endl;
    // else
    //     std::cout << "Data structure uncorrupted" << std::endl;
    // printf("*---------------*\n\n");


    // // sleep(5);

    // // printKernel<<< 1, 1>>>(device_vertex_block, vertex_size);
    // // printKernelmodded<<< 1, 1>>>(device_vertex_dictionary, vertex_size);
    // // printKernelmodded_v1<<< 1, 1>>>(device_vertex_dictionary, vertex_size);

    // cudaDeviceSynchronize();

    main_menu:

    unsigned long exitFlag = 1;
    unsigned long menuChoice;
    unsigned long h_search_flag, *d_search_flag;
    unsigned long search_source, search_destination, total_search_threads;
    cudaMalloc(&d_search_flag, sizeof(unsigned long));

    clock_t prefix_sum_time = 0;

    while(exitFlag) {

        std::cout << std::endl << "Please enter any of the below options" << std::endl << "1. Search for and edge" << std::endl << "2. Delete an edge" << std::endl << "3. Print Adjacency" << std::endl << "4. PageRank Calculation" << std::endl << "5. Static Traingle Counting" << std::endl << "6. Single-Source Shortest Path" << std::endl << "7. Exit" << std::endl << "8. Insert Batch" << std::endl << "9. Delete Batch" << std::endl << "10. Compaction Overhead" << std::endl << "11. Edge-Centric TC" << std::endl;
        scanf("%lu", &menuChoice);

        switch(menuChoice) {

            case 1  :
                std::cout << "Enter the source and destination vertices respectively" << std::endl;
                // unsigned long search_source, search_destination, total_search_threads;
                scanf("%lu %lu", &search_source, &search_destination);
                std::cout << "Edge blocks count for " << search_source << " is " << h_edge_blocks_count_init[search_source - 1] << std::endl;

                search_times = clock();

                total_search_threads = h_edge_blocks_count_init[search_source - 1] * EDGE_BLOCK_SIZE;
                thread_blocks = ceil(double(total_search_threads) / THREADS_PER_BLOCK);

                // search_edge_kernel<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, total_search_threads, search_source, search_destination, d_search_flag);
                search_pre_processing<<< 1, 1>>>(device_vertex_dictionary, search_source);
                search_edge_kernel_v1<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, total_search_threads, search_source, search_destination, d_search_flag);

                cudaMemcpy(&h_search_flag, d_search_flag, sizeof(unsigned long),cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                search_times = clock() - search_times;

                std::cout << "Search result was " << h_search_flag << " with time " << (float)search_times/CLOCKS_PER_SEC << " seconds" << std::endl;
                h_search_flag = 0;
                cudaMemcpy(d_search_flag, &h_search_flag, sizeof(unsigned long), cudaMemcpyHostToDevice);
                cudaDeviceSynchronize();
                break;

            case 2  :
                std::cout << "Enter the source and destination vertices respectively" << std::endl;
                // unsigned long search_source, search_destination, total_search_threads;
                scanf("%lu %lu", &search_source, &search_destination);
                // std::cout << "Edge blocks count for " << search_source << " is " << h_edge_blocks_count_init[search_source - 1] << std::endl;

                total_search_threads = h_edge_blocks_count_init[search_source - 1] * EDGE_BLOCK_SIZE;
                thread_blocks = ceil(double(total_search_threads) / THREADS_PER_BLOCK);

                // search_edge_kernel<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, total_search_threads, search_source, search_destination, d_search_flag);
                search_pre_processing<<< 1, 1>>>(device_vertex_dictionary, search_source);
                delete_edge_kernel<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, total_search_threads, search_source, search_destination, d_search_flag);

                search_pre_processing<<< 1, 1>>>(device_vertex_dictionary, search_destination);
                delete_edge_kernel<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, total_search_threads, search_destination, search_source, d_search_flag);

                cudaDeviceSynchronize();

                break;

            case 3  :

                printKernelmodded_v2<<< 1, 1>>>(device_vertex_dictionary, vertex_size);
                cudaDeviceSynchronize();
                break;

            case 4  :
            {
                // float *d_pageRankVector_1, *d_pageRankVector_2;



                thrust::device_vector <float> d_pageRankVector_1(vertex_size);
                thrust::device_vector <float> d_pageRankVector_2(vertex_size);

                float* d_pageRankVector_1_pointer = thrust::raw_pointer_cast(d_pageRankVector_1.data());
                float* d_pageRankVector_2_pointer = thrust::raw_pointer_cast(d_pageRankVector_2.data());

                pageRank_time = clock();

                thrust::fill(d_pageRankVector_1.begin(), d_pageRankVector_1.end(), 0.25f);

                // cudaMalloc(&d_pageRankVector_1, vertex_size * sizeof(float));
                // cudaMalloc(&d_pageRankVector_2, vertex_size * sizeof(float));
                // d_pageRankVector_2 = d_pageRankVector_1 + vertex_size;

                // pageRankInitialization<<< 1, 1>>>(d_pageRankVector_1, vertex_size);

                // cudaMemset(d_pageRankVector_1, 0.25f, vertex_size * sizeof(float));


                // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);

                // pageRank_kernel_preprocesssing<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_pageRankVector_1_pointer, d_pageRankVector_2_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer);

                thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
                pageRank_kernel_preprocessing<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);


                // launch with number of threads equal to the edge blocks used by the data structure
                thread_blocks = ceil(double(h_prefix_sum_edge_blocks_new[vertex_size]) / THREADS_PER_BLOCK);
                pageRank_kernel<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_pageRankVector_1_pointer, d_pageRankVector_2_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);



                // pageRankKernel<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_pageRankVector_1_pointer, d_pageRankVector_2_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer);
                cudaDeviceSynchronize();
                pageRank_time = clock() - pageRank_time;                        // thread_blocks = ceil(double(vertex_size) / 256);
                // pageRankKernel<<< thread_blocks, 256>>>(device_vertex_dictionary, vertex_size, d_pageRankVector_1, d_pageRankVector_2, d_source_degrees_new_pointer);

                break;
            }

            case 5  :
            {
                // float *d_pageRankVector_1, *d_pageRankVector_2;



                thrust::device_vector <float> d_triangleCount(vertex_size);
                thrust::device_vector <unsigned long> d_TC_edge_vector(edge_size);
                // thrust::device_vector <float> d_pageRankVector_2(vertex_size);

                float* d_triangleCount_pointer = thrust::raw_pointer_cast(d_triangleCount.data());
                unsigned long* d_TC_edge_vector_pointer = thrust::raw_pointer_cast(d_TC_edge_vector.data());

                // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
                // pageRank_kernel_preprocessing<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);

                // thread_blocks = ceil(double(h_prefix_sum_edge_blocks_new[vertex_size]) / THREADS_PER_BLOCK);
                // triangle_counting_kernel_1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer);

                unsigned long tc_type;
                std::cout << "Enter type of TC to be performed" << std::endl << "1. Vertex-centric" << std::endl << "2. Edge-centric" << std::endl;
                std::cin >> tc_type;

                if(tc_type == 1) {
                    triangleCounting_time = clock();
                    thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
                    triangle_counting_kernel_VC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_triangleCount_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer);
                }
                else if(tc_type == 2) {

                    unsigned long h_tc_thread_count, *d_tc_thread_count;
                    cudaMalloc(&d_tc_thread_count, sizeof(unsigned long));

                    thrust::device_vector <unsigned long> d_second_vertex_degrees(vertex_size + 1);
                    unsigned long *d_second_vertex_degrees_pointer = thrust::raw_pointer_cast(d_second_vertex_degrees.data());

                    triangleCounting_time = clock();
                    thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
                    triangle_counting_kernel_EC_1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_triangleCount_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_second_vertex_degrees_pointer);
                    // tc_second_vertex_offset_calculation<<< 1, 1>>>(d_source_degrees_new_pointer ,d_csr_offset_new_pointer, vertex_size, d_second_vertex_degrees_pointer, d_tc_thread_count);
                    cudaDeviceSynchronize();

                    thrust::exclusive_scan(thrust::device, d_second_vertex_degrees_pointer, d_second_vertex_degrees_pointer + vertex_size + 1, d_second_vertex_degrees_pointer);
                    cudaDeviceSynchronize();

                    tc_second_vertex_offset_calculation<<< 1, 1>>>(d_source_degrees_new_pointer ,d_csr_offset_new_pointer, vertex_size, d_second_vertex_degrees_pointer, d_tc_thread_count);

                    cudaMemcpy(&h_tc_thread_count, d_tc_thread_count, sizeof(unsigned long),cudaMemcpyDeviceToHost);
                    cudaDeviceSynchronize();
                    // std::cout << "Second kernel thread number is " << h_tc_thread_count << std::endl;
                    thread_blocks = ceil(double(h_tc_thread_count) / THREADS_PER_BLOCK);
                    triangle_counting_kernel_EC_2<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_triangleCount_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_second_vertex_degrees_pointer);


                }

                // thread_blocks = ceil(double(edge_size) / THREADS_PER_BLOCK);
                // triangle_counting_kernel_2<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_triangleCount_pointer, d_source_vector_1_pointer);

                // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
                // triangleCountingKernel<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_triangleCount_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer);
                cudaDeviceSynchronize();
                triangleCounting_time = clock() - triangleCounting_time;
                // thread_blocks = ceil(double(vertex_size) / 256);
                // pageRankKernel<<< thread_blocks, 256>>>(device_vertex_dictionary, vertex_size, d_pageRankVector_1, d_pageRankVector_2, d_source_degrees_new_pointer);

                print_triangle_counts<<<1, 1>>>(d_triangleCount_pointer, vertex_size);

                // pageRank_time = clock() - pageRank_time;
                break;
            }

            case 6  :
            {

                thrust::device_vector <unsigned int> d_shortest_path(vertex_size);
                thrust::fill(d_shortest_path.begin(), d_shortest_path.end(), UINT_MAX);
                // thrust::device_vector <unsigned long long> d_mutex(vertex_size);
                thrust::device_vector <unsigned long> d_mutex(vertex_size);

                thrust::device_vector <unsigned long> d_TC_edge_vector(edge_size);

                unsigned int* d_shortest_path_pointer = thrust::raw_pointer_cast(d_shortest_path.data());
                // unsigned long long* d_mutex_pointer = thrust::raw_pointer_cast(d_mutex.data());
                unsigned long* d_mutex_pointer = thrust::raw_pointer_cast(d_mutex.data());
                unsigned long* d_TC_edge_vector_pointer = thrust::raw_pointer_cast(d_TC_edge_vector.data());

                // thrust::device_vector <float> d_sssp_queue_1(vertex_size);
                // thrust::device_vector <float> d_sssp_queue_2(vertex_size);
                thrust::device_vector <unsigned long long> d_sssp_queue_1(vertex_size * 4);
                thrust::device_vector <unsigned long long> d_sssp_queue_2(vertex_size * 4);

                // float* d_sssp_queue_1_pointer = thrust::raw_pointer_cast(d_sssp_queue_1.data());
                // float* d_sssp_queue_2_pointer = thrust::raw_pointer_cast(d_sssp_queue_2.data());
                unsigned long long* d_sssp_queue_1_pointer = thrust::raw_pointer_cast(d_sssp_queue_1.data());
                unsigned long long* d_sssp_queue_2_pointer = thrust::raw_pointer_cast(d_sssp_queue_2.data());
                // UFM Test
                // short *u_search_flag;
                // cudaMallocManaged(&u_search_flag, sizeof(short));
                unsigned long iterations = 0;

                unsigned long long h_dp_thread_count = 1, h_type = 0;
                unsigned long long *d_dp_thread_count, *d_type;
                cudaMalloc(&d_dp_thread_count, sizeof(unsigned long long));

                unsigned long sssp_type;
                std::cout << "Enter type of SSSP to be performed" << std::endl << "1. Vertex-centric Worklist" << std::endl << "2. Edge-centric Worklist" << std::endl << "3. Multiple Worklists" << std::endl << "4. Load-balanced edge-centric Worklist" << std::endl << "5. Edge-Block-centric Worklist" << std::endl << "6. Load-balanced vertex-centric Worklist" << std::endl;
                std::cin >> sssp_type;

                cudaDeviceSynchronize();

                if(sssp_type == 1) {

                    sssp_time = clock();

                    // // below is Approach 1
                    // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
                    // pageRank_kernel_preprocessing<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);

                    // // sssp_kernel_preprocessing<<< 1, 1>>>(d_shortest_path_pointer, d_search_flag);
                    // sssp_kernel_preprocessing<<< 1, 1>>>(d_shortest_path_pointer, d_search_flag, d_mutex_pointer);

                    // // // might need to launch this kernel V times worst case
                    // // // launch with number of threads equal to the edge blocks used by the data structure
                    // do {

                    //     h_search_flag = 0;
                    //     cudaMemcpy(d_search_flag, &h_search_flag, sizeof(unsigned long), cudaMemcpyHostToDevice);

                    //     thread_blocks = ceil(double(h_prefix_sum_edge_blocks_new[vertex_size]) / THREADS_PER_BLOCK);
                    //     sssp_kernel<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_mutex_pointer, d_search_flag);

                    //     cudaMemcpy(&h_search_flag, d_search_flag, sizeof(unsigned long),cudaMemcpyDeviceToHost);

                    //     cudaDeviceSynchronize();
                    //     // printf("h_search_flag at host is %lu\n", h_search_flag);

                    // } while(h_search_flag);

                    // Below is Approach 2

                    // sssp_kernel_master<<< 1, 1>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer);

                    // below is UFM test

                    // do {

                    //     // std::cout << "Iteration #" << iterations << std::endl;

                    //     d_search_flag = 0;
                    //     // u_search_flag = 0;
                    //     h_search_flag = 0;
                    //     cudaMemcpy(d_search_flag, &h_search_flag, sizeof(unsigned long), cudaMemcpyHostToDevice);

                    //     // unsigned long thread_blocks = ceil(double(batch_size) / THREADS_PER_BLOCK);
                    //     // sssp_kernel_child<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_search_flag);

                    //     unsigned long thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
                    //     sssp_kernel_child_VC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_search_flag, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer);

                    //     cudaMemcpy(&h_search_flag, d_search_flag, sizeof(unsigned long),cudaMemcpyDeviceToHost);

                    //     cudaDeviceSynchronize();
                    //     iterations++;

                    //     // printf("Flag after iteration is %lu\n", d_search_flag);

                    // } while(h_search_flag);
                    // } while(iterations < 3355);
                    // } while(iterations < 3495);
                    // } while(d_search_flag);


                    // thread_blocks = ceil(double(h_prefix_sum_edge_blocks_new[vertex_size]) / THREADS_PER_BLOCK);
                    // triangle_counting_kernel_1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer);


                    // thread_blocks = ceil(double(edge_size) / THREADS_PER_BLOCK);
                    // triangle_counting_kernel_2<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_triangleCount_pointer, d_source_vector_1_pointer);



                    // std::cout << "Iterations for SSSP: " << iterations << std::endl;
                    // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
                    // pageRank_kernel_preprocessing<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);

                    // thread_blocks = ceil(double(h_prefix_sum_edge_blocks_new[vertex_size]) / THREADS_PER_BLOCK);
                    // triangle_counting_kernel_1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer);

                    sssp_kernel_preprocessing<<< 1, 1>>>(d_shortest_path_pointer, d_search_flag, d_mutex_pointer, d_sssp_queue_1_pointer);


                    // below is worklist based

                    do {

                        // d_search_flag = 0;
                        // printf("Iteration #%lu, thread_count=%lu\n", iterations, int(d_dp_thread_count));
                        // printf("Iteration #%lu, thread_count=%lu\n", iterations, d_dp_thread_count);

                        unsigned long thread_blocks = ceil(double(h_dp_thread_count) / THREADS_PER_BLOCK);

                        // if(iterations)
                        h_dp_thread_count = 0;
                        cudaMemcpy(d_dp_thread_count, &h_dp_thread_count, sizeof(unsigned long long), cudaMemcpyHostToDevice);
                        // cudaMemcpy(d_type, &h_type, sizeof(unsigned long), cudaMemcpyHostToDevice);

                        // sssp_kernel_child<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path, d_source_degrees, d_csr_offset, d_csr_edges, d_prefix_sum_edge_blocks, d_source_vector, d_TC_edge_vector, d_source_vector_1, d_mutex);
                        // sssp_kernel_child_VC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_dp_thread_count);

                        // if(!(iterations % 2))
                        //     sssp_kernel_VC_preorder<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_dp_thread_count);
                        // else
                        //     sssp_kernel_VC_preorder<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_2_pointer, d_sssp_queue_1_pointer, d_dp_thread_count);

                        if(!(iterations % 2))
                            sssp_kernel_VC_iterative<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_dp_thread_count);
                        else
                            sssp_kernel_VC_iterative<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_2_pointer, d_sssp_queue_1_pointer, d_dp_thread_count);


                        // sssp_kernel_master<<< 1, 1>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer);
                        sssp_kernel_postprocess<<< 1, 1>>>(d_dp_thread_count);
                        cudaDeviceSynchronize();
                        // d_dp_thread_count = 0;

                        // if(h_type)
                        //     h_type = 0;
                        // else
                        //     h_type = 1;

                        iterations++;
                        printf("Iteration %lu\n", iterations);

                        cudaMemcpy(&h_dp_thread_count, d_dp_thread_count, sizeof(unsigned long long),cudaMemcpyDeviceToHost);

                        // d_prev_thread_count = d_dp_thread_count;

                    } while(h_dp_thread_count);

                    cudaDeviceSynchronize();
                    sssp_time = clock() - sssp_time;

                }

                else if (sssp_type == 2) {

                    clock_t temp;


                    unsigned long long h_sssp_e_threads, *d_sssp_e_threads;
                    cudaMalloc(&d_sssp_e_threads, sizeof(unsigned long long));

                    unsigned long long *d_sssp_output_frontier_offset;
                    cudaMalloc((unsigned long long**)&d_sssp_output_frontier_offset, (vertex_size + 1) * sizeof(unsigned long long));

                    unsigned long long *d_sssp_output_frontier_degrees;
                    cudaMalloc((unsigned long long**)&d_sssp_output_frontier_degrees, (vertex_size + 1) * sizeof(unsigned long long));


                    // thrust::device_pointer_cast(deg_for_input_frontier);
                    // thrust_output_ptr = thrust::device_pointer_cast(frontier_offset);
                    // thrust::exclusive_scan(thrust::device, thrust_input_ptr, thrust_input_ptr + frontier_size + 1, thrust_output_ptr);


                    // thrust::device_vector <unsigned long long> d_sssp_output_frontier_offset(edge_size);
                    // unsigned long long* d_sssp_output_frontier_offset_pointer = thrust::raw_pointer_cast(d_sssp_output_frontier_offset.data());

                    unsigned long long* d_sssp_output_frontier_offset_pointer = thrust::raw_pointer_cast(d_sssp_output_frontier_offset);
                    unsigned long long* d_sssp_output_frontier_degrees_pointer = thrust::raw_pointer_cast(d_sssp_output_frontier_degrees);
                    // unsigned long long *d_sssp_output_frontier_offset_pointer = thrust::device_pointer_cast(d_sssp_output_frontier_offset);

                    float time;
                    sssp_time = clock();

                    // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
                    // pageRank_kernel_preprocessing<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);

                    // thread_blocks = ceil(double(h_prefix_sum_edge_blocks_new[vertex_size]) / THREADS_PER_BLOCK);
                    // triangle_counting_kernel_1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer);

                    sssp_kernel_preprocessing<<< 1, 1>>>(d_shortest_path_pointer, d_search_flag, d_mutex_pointer, d_sssp_queue_1_pointer);
                    cudaDeviceSynchronize();
                    cudaEvent_t start, stop;
                    cudaEventCreate(&start);
                    cudaEventCreate(&stop);
                    cudaEventRecord(start);
                    do {

                        h_sssp_e_threads = 0;


                        thread_blocks = ceil(double(h_dp_thread_count) / THREADS_PER_BLOCK);
                        temp = clock();

                        if(!(iterations % 2))
                            sssp_output_frontier_preprocessing<<<thread_blocks, THREADS_PER_BLOCK>>>(d_source_degrees_new_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_sssp_output_frontier_degrees_pointer, d_csr_offset_new_pointer);
                        else
                            sssp_output_frontier_preprocessing<<<thread_blocks, THREADS_PER_BLOCK>>>(d_source_degrees_new_pointer, d_sssp_queue_2_pointer, d_sssp_queue_1_pointer, d_sssp_output_frontier_degrees_pointer, d_csr_offset_new_pointer);

                        cudaDeviceSynchronize();


                        // std::cout << "Frontier size is " << h_dp_thread_count << " and Prefix sum time is " << (float)temp/CLOCKS_PER_SEC << "seconds" << std::endl;

                        // thrust::exclusive_scan(d_sssp_output_frontier_offset.begin(), d_sssp_output_frontier_offset.begin() + h_dp_thread_count + 1, d_sssp_output_frontier_offset.begin());
                        // thrust::exclusive_scan(thrust::device, d_sssp_output_frontier_offset_pointer, d_sssp_output_frontier_offset_pointer + h_dp_thread_count + 1, d_sssp_output_frontier_offset_pointer);
                        thrust::exclusive_scan(thrust::device, d_sssp_output_frontier_degrees_pointer, d_sssp_output_frontier_degrees_pointer + h_dp_thread_count + 1, d_sssp_output_frontier_offset_pointer);
                        cudaDeviceSynchronize();

                        sssp_output_frontier_offset_calculation<<< 1, 1>>>(d_source_degrees_new_pointer ,d_csr_offset_new_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_sssp_output_frontier_offset_pointer, d_sssp_e_threads);
                        cudaDeviceSynchronize();
                        temp = clock() - temp;
                        prefix_sum_time += temp;
                        // std::cout << "Frontier vertex count at host is " << h_dp_thread_count << std::endl;
                        // std::cout << "Frontier size is " << h_dp_thread_count << " and Prefix sum time is " << (float)temp/CLOCKS_PER_SEC << "seconds" << std::endl;

                        // cudaMemcpy(&h_sssp_e_threads, (d_sssp_output_frontier_offset_pointer + h_dp_thread_count), sizeof(unsigned long long),cudaMemcpyDeviceToHost);
                        // cudaMemcpy(&h_sssp_e_threads, &(d_sssp_output_frontier_offset_pointer[h_dp_thread_count]), sizeof(unsigned long long),cudaMemcpyDeviceToHost);
                        cudaMemcpy(&h_sssp_e_threads, d_sssp_e_threads, sizeof(unsigned long long),cudaMemcpyDeviceToHost);
                        cudaDeviceSynchronize();
                        // std::cout << "thread_count is " << h_dp_thread_count << std::endl;
                        h_dp_thread_count = 0;
                        cudaMemcpy(d_dp_thread_count, &h_dp_thread_count, sizeof(unsigned long long), cudaMemcpyHostToDevice);
                        cudaDeviceSynchronize();
                        // std::cout << "Before kernel" << std::endl;

                        // std::cout << h_sssp_e_threads / 8 << " threads needed" << std::endl;
                        // if(!(h_sssp_e_threads / 8))
                        //     h_sssp_e_threads = 8;
                        unsigned long thread_blocks = ceil(double(h_sssp_e_threads) / THREADS_PER_BLOCK);

                        // if(iterations == 2)
                        //     break;

                        // sssp_kernel_child_VC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_dp_thread_count);

                        if(!(iterations % 2))
                            sssp_kernel_EC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_dp_thread_count, d_sssp_output_frontier_offset_pointer);
                        else
                            sssp_kernel_EC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_2_pointer, d_sssp_queue_1_pointer, d_dp_thread_count, d_sssp_output_frontier_offset_pointer);


                        sssp_kernel_postprocess<<< 1, 1>>>(d_dp_thread_count);
                        cudaDeviceSynchronize();

                        iterations++;
                        // printf("Iteration %lu\n", iterations);

                        cudaMemcpy(&h_dp_thread_count, d_dp_thread_count, sizeof(unsigned long long),cudaMemcpyDeviceToHost);
                        cudaDeviceSynchronize();

                    } while(h_dp_thread_count);

                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&time, start, stop);

                    cudaDeviceSynchronize();
                    sssp_time = clock() - sssp_time;
                    std::cout << "Total Time: " << time << std::endl;

                }

                else if(sssp_type == 3) {

                    unsigned long long h_dp_thread_count_16to31 = 0;
                    unsigned long long *d_dp_thread_count_16to31;
                    cudaMalloc(&d_dp_thread_count_16to31, sizeof(unsigned long long));


                    thrust::device_vector <unsigned long long> d_sssp_queue_1_16to32(vertex_size);
                    thrust::device_vector <unsigned long long> d_sssp_queue_2_16to32(vertex_size);

                    // float* d_sssp_queue_1_pointer = thrust::raw_pointer_cast(d_sssp_queue_1.data());
                    // float* d_sssp_queue_2_pointer = thrust::raw_pointer_cast(d_sssp_queue_2.data());
                    unsigned long long* d_sssp_queue_1_16to32_pointer = thrust::raw_pointer_cast(d_sssp_queue_1_16to32.data());
                    unsigned long long* d_sssp_queue_2_16to32_pointer = thrust::raw_pointer_cast(d_sssp_queue_2_16to32.data());

                    sssp_time = clock();

                    thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
                    pageRank_kernel_preprocessing<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);

                    thread_blocks = ceil(double(h_prefix_sum_edge_blocks_new[vertex_size]) / THREADS_PER_BLOCK);
                    triangle_counting_kernel_1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer);

                    sssp_kernel_preprocessing<<< 1, 1>>>(d_shortest_path_pointer, d_search_flag, d_mutex_pointer, d_sssp_queue_1_pointer);


                    // below is worklist based

                    do {

                        // d_search_flag = 0;
                        // printf("Iteration #%lu, thread_count=%lu\n", iterations, int(d_dp_thread_count));
                        // printf("Iteration #%lu, thread_count=%lu\n", iterations, d_dp_thread_count);

                        unsigned long thread_blocks = ceil(double(h_dp_thread_count) / THREADS_PER_BLOCK);
                        unsigned long h_dp_thread_count_16to31_copy = h_dp_thread_count_16to31;
                        // if(iterations)
                        h_dp_thread_count = 0;
                        cudaMemcpy(d_dp_thread_count, &h_dp_thread_count, sizeof(unsigned long long), cudaMemcpyHostToDevice);
                        h_dp_thread_count_16to31 = 0;
                        cudaMemcpy(d_dp_thread_count_16to31, &h_dp_thread_count_16to31, sizeof(unsigned long long), cudaMemcpyHostToDevice);
                        cudaDeviceSynchronize();

                        // cudaMemcpy(d_type, &h_type, sizeof(unsigned long), cudaMemcpyHostToDevice);

                        // sssp_kernel_child<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path, d_source_degrees, d_csr_offset, d_csr_edges, d_prefix_sum_edge_blocks, d_source_vector, d_TC_edge_vector, d_source_vector_1, d_mutex);
                        sssp_kernel_multiple_worklists_0to15<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_dp_thread_count, d_sssp_queue_1_16to32_pointer, d_sssp_queue_2_16to32_pointer, d_dp_thread_count_16to31);

                        cudaDeviceSynchronize();
                        // std::cout << "Checkpoint" << std::endl;

                        if(h_dp_thread_count_16to31_copy) {

                            // printf("Hit Iteration %lu\n", iterations);

                            thread_blocks = ceil(double(h_dp_thread_count_16to31_copy) / THREADS_PER_BLOCK);
                            // h_dp_thread_count_16to31 = 0;
                            // cudaMemcpy(d_dp_thread_count_16to31, &h_dp_thread_count_16to31, sizeof(unsigned long long), cudaMemcpyHostToDevice);
                            sssp_kernel_multiple_worklists_16to31<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_dp_thread_count, d_sssp_queue_1_16to32_pointer, d_sssp_queue_2_16to32_pointer, d_dp_thread_count_16to31);

                            cudaDeviceSynchronize();
                            // std::cout << "Checkpoint" << std::endl;
                        }
                        // sssp_kernel_master<<< 1, 1>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer);
                        sssp_kernel_postprocess<<< 1, 1>>>(d_dp_thread_count, d_dp_thread_count_16to31, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_sssp_queue_1_16to32_pointer, d_sssp_queue_2_16to32_pointer);
                        cudaDeviceSynchronize();
                        // d_dp_thread_count = 0;



                        iterations++;
                        // printf("Iteration %lu\n", iterations);

                        cudaMemcpy(&h_dp_thread_count, d_dp_thread_count, sizeof(unsigned long long),cudaMemcpyDeviceToHost);
                        cudaMemcpy(&h_dp_thread_count_16to31, d_dp_thread_count_16to31, sizeof(unsigned long long),cudaMemcpyDeviceToHost);
                        cudaDeviceSynchronize();

                        // std::cout << "Total threads is " << h_dp_thread_count + h_dp_thread_count_16to31 << std::endl;

                        // d_prev_thread_count = d_dp_thread_count;

                    } while(h_dp_thread_count + h_dp_thread_count_16to31);

                    cudaDeviceSynchronize();
                    sssp_time = clock() - sssp_time;

                }

                else if (sssp_type == 4) {

                    clock_t temp;


                    unsigned long long h_sssp_e_threads, *d_sssp_e_threads;
                    cudaMalloc(&d_sssp_e_threads, sizeof(unsigned long long));

                    unsigned long long *d_sssp_output_frontier_offset;
                    cudaMalloc((unsigned long long**)&d_sssp_output_frontier_offset, (vertex_size + 1) * sizeof(unsigned long long));

                    unsigned long long *d_sssp_output_frontier_degrees;
                    cudaMalloc((unsigned long long**)&d_sssp_output_frontier_degrees, (vertex_size + 1) * sizeof(unsigned long long));


                    // thrust::device_pointer_cast(deg_for_input_frontier);
                    // thrust_output_ptr = thrust::device_pointer_cast(frontier_offset);
                    // thrust::exclusive_scan(thrust::device, thrust_input_ptr, thrust_input_ptr + frontier_size + 1, thrust_output_ptr);


                    // thrust::device_vector <unsigned long long> d_sssp_output_frontier_offset(edge_size);
                    // unsigned long long* d_sssp_output_frontier_offset_pointer = thrust::raw_pointer_cast(d_sssp_output_frontier_offset.data());

                    unsigned long long* d_sssp_output_frontier_offset_pointer = thrust::raw_pointer_cast(d_sssp_output_frontier_offset);
                    unsigned long long* d_sssp_output_frontier_degrees_pointer = thrust::raw_pointer_cast(d_sssp_output_frontier_degrees);
                    // unsigned long long *d_sssp_output_frontier_offset_pointer = thrust::device_pointer_cast(d_sssp_output_frontier_offset);

                    float time;
                    sssp_time = clock();

                    // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
                    // pageRank_kernel_preprocessing<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);

                    // thread_blocks = ceil(double(h_prefix_sum_edge_blocks_new[vertex_size]) / THREADS_PER_BLOCK);
                    // triangle_counting_kernel_1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer);

                    sssp_kernel_preprocessing<<< 1, 1>>>(d_shortest_path_pointer, d_search_flag, d_mutex_pointer, d_sssp_queue_1_pointer);
                    cudaDeviceSynchronize();
                    cudaEvent_t start, stop;
                    cudaEventCreate(&start);
                    cudaEventCreate(&stop);
                    cudaEventRecord(start);
                    do {

                        iterations++;
                        // printf("\nIteration %lu\n", iterations);

                        h_sssp_e_threads = 0;


                        thread_blocks = ceil(double(h_dp_thread_count) / THREADS_PER_BLOCK);
                        temp = clock();

                        if(!(iterations % 2))
                            sssp_output_frontier_preprocessing<<<thread_blocks, THREADS_PER_BLOCK>>>(d_source_degrees_new_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_sssp_output_frontier_degrees_pointer, d_csr_offset_new_pointer);
                        else
                            sssp_output_frontier_preprocessing<<<thread_blocks, THREADS_PER_BLOCK>>>(d_source_degrees_new_pointer, d_sssp_queue_2_pointer, d_sssp_queue_1_pointer, d_sssp_output_frontier_degrees_pointer, d_csr_offset_new_pointer);

                        cudaDeviceSynchronize();


                        // std::cout << "Frontier size is " << h_dp_thread_count << " and Prefix sum time is " << (float)temp/CLOCKS_PER_SEC << "seconds" << std::endl;

                        // thrust::exclusive_scan(d_sssp_output_frontier_offset.begin(), d_sssp_output_frontier_offset.begin() + h_dp_thread_count + 1, d_sssp_output_frontier_offset.begin());
                        // thrust::exclusive_scan(thrust::device, d_sssp_output_frontier_offset_pointer, d_sssp_output_frontier_offset_pointer + h_dp_thread_count + 1, d_sssp_output_frontier_offset_pointer);
                        thrust::exclusive_scan(thrust::device, d_sssp_output_frontier_degrees_pointer, d_sssp_output_frontier_degrees_pointer + h_dp_thread_count + 1, d_sssp_output_frontier_offset_pointer);
                        cudaDeviceSynchronize();

                        sssp_output_frontier_offset_calculation<<< 1, 1>>>(d_source_degrees_new_pointer ,d_csr_offset_new_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_sssp_output_frontier_offset_pointer, d_sssp_e_threads);
                        cudaDeviceSynchronize();
                        temp = clock() - temp;
                        prefix_sum_time += temp;
                        // std::cout << "Frontier vertex count at host is " << h_dp_thread_count << std::endl;
                        // std::cout << "Frontier size is " << h_dp_thread_count << " and Prefix sum time is " << (float)temp/CLOCKS_PER_SEC << "seconds" << std::endl;

                        // cudaMemcpy(&h_sssp_e_threads, (d_sssp_output_frontier_offset_pointer + h_dp_thread_count), sizeof(unsigned long long),cudaMemcpyDeviceToHost);
                        // cudaMemcpy(&h_sssp_e_threads, &(d_sssp_output_frontier_offset_pointer[h_dp_thread_count]), sizeof(unsigned long long),cudaMemcpyDeviceToHost);
                        cudaMemcpy(&h_sssp_e_threads, d_sssp_e_threads, sizeof(unsigned long long),cudaMemcpyDeviceToHost);
                        cudaDeviceSynchronize();
                        // std::cout << "thread_count is " << h_dp_thread_count << std::endl;
                        h_dp_thread_count = 0;
                        cudaMemcpy(d_dp_thread_count, &h_dp_thread_count, sizeof(unsigned long long), cudaMemcpyHostToDevice);
                        cudaDeviceSynchronize();
                        // std::cout << "Before kernel" << std::endl;

                        if(!(h_sssp_e_threads / SSSP_LOAD_FACTOR))
                            h_sssp_e_threads = SSSP_LOAD_FACTOR;
                        // std::cout << ceil((double)h_sssp_e_threads / 8) << " threads needed" << std::endl;
                        unsigned long thread_blocks = ceil(double(h_sssp_e_threads / SSSP_LOAD_FACTOR) / THREADS_PER_BLOCK);

                        // if(iterations == 2)
                        //     break;

                        // sssp_kernel_child_VC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_dp_thread_count);

                        if(!(iterations % 2))
                            sssp_kernel_EC_load_balanced<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, edge_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_dp_thread_count, d_sssp_output_frontier_offset_pointer);
                        else
                            sssp_kernel_EC_load_balanced<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, edge_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_2_pointer, d_sssp_queue_1_pointer, d_dp_thread_count, d_sssp_output_frontier_offset_pointer);


                        sssp_kernel_postprocess<<< 1, 1>>>(d_dp_thread_count);
                        cudaDeviceSynchronize();

                        cudaMemcpy(&h_dp_thread_count, d_dp_thread_count, sizeof(unsigned long long),cudaMemcpyDeviceToHost);
                        cudaDeviceSynchronize();

                    } while(h_dp_thread_count);

                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&time, start, stop);

                    cudaDeviceSynchronize();
                    sssp_time = clock() - sssp_time;
                    std::cout << "Total Time: " << time << std::endl;

                }

                else if (sssp_type == 5) {

                    clock_t temp;


                    unsigned long long h_sssp_e_threads, *d_sssp_e_threads;
                    cudaMalloc(&d_sssp_e_threads, sizeof(unsigned long long));

                    unsigned long long *d_sssp_output_frontier_offset;
                    cudaMalloc((unsigned long long**)&d_sssp_output_frontier_offset, (vertex_size + 1) * sizeof(unsigned long long));

                    unsigned long long *d_sssp_output_frontier_degrees;
                    cudaMalloc((unsigned long long**)&d_sssp_output_frontier_degrees, (vertex_size + 1) * sizeof(unsigned long long));

                    unsigned long thread_multiplier = EDGE_BLOCK_SIZE / 10;

                    // thrust::device_pointer_cast(deg_for_input_frontier);
                    // thrust_output_ptr = thrust::device_pointer_cast(frontier_offset);
                    // thrust::exclusive_scan(thrust::device, thrust_input_ptr, thrust_input_ptr + frontier_size + 1, thrust_output_ptr);


                    // thrust::device_vector <unsigned long long> d_sssp_output_frontier_offset(edge_size);
                    // unsigned long long* d_sssp_output_frontier_offset_pointer = thrust::raw_pointer_cast(d_sssp_output_frontier_offset.data());

                    unsigned long long* d_sssp_output_frontier_offset_pointer = thrust::raw_pointer_cast(d_sssp_output_frontier_offset);
                    unsigned long long* d_sssp_output_frontier_degrees_pointer = thrust::raw_pointer_cast(d_sssp_output_frontier_degrees);
                    // unsigned long long *d_sssp_output_frontier_offset_pointer = thrust::device_pointer_cast(d_sssp_output_frontier_offset);

                    float time;
                    sssp_time = clock();

                    // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
                    // pageRank_kernel_preprocessing<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);

                    // thread_blocks = ceil(double(h_prefix_sum_edge_blocks_new[vertex_size]) / THREADS_PER_BLOCK);
                    // triangle_counting_kernel_1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer);

                    sssp_kernel_preprocessing<<< 1, 1>>>(d_shortest_path_pointer, d_search_flag, d_mutex_pointer, d_sssp_queue_1_pointer);
                    cudaDeviceSynchronize();
                    cudaEvent_t start, stop;
                    cudaEventCreate(&start);
                    cudaEventCreate(&stop);
                    cudaEventRecord(start);
                    do {

                        iterations++;
                        // printf("\nIteration %lu\n", iterations);

                        h_sssp_e_threads = 0;


                        thread_blocks = ceil(double(h_dp_thread_count) / THREADS_PER_BLOCK);
                        temp = clock();

                        if(!(iterations % 2))
                            sssp_output_frontier_preprocessing_EBC<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, d_source_degrees_new_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_sssp_output_frontier_degrees_pointer, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer);
                        else
                            sssp_output_frontier_preprocessing_EBC<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, d_source_degrees_new_pointer, d_sssp_queue_2_pointer, d_sssp_queue_1_pointer, d_sssp_output_frontier_degrees_pointer, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer);

                        cudaDeviceSynchronize();


                        // std::cout << "Frontier size is " << h_dp_thread_count << " and Prefix sum time is " << (float)temp/CLOCKS_PER_SEC << "seconds" << std::endl;

                        // thrust::exclusive_scan(d_sssp_output_frontier_offset.begin(), d_sssp_output_frontier_offset.begin() + h_dp_thread_count + 1, d_sssp_output_frontier_offset.begin());
                        // thrust::exclusive_scan(thrust::device, d_sssp_output_frontier_offset_pointer, d_sssp_output_frontier_offset_pointer + h_dp_thread_count + 1, d_sssp_output_frontier_offset_pointer);
                        thrust::exclusive_scan(thrust::device, d_sssp_output_frontier_degrees_pointer, d_sssp_output_frontier_degrees_pointer + h_dp_thread_count + 1, d_sssp_output_frontier_offset_pointer);
                        cudaDeviceSynchronize();

                        sssp_output_frontier_offset_calculation<<< 1, 1>>>(d_source_degrees_new_pointer ,d_csr_offset_new_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_sssp_output_frontier_offset_pointer, d_sssp_e_threads);
                        cudaDeviceSynchronize();
                        temp = clock() - temp;
                        prefix_sum_time += temp;
                        // std::cout << "Frontier vertex count at host is " << h_dp_thread_count << std::endl;
                        // std::cout << "Frontier size is " << h_dp_thread_count << " and Prefix sum time is " << (float)temp/CLOCKS_PER_SEC << "seconds" << std::endl;

                        // cudaMemcpy(&h_sssp_e_threads, (d_sssp_output_frontier_offset_pointer + h_dp_thread_count), sizeof(unsigned long long),cudaMemcpyDeviceToHost);
                        // cudaMemcpy(&h_sssp_e_threads, &(d_sssp_output_frontier_offset_pointer[h_dp_thread_count]), sizeof(unsigned long long),cudaMemcpyDeviceToHost);
                        cudaMemcpy(&h_sssp_e_threads, d_sssp_e_threads, sizeof(unsigned long long),cudaMemcpyDeviceToHost);
                        cudaDeviceSynchronize();
                        // std::cout << "thread_count is " << h_dp_thread_count << std::endl;
                        h_dp_thread_count = 0;
                        cudaMemcpy(d_dp_thread_count, &h_dp_thread_count, sizeof(unsigned long long), cudaMemcpyHostToDevice);
                        cudaDeviceSynchronize();
                        // std::cout << "Before kernel" << std::endl;

                        // if(!(h_sssp_e_threads / SSSP_LOAD_FACTOR))
                        //     h_sssp_e_threads = SSSP_LOAD_FACTOR;
                        // std::cout << h_sssp_e_threads << " threads needed" << std::endl;
                        unsigned long thread_blocks = ceil((double(h_sssp_e_threads) * thread_multiplier) / THREADS_PER_BLOCK);

                        // if(iterations == 2)
                        //     break;

                        // sssp_kernel_child_VC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_dp_thread_count);

                        if(!(iterations % 2))
                            sssp_kernel_EBC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, edge_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_dp_thread_count, d_sssp_output_frontier_offset_pointer, thread_multiplier);
                        else
                            sssp_kernel_EBC<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, edge_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_2_pointer, d_sssp_queue_1_pointer, d_dp_thread_count, d_sssp_output_frontier_offset_pointer, thread_multiplier);


                        sssp_kernel_postprocess<<< 1, 1>>>(d_dp_thread_count);
                        cudaDeviceSynchronize();

                        cudaMemcpy(&h_dp_thread_count, d_dp_thread_count, sizeof(unsigned long long),cudaMemcpyDeviceToHost);
                        cudaDeviceSynchronize();

                    } while(h_dp_thread_count);

                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&time, start, stop);

                    cudaDeviceSynchronize();
                    sssp_time = clock() - sssp_time;
                    std::cout << "Total Time: " << time << std::endl;

                }

                if(sssp_type == 6) {

                    unsigned long thread_multiplier = EDGE_BLOCK_SIZE / 10;
                    if(!thread_multiplier)
                        thread_multiplier = 1;

                    sssp_time = clock();

                    sssp_kernel_preprocessing<<< 1, 1>>>(d_shortest_path_pointer, d_search_flag, d_mutex_pointer, d_sssp_queue_1_pointer);


                    // below is worklist based

                    do {

                        // d_search_flag = 0;
                        // printf("Iteration #%lu, thread_count=%lu\n", iterations, int(d_dp_thread_count));
                        // printf("Iteration #%lu, thread_count=%lu\n", iterations, d_dp_thread_count);

                        unsigned long thread_blocks = ceil(double(h_dp_thread_count * thread_multiplier) / THREADS_PER_BLOCK);

                        // if(iterations)
                        h_dp_thread_count = 0;
                        cudaMemcpy(d_dp_thread_count, &h_dp_thread_count, sizeof(unsigned long long), cudaMemcpyHostToDevice);
                        // cudaMemcpy(d_type, &h_type, sizeof(unsigned long), cudaMemcpyHostToDevice);


                        if(!(iterations % 2))
                            sssp_kernel_VC_iterative_LB<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_1_pointer, d_sssp_queue_2_pointer, d_dp_thread_count, thread_multiplier);
                        else
                            sssp_kernel_VC_iterative_LB<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, batch_size, d_shortest_path_pointer, d_source_degrees_new_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer, d_TC_edge_vector_pointer, d_source_vector_1_pointer, d_mutex_pointer, d_sssp_queue_2_pointer, d_sssp_queue_1_pointer, d_dp_thread_count, thread_multiplier);


                        sssp_kernel_postprocess<<< 1, 1>>>(d_dp_thread_count);
                        cudaDeviceSynchronize();

                        iterations++;
                        // printf("Iteration %lu\n", iterations);

                        cudaMemcpy(&h_dp_thread_count, d_dp_thread_count, sizeof(unsigned long long),cudaMemcpyDeviceToHost);

                        // d_prev_thread_count = d_dp_thread_count;

                    } while(h_dp_thread_count);

                    cudaDeviceSynchronize();
                    sssp_time = clock() - sssp_time;

                }

                printf("Iterations for SSSP: %lu", iterations);

                // worklist based end

                print_sssp_values<<< 1, 1>>>(d_shortest_path_pointer, vertex_size);

                break;
            }

            case 7  :
                exitFlag = 0;
                break;

            case 8  :
                goto insert_processing;

            case 9  :
                goto delete_processing;
            case 10 :
                fullTime += ((double) init_time / CLOCKS_PER_SEC * 1000);
                compactionFunction(device_vertex_dictionary, h_graph_prop, fullTime);
            case 11:
                triangleCount(device_vertex_dictionary, h_graph_prop);
            default :;

        }

    }

    printf("xDim = %lu, yDim = %lu, Total Edges = %lu\n", h_graph_prop -> xDim, h_graph_prop -> yDim, h_graph_prop -> total_edges);
    std::cout << "Queues: " << (float)push_to_queues_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Vertex Dictionary: " << (float)vd_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Initialization   : " << (float)init_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Adjacency List   : " << (float)al_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Delete Batch     : " << (float)delete_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "PageRank Time    : " << (float)pageRank_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Triangle Counting Time    : " << (float)triangleCounting_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "SSSP Time    : " << (float)sssp_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Prefix sum Time    : " << (float)prefix_sum_time/CLOCKS_PER_SEC << " seconds" << std::endl;

    std::cout << "Time taken: " << (float)time_req/CLOCKS_PER_SEC << " seconds" << std::endl;
    // std::cout << "Added time: " << (float)total_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Read file : " << (float)section1/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "CSR gen   : " << (float)section2/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Prefix sum, cudaMalloc, and cudaMemcpy: " << (float)section2a/CLOCKS_PER_SEC << " seconds" << std::endl;

    std::cout << std::endl << "Vertex Insert   : " << (float)vertex_insert_time/CLOCKS_PER_SEC << " seconds" << std::endl;

    // std::cout << "Prefix sum, cudaMalloc, and cudaMemcpy: " << (float)section3/CLOCKS_PER_SEC << " seconds" << std::endl;
    // // Cleanup
    // cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    // printf("%d\n", c);

    gv_end:

    return 0;
}
