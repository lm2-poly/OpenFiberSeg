// by Facundo Sosa-Rey, 2021. MIT license

// see https://github.com/nosferalatu/SimpleGPUHashTable for hashtable implementation

#include "stdio.h"
#include "stdint.h"
#include "vector"
#include "gpuInterface.h"
#include "iostream"

// 32 bit Murmur3 hash
__device__ uint32_t hash(uint32_t k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (kHashTableCapacity - 1);
}

// Create a hash table. For linear probing, this is just an array of KeyValues
KeyValue *create_hashtable()
{
    // Allocate memory
    KeyValue *hashtable;
    cudaMalloc(&hashtable, sizeof(KeyValue) * kHashTableCapacity);

    // Initialize hash table to empty
    static_assert(kEmpty == 0xffffffff, "memset expected kEmpty=0xffffffff");
    cudaMemset(hashtable, 0xff, sizeof(KeyValue) * kHashTableCapacity);

    return hashtable;
}

// Insert the key/values in kvs into the hashtable
__global__ void gpu_hashtable_insert(KeyValue *hashtable, const KeyValue *kvs, unsigned int numkvs)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < numkvs)
    {
        uint32_t key = kvs[threadid].key;
        uint32_t value = kvs[threadid].value;
        uint32_t slot = hash(key);

        while (true)
        {
            uint32_t prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
            if (prev == kEmpty || prev == key)
            {
                hashtable[slot].value = value;
                return;
            }

            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}

void insert_hashtable(KeyValue *pHashTable, const KeyValue *kvs, uint32_t num_kvs)
{
    // Copy the keyvalues to the GPU
    KeyValue *device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * num_kvs);
    cudaMemcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice);

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0);

    // Create events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Insert all the keys into the hash table
    int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;
    gpu_hashtable_insert<<<gridsize, threadblocksize>>>(pHashTable, device_kvs, (uint32_t)num_kvs);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    printf("    GPU inserted %d items in %f ms (%f million keys/second)\n",
           num_kvs, milliseconds, num_kvs / (double)seconds / 1000000.0f);

    cudaFree(device_kvs);
}

// Looks up the  keys in the hashtable, and return the values
__global__ void gpu_hashtable_lookup(KeyValue* hashtable, KeyValue* kvs)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < kHashTableCapacity)
    {
        uint32_t key = kvs[threadid].key;
        uint32_t slot = hash(key);

        while (true)
        {
            if (hashtable[slot].key == key)
            {
                kvs[threadid].value = hashtable[slot].value;
                return;
            }
            if (hashtable[slot].key == kEmpty)
            {
                kvs[threadid].value = kEmpty;
                return;
            }
            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}


// Looks up the  keys in the hashtable, and return the values
__global__ void gpu_hashtable_lookup_onArray(KeyValue* hashtable, uint32_t* d_array )
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < kHashTableCapacity)
    {
        uint32_t key = d_array[threadid];
        uint32_t slot = hash(key);

        while (true)
        {
            if (hashtable[slot].key == key)
            {
                d_array[threadid] = hashtable[slot].value;
                return;
            }
            if (hashtable[slot].key == kEmpty)
            {
                d_array[threadid] = kEmpty;
                return;
            }
            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}

void lookup_hashtable_single_query(KeyValue *hashTable, KeyValue *kvs_query)
{
    // Copy the single keyvalue to the GPU
    KeyValue *device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) );
    cudaMemcpy(device_kvs, kvs_query, sizeof(KeyValue) , cudaMemcpyHostToDevice);

    gpu_hashtable_lookup<<<1, 1>>>(hashTable, device_kvs);

    cudaMemcpy(kvs_query, device_kvs, sizeof(KeyValue) , cudaMemcpyDeviceToHost);

    cudaFree(device_kvs);
}

void lookup_hashtable_multiple_query(KeyValue* hashTable, KeyValue* kvs_query, uint32_t num_kvs)
{
    // Copy keyvalues to the GPU
    KeyValue *device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue)*num_kvs );
    cudaMemcpy(device_kvs, kvs_query, sizeof(KeyValue)*num_kvs , cudaMemcpyHostToDevice);

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_lookup, 0, 0);

    int myGridSize=num_kvs/threadblocksize+1;

    printf("\ncall to: gpu_hashtable_lookup \n myGridSize: %d\nthreadblocksize: %d\n\n\n",myGridSize, threadblocksize);

    gpu_hashtable_lookup<<<myGridSize, threadblocksize>>>(hashTable, device_kvs);

    cudaMemcpy(kvs_query, device_kvs, sizeof(KeyValue)*num_kvs , cudaMemcpyDeviceToHost);

    cudaFree(device_kvs);
}

void lookup_hashtable_on_array(KeyValue* hashTable, uint32_t* h_array,uint32_t ARRAY_SIZE, uint32_t ARRAY_BYTES)
{
    //copy marker array from host to device
    uint32_t* d_array;
    cudaMalloc(&d_array, ARRAY_BYTES );
    cudaMemcpy(d_array, h_array, ARRAY_BYTES , cudaMemcpyHostToDevice);

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_lookup_onArray, 0, 0);

    int myGridSize=ARRAY_SIZE/threadblocksize+1;

    printf(
        "\ncall to: gpu_hashtable_lookup_onArray \nmyGridSize: %d\nthreadblocksize: %d \nARRAY_SIZE: %d\n\n\n",
        myGridSize, 
        threadblocksize,
        ARRAY_SIZE
        );

    // replace marker keys with corresponding values in the hashtable 
    gpu_hashtable_lookup_onArray<<<myGridSize, threadblocksize>>>(hashTable, d_array);

    cudaMemcpy(h_array, d_array, ARRAY_BYTES , cudaMemcpyDeviceToHost);

    cudaFree(d_array);
}


// Delete each key in kvs from the hash table, if the key exists
// A deleted key is left in the hash table, but its value is set to kEmpty
// Deleted keys are not reused; once a key is assigned a slot, it never moves
__global__ void gpu_hashtable_delete(KeyValue *hashtable, const KeyValue *kvs, unsigned int numkvs)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < kHashTableCapacity)
    {
        uint32_t key = kvs[threadid].key;
        uint32_t slot = hash(key);

        while (true)
        {
            if (hashtable[slot].key == key)
            {
                hashtable[slot].value = kEmpty;
                return;
            }
            if (hashtable[slot].key == kEmpty)
            {
                return;
            }
            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}

void delete_hashtable(KeyValue *pHashTable, const KeyValue *kvs, uint32_t num_kvs)
{
    // Copy the keyvalues to the GPU
    KeyValue *device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * num_kvs);
    cudaMemcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice);

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0);

    // Create events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Insert all the keys into the hash table
    int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;
    gpu_hashtable_delete<<<gridsize, threadblocksize>>>(pHashTable, device_kvs, (uint32_t)num_kvs);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    printf("    GPU delete %d items in %f ms (%f million keys/second)\n",
           num_kvs, milliseconds, num_kvs / (double)seconds / 1000000.0f);

    cudaFree(device_kvs);
}

// Iterate over every item in the hashtable; return non-empty key/values
__global__ void gpu_iterate_hashtable(KeyValue *pHashTable, KeyValue *kvs, uint32_t *kvs_size)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < kHashTableCapacity)
    {
        if (pHashTable[threadid].key != kEmpty)
        {
            uint32_t value = pHashTable[threadid].value;
            if (value != kEmpty)
            {
                uint32_t size = atomicAdd(kvs_size, 1);
                kvs[size] = pHashTable[threadid];
            }
        }
    }
}

// Notes: this one creates an array device_kvs of max size (kNumKeyValues).
// the kernel iterates over entire passed hashtable, and returns non-empty keys
std::vector<KeyValue> iterate_hashtable(KeyValue *pHashTable)
{
    uint32_t *device_num_kvs;
    cudaMalloc(&device_num_kvs, sizeof(uint32_t));
    cudaMemset(device_num_kvs, 0, sizeof(uint32_t));

    KeyValue *device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * kNumKeyValues);

    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_iterate_hashtable, 0, 0);

    int gridsize = (kHashTableCapacity + threadblocksize - 1) / threadblocksize;
    gpu_iterate_hashtable<<<gridsize, threadblocksize>>>(pHashTable, device_kvs, device_num_kvs);

    uint32_t num_kvs;
    cudaMemcpy(&num_kvs, device_num_kvs, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<KeyValue> kvs;
    kvs.resize(num_kvs);

    cudaMemcpy(kvs.data(), device_kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyDeviceToHost);

    cudaFree(device_kvs);
    cudaFree(device_num_kvs);

    return kvs;
}

// Free the memory of the hashtable
void destroy_hashtable(KeyValue *pHashTable)
{
    cudaFree(pHashTable);
}
