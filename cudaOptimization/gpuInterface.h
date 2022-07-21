# by Facundo Sosa-Rey, 2021. MIT license
// see https://github.com/nosferalatu/SimpleGPUHashTable for hashtable implementation

#pragma once
#include <string>


struct KeyValue
{
    uint32_t key;
    uint32_t value;
};

const uint32_t kHashTableCapacity = 128 * 1024 * 1024;

const uint32_t kNumKeyValues = kHashTableCapacity / 2;

const uint32_t kEmpty = 0xffffffff;

KeyValue* create_hashtable();

void insert_hashtable(KeyValue* hashtable, const KeyValue* kvs, uint32_t num_kvs);

void lookup_hashtable(KeyValue* hashtable, KeyValue* kvs, uint32_t num_kvs);

void lookup_hashtable_single_query(KeyValue* hashtable, KeyValue* kvs);

void lookup_hashtable_multiple_query(KeyValue* hashtable, KeyValue* kvs, uint32_t num_kvs);

void lookup_hashtable_on_array(KeyValue* hashTable, uint32_t* h_array, uint32_t ARRAY_SIZE, uint32_t ARRAY_BYTES);

void delete_hashtable(KeyValue* hashtable, const KeyValue* kvs, uint32_t num_kvs);

std::vector<KeyValue> iterate_hashtable(KeyValue* hashtable);

void destroy_hashtable(KeyValue* hashtable);

// tiff.cpp
void loadTiff(long*** imageStack, std::string pathTiff, int startPage,int endPage,std::tuple<int,int>& dimensions ) ;

