// by Facundo Sosa-Rey, 2021. MIT license

#include <iostream>
#include <fstream>

#include "nlohmann/json.hpp"
#include "vector"

#include "gpuInterface.h"

#include <Magick++.h> 


using json = nlohmann::json;
using namespace std; 
using namespace Magick; 


//hash table implemented for uint32_t. this offsets the float value so some precision can be 
//kept when converting float to int. (further precision loss is inconsequential)
const float scalarOffset=1E6; 

uint32_t loadMarkersFromJSON(string filename, vector<KeyValue>& insert_kvs){

  uint32_t currentMarker=0;
  string keyStr,valueStr;
  ifstream ifs;

  ifs.open(filename, ios_base::in);
  
  if (!ifs.is_open()) {

    cout<<"\n The file could not be opened"<<endl;
    exit(1);
  }

  json data = json::parse(ifs);

  ifs.close();


  bool endOfFile=false;

  while(!endOfFile){

    cout << "to_string(currentMarker): " << to_string(currentMarker) <<endl;

    keyStr="/"+to_string(currentMarker);
  
    cout << "keyStr : " << keyStr << endl;

    json::json_pointer pJ(keyStr);

    auto valueFl=data[pJ];

    cout << "marker_to_propertyLUT[" <<currentMarker <<"]: " << valueFl << endl;

    // if (currentMarker==512){
    if (to_string(valueFl)=="null"){

      endOfFile=true;

    }else{

      KeyValue tempKeyValue={currentMarker,uint32_t(float(valueFl)*scalarOffset)};

      insert_kvs.push_back({tempKeyValue});
      currentMarker++;
    }
  }

  return currentMarker;
}



int main(int argc,char **argv){
  InitializeMagick(*argv);

  string filename="marker_to_propertyLUT.json";

  std::vector<KeyValue> insert_kvs,query_kvs;

  uint32_t currentMarker=loadMarkersFromJSON(filename, insert_kvs);


  cout << "\n\nfirst in insert_kvs: " << insert_kvs.front().key << " , "<< float(insert_kvs.front().value)/scalarOffset << endl;
  cout << "2nd in insert_kvs: " << insert_kvs[2].key    << " , "<< float(insert_kvs[2].value)/scalarOffset <<endl;
  cout << "last in insert_kvs: " << insert_kvs.back().key    << " , "<< float(insert_kvs.back().value)/scalarOffset << "\n\n"<<endl;

  cout<<"starting insertion"<<endl;


  /*
  //this is to make a dummy array with contiguous markers as a test

  const uint32_t ARRAY_SIZE =currentMarker;
  // const uint32_t ARRAY_SIZE =1024;
  const uint32_t ARRAY_BYTES = ARRAY_SIZE * sizeof(uint32_t);

  printf("ARRAY_SIZE= %d",ARRAY_SIZE);
  
  // generate the input array on the host
  uint32_t* h_in=new uint32_t[ARRAY_SIZE];

  for (int i = 0; i < ARRAY_SIZE; i++) {
    h_in[i] = i;
  }
  */

  // load image stack

  std::string pathTiff="./";
  std::string pathTiff_output="./output/";


  int startPage=atoi(argv[1]);
  int endPage  =atoi(argv[2]);

  int pages=endPage-startPage;

  long*** imageStack=new long**[pages];

  std::tuple<int,int> dimensions;

  loadTiff(imageStack, pathTiff, startPage, endPage,dimensions );

  int rows=get<0>(dimensions);
  int columns=get<1>(dimensions);

  printf("rows: %d, columns: %d",rows,columns);

  uint32_t index1D;
  const uint32_t ARRAY_SIZE= rows*columns*pages;
  const uint32_t ARRAY_BYTES = ARRAY_SIZE * sizeof(uint32_t);

  //serialize to 1D array for simplicity

  cout << "Serialization begins" << endl;

  // generate the input array on the host
  uint32_t* h_in=new uint32_t[ARRAY_SIZE];

  for (int iz=0;iz<endPage-startPage;iz++){
    for (int ix = 0; ix < rows; ix++) {
      for (int iy = 0; iy < columns; iy++) {
        
        index1D= iz*rows*columns + ix*columns + iy;

        h_in[index1D]=imageStack[iz][ix][iy];

      }
    }
  }

  KeyValue* pHashTable = create_hashtable(); //TODO add check for hash table size vs marker count


  // Insert items into the hash table
  const uint32_t num_insert_batches = std::min(16,(int)ARRAY_SIZE); // emtpy inserts (for small ARRAY_SIZE) will cause silent crash
  uint32_t num_inserts_per_batch = (uint32_t)insert_kvs.size() / num_insert_batches;

  for (uint32_t i = 0; i < num_insert_batches; i++)
  {
      insert_hashtable(pHashTable, insert_kvs.data() + i * num_inserts_per_batch, num_inserts_per_batch);
  }

  uint32_t num_keys_query=16;

  std::vector<KeyValue> found_kvs;

  cout << "\n\n single lookup, sequential: \n" << endl;


  for (int i=0;i<num_keys_query;i++){
      query_kvs.push_back(insert_kvs[i]);
      query_kvs.back().value=999999; //assign dummy value to test retrieval from hashtable on gpu
              
      lookup_hashtable_single_query(pHashTable, query_kvs.data()+i);    
  }

  for (int i=0;i<num_keys_query;i++){

      cout << "items in found_kvs: " << query_kvs[i].key <<" , "<< query_kvs[i].value/scalarOffset;
      cout << "\tshould be: " << insert_kvs[i].value/scalarOffset << endl;

  }

  // attempting multiple lookups:

  // resetting query values

  query_kvs.clear();

  for (int i=0;i<currentMarker;i++){
      query_kvs.push_back(insert_kvs[i]);
      query_kvs[i].value=999999; //assign dummy value to test retrieval from hashtable on gpu            
  }


  cout << "\n\n multiple lookups in parallel: \n" << endl;

  lookup_hashtable_multiple_query(pHashTable, query_kvs.data(),num_keys_query);

  for (int i=0;i<num_keys_query;i++){

      cout << "items in found_kvs: " << query_kvs[i].key <<" , "<< query_kvs[i].value/scalarOffset;
      cout << "\tshould be: " << insert_kvs[i].value/scalarOffset << endl;

  }

  cout << "\n\n lookup directly on array, parallel: \n" << endl;

  uint32_t num_lookup_per_batch = ARRAY_SIZE / num_insert_batches;

  for (uint32_t i = 0; i < num_insert_batches; i++)
  {
    insert_hashtable(
      pHashTable, 
      insert_kvs.data() + i * num_inserts_per_batch, 
      num_inserts_per_batch
      );
  }

  for (uint32_t i = 0; i < num_insert_batches; i++)
  {
    lookup_hashtable_on_array(
      pHashTable, 
      &h_in[i*num_lookup_per_batch], 
      num_lookup_per_batch, 
      num_lookup_per_batch*sizeof(uint32_t)
      );
  }

  // print out the resulting array
  for (int i =ARRAY_SIZE-5000; i < ARRAY_SIZE; i++) {
    printf("%.4f", float(h_in[i])/scalarOffset);
    printf(((i % 16) != 15) ? "\t" : "\n");
  }

  destroy_hashtable(pHashTable);

  for (int iz=startPage;iz<endPage;iz++){
    for (int ix = 0; ix < rows; ix++) {
        delete[] imageStack[iz][ix];
    }
    delete[] imageStack[iz];
  }

  delete[ ] imageStack; 


}