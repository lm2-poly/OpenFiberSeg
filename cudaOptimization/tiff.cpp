#include <Magick++.h> 
#include <iostream> 

#include "tiff.h"

using namespace std; 
using namespace Magick; 

void loadTiff(long*** imageStack, std::string pathTiff, int startPage,int endPage,std::tuple<int,int>& dimensions ) 
{ 
  Magick::Image imageInput;

  int columns;
  int rows;
    
  long maxPixelValue=0;

  for (int iz=0;iz<endPage-startPage;iz++){

    char filename [100];
    int countStr;

    countStr = sprintf ( filename, "V_uint16.tiff[%d]",iz );

    cout << "\nfilename: " << filename << "\n" << endl;

    imageInput.read(filename);

    columns=imageInput.columns();//yRange in my notation
    rows   =imageInput.rows();   //xRange in my notation

    cout << "columns: " << columns << "\n";
    cout << "rows   : " << rows << "\n";
    
    imageStack[iz] = new long*[rows];

    for (int ix = 0; ix < rows; ix++) {
        imageStack[iz][ix] = new long[columns];
    }

    for (int ix = 0; ix < rows; ix++) {
        for (int iy = 0; iy < columns; iy++) {
            imageStack[iz][ix][iy] = imageInput.pixelColor(iy,ix).quantumBlue();
        }
    }

  }


  get<0>(dimensions)=rows;
  get<1>(dimensions)=columns;

}


int mainDummy(int argc,char **argv){

  InitializeMagick(*argv);

  std::string pathTiff="./";
  int startPage=0;
  int endPage=4;

  int pages=endPage-startPage;

  long*** imageStack=new long**[pages];

  std::tuple<int,int> dimensions;

  loadTiff(imageStack, pathTiff, startPage, endPage,dimensions );

  int rows=get<0>(dimensions);
  int columns=get<1>(dimensions);

  printf("rows: %d, columns: %d",rows,columns);

  uint32_t ARRAY_SIZE= rows*columns*pages;


  for (int iz=startPage;iz<endPage;iz++){
    for (int ix = 0; ix < rows; ix++) {
        delete[] imageStack[iz][ix];
    }
    delete[] imageStack[iz];
  }

  delete[ ] imageStack;

  return 0;
}