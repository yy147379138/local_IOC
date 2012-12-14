#include "grid.h"
#include <iostream>
#include <fstream>
#include "main.h"

double LogAdd(double a, double b) {
   if (a <= -HUGE_VAL) return b;
   if (b <= -HUGE_VAL) return a;
   double minexp = min(a, b);
   double maxexp = max(a,b);
   return (maxexp + log(1+exp(-maxexp+minexp)));
} 



double LogSubtract(double a, double b) {
   if (a==b) return -HUGE_VAL;
   if (a<b) cerr<<"Value a should be larger than value b"<<endl;
   if (b <= -HUGE_VAL) return a;

} 

Grid::Grid(string filename) : values(0) {
   ifstream file(filename.c_str());

   char buf[1024*64];

   while (file.getline(buf, 1024*64) > 0) {
      vector<int> valueVec;
      convert(string(buf), " ", valueVec);
     
      if (values.size() == 0) 
         values.resize(valueVec.size(), vector<int>(0)); 

      for (int i=0; i < valueVec.size(); i++) 
         values.at(i).push_back(valueVec.at(i)); 
   }

   center.first = -3.82;
   center.second = -1.93;

   scale = 0.04;

   cout << "Grid loaded: "<<values.size()<<"  "<<values.at(0).size()<<endl; 

   cout << "Value Check: "<<values.back().back()<<endl;

   dimens.first = values.size();
   dimens.second = values.at(0).size();
}

Grid::Grid(BMPFile &bmp, RGBTRIPLE rgbObs) {

   pair<int, int> bmpDims = bmp.getDims();

   values.resize(bmpDims.first, vector<int>(bmpDims.second, 0));

   for (int i=0; i < bmpDims.first; i++) {
      for (int j=0; j < bmpDims.second; j++) { 
         RGBTRIPLE rgb = bmp.getPixel(i,j);
         if (rgb == rgbObs)
            values.at(i).at(j) = 1; 
      }
   }

   center.first = -3.82;
   center.second = -1.93;
   corner.first = -2.7; //minimum x coord
   corner.second = -2.0; //minimum u coord
   scale = 0.04;
   dimens = bmpDims; 
}

pair<int, int> Grid::realToGrid(double x, double y, int factor) {
   pair<int, int> point;

   point.first = ((int)floor((x-center.first)/scale+ceil(dimens.first/2)))
      /factor;
   point.second = (dimens.second - 
      (int)floor((y-center.second)/scale+ceil(dimens.second/2)))
      /factor;

   return point;
}

pair<int, int> Grid::real2Grid(double x, double y) {
   pair<int, int> point;
   double rot = sqrt(2)/2;

   double granularity = 20;//Width/(max-min)
   point.first = (int)floor((x*rot-y*rot-corner.first)*granularity);
   point.second = dimens.second - 
      (int)floor((x*rot+y*rot-corner.second)*granularity);

   return point;
}

pair<double, double> Grid::gridToReal(int x, int y, int factor) {
   pair<double, double> point;

   point.first = (x*factor-ceil(dimens.first/2))*scale+center.first;
   point.second = (dimens.second-y*factor-ceil(dimens.second/2))*scale
	   +center.second;
   return point;
}


pair<double, double> Grid::grid2Real(int x, int y) {
   pair<double, double> point;
   if(dimens.first==210&&dimens.second==250){
		double rot = sqrt(2)/2;
		double granularity = 20;
		double tempx = x/granularity+corner.first;
		double tempy = (dimens.second-y)/granularity+corner.second;
		point.first = (tempx+tempy)/rot/2;
		point.second = (tempy-tempx)/rot/2;
   }else{
		point.first = (x-ceil(dimens.first/2))*scale+center.first;
		point.second = (dimens.second-y-ceil(dimens.second/2))*scale
				+center.second;
   }
   return point;
}


void Grid::addObstacles(BMPFile &bitmap, RGBTRIPLE rgb) {
   for (int i=0; i < dimens.first; i++) {
      for (int j=0; j < dimens.second; j++) {
          if (values.at(i).at(j) == 1)
             bitmap.setPixel(i,j,rgb);
      } 
   } 
}

#if 1 
void convert(string str, string delim, vector<int> &results) {
   int next;
   char buf[20];
   while ( (next=str.find_first_of(delim)) != str.npos) {
      if (next > 0) 
         results.push_back(atoi(str.substr(0,next).c_str()));
         
      str = str.substr(next+1); 
   }
} 

void convert(string str, string delim, vector<double> &results) {
   int next;
   char buf[20];
   while ( (next=str.find_first_of(delim)) != str.npos) {
      if (next > 0) 
         results.push_back(atof(str.substr(0,next).c_str()));
         
      str = str.substr(next+1); 
   }
   results.push_back(atof(str.c_str()));
} 
#endif


