#include <iostream>
#include <cstdlib>
#include <math.h>

using namespace std;


pair<int,int> dimens(490,321);
pair<double,double> center(-3.82,-1.93);
double scale = 0.04;


pair<int, int> realToGrid(double x, double y, int factor) {
   pair<int, int> point;

   point.first = ((int)floor((x-center.first)/scale+ceil(dimens.first/2)))
      /factor;
   point.second = (dimens.second - 
      (int)floor((y-center.second)/scale+ceil(dimens.second/2)))
      /factor;

   return point;
}

pair<double, double> gridToReal(int x, int y, int factor) {
   pair<double, double> point;

   point.first = (x*factor-ceil(dimens.first/2))*scale+center.first;
   point.second = (dimens.second-y*factor-ceil(dimens.second/2))*scale
	   +center.second;
   return point;
}


int main(int argc, char** argv){
  pair<double,double> orig_r(0.98,-2.3);
  pair<int,int> g = realToGrid(orig_r.first,orig_r.second,1);
  pair<double,double> check = gridToReal(g.first,g.second,1);
  cout<<"Original: "<<orig_r.first<<" "<<orig_r.second
	  <<" Grid: "<<g.first<<" "<<g.second
	  <<" Check: "<<check.first<<" "<<check.second<<endl;
  return 0;
}
