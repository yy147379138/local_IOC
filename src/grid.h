#ifndef GRID_H__
#define GRID_H__

#include <string>
#include <vector>
#include "visualize.h"

using namespace std;

class Grid {
   public:
      Grid(string filename);
      Grid(BMPFile &bitmap, RGBTRIPLE rgb);
      int at(int x, int y) { return values.at(x).at(y); }
	  /* converting function for Intel data */
      pair<int, int> realToGrid(double x, double y, int factor=1);
	  /* converting function for NSH data */
      pair<int, int> real2Grid(double x, double y);
	  /* converting function for Intel data */
      pair<double, double> gridToReal(int x, int y, int factor=1);
	  /* converting function for NSH data */
      pair<double, double> grid2Real(int x, int y);
      pair<int, int> dims() {  return dimens; }
      void addObstacles(BMPFile &bitmap, RGBTRIPLE rgb); 
   protected:
      vector<vector<int> > values;
      pair<double, double> center;
	  pair<double, double> corner;
      pair<int, int> dimens;
      double scale;
};

double LogAdd(double a, double b); 
double LogSubtract(double a, double b); 
#endif
