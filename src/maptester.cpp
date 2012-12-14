#include "evidence.h"
#include "main.h"
#include "visualize.h"


int main(int argc, char** argv){
    
	RGBTRIPLE black = {0,0,0};
	string mapfile = "./input/map.bmp";
	string evidfile = "./data/codetester.sicktraj";
	BMPFile bmpMap(mapfile);
	Grid grid(bmpMap,black);
	Evidence evid(evidfile,grid);

	pair<int,int> dims = grid.dims();
    BMPFile gridView(dims.first,dims.second);

	char buf[512];

	grid.addObstacles(gridView,black);
//	vector<pair<int,int> > path;
//	for(int i=0;i<50;i++){
//		path.push_back(make_pair(i*3,i));
//	}
	gridView.addVector(evid.at(0),blue,1);
	sprintf(buf,"./testerOutput/tester.bmp");
    gridView.write(buf);

	pair<double,double> check = 
		grid.grid2Real(evid.at(0).at(1).first,
					evid.at(0).at(1).second);
	
	cout<<check.first<<" "<<check.second<<endl; 

	return 0;

}
