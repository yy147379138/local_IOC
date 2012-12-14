#include "evidence.h"
#include "main.h"
#include "visualize.h"
#include "linearquadratic.h"

int main(int argc, char** argv){
    
	RGBTRIPLE black = {0,0,0};
	string mapfile = "./input/map.bmp";
	string evidfile = "./data/codetester.sicktraj";
	BMPFile bmpMap(mapfile);
	Grid grid(bmpMap,black);

	Evidence evid(evidfile,grid);

	vector<VectorXd> states;
    ContinuousState cs;
	cs.convertState(states,evid.at_raw(0),
			evid.at_raw(0).back());
	ofstream file("./testerOutput/states.txt");
	for(int i=0;i<states.size();i++){
		file<<"State "<<i<<endl
			<<" "<<states.at(i)<<endl;
	}
	file.close();

	return 0;

}
