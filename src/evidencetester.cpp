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

	evid.write("./testerOutput/traces.txt");
	evid.writeRaw("./testerOutput/raws.txt");

	return 0;

}
