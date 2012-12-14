#include "evidence.h"
#include "main.h"
#include "visualize.h"

#define PI 3.1415927

inline double product(pair<double,double>& v1, pair<double,double>& v2){
	return (v1.first*v2.first+v1.second*v2.second)/
				sqrt((v1.first*v1.first+v1.second*v1.second)*
							(v2.first*v2.first+v2.second*v2.second));
}

bool turnningDetector(vector<pair<double,double> >& obs, 
			int index, int prev_index, int restart){
	for (int i = index; i >= prev_index; i--){
		if(i < restart+15){
			return false;
		}
		pair<double,double> pointS = obs.at(i-5);
		pair<double,double> pointM1 = obs.at(i-2);
		pair<double,double> pointM2 = obs.at(i-1);
		pair<double,double> pointE = obs.at(i);
	
		pair<double,double> vectorSM1 = make_pair(pointM1.first-pointS.first,
					pointM1.second-pointS.second);
		pair<double,double> vectorME1 = make_pair(pointE.first-pointM1.first,
					pointE.second-pointM1.second);
		pair<double,double> vectorSM2 = make_pair(pointM2.first-pointS.first,
					pointM2.second-pointS.second);
		pair<double,double> vectorME2 = make_pair(pointE.first-pointM2.first,
					pointE.second-pointM2.second);
		if ((product(vectorSM1,vectorME1)+product(vectorSM2,vectorME2))/2 < 0.20){
			//cout<<product(vector35,vector13)<<endl; 

			return true;
		}
	}
	return false;
}


int main(int argc, char** argv){
    
	RGBTRIPLE black = {0,0,0};
	string mapfile = "./input/map.bmp";
	string evidfile = "./data/test2.sicktraj";
	BMPFile bmpMap(mapfile);
	Grid grid(bmpMap,black);

	Evidence evid(evidfile,grid);

	for(int i = 0; i < evid.size(); i++){
		vector<pair<double,double> >& Obs = evid.at_raw(i);
		//cout<<"Examine traj "<<i<<endl;
		
		int re = 0;
		bool two_before = false; // Have seen turning two steps before
		bool one_before = false;
		bool turn_now = false;

		for(int j= 0; j < Obs.size(); j++){
			turn_now = turnningDetector(Obs,j,j,re);
			if(!turn_now && !one_before && two_before){
			    re = j;	
				cout<<"Traj: "<<i+1<<" turns at: "<<1+j<<endl;
			}
			two_before = one_before;
		    one_before = turn_now;
			
		}
	}

//	evid.write("./testerOutput/traces.txt");
//	evid.writeRaw("./testerOutput/raws.txt");

	return 0;

}
