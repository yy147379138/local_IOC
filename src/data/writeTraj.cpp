#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>

using namespace std;

int main(int argc, char** argv){
	double ele = -5.0;
	vector<double> x;
	vector<double> y;
	for(;ele>-13.0;ele-=0.05){
		x.push_back(ele);
		y.push_back(0.08*(ele+9.0)*(ele+9.0));
	}
	x.push_back(x.back());
	y.push_back(y.back());


	ofstream file("./lqToy.trajectory");
	file<<x.size()-1<<" ";
	for(int i=0;i<x.size();i++){
		file<<x[i]<<" "<<y[i]<<" ";
	}

    file.close();
	exit(0);
	
}
