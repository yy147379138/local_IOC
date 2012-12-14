#ifndef EVIDENCE_H__
#define EVIDENCE_H__

#include "grid.h"

class Evidence {
   public:
      Evidence(string filename, Grid &_grid, int factor=1);
      Evidence(Grid &_grid) : grid(_grid) { }
      Evidence(Grid &_grid, Evidence &evid);

      vector<pair<int, int> > & at(int i) { return traces.at(i); }
      vector<pair<double, double> > & at_raw(int i) { return rawTrajs.at(i); }
      vector<double> & at_rawTime(int i) { return rawTimestamps.at(i); }
	  vector<double> & at_v(int i) {return velocities.at(i); }
	  pair<int,int> & at_bot(int i) {return robots.at(i); }
	  pair<double,double> & at_rbot(int i) {return r_robots.at(i); }

      vector<double> getTimes(int i) { return timestamps.at(i); }
      int size() { return traces.size(); }
	  int _size() { return velocities.size(); }
      void loadTrack(string filename, int factor);
      void loadTrajectory(string filename, int factor);
      void loadSICKTraj(string filename);
      void write(string filename);
      void writeRaw(string filename);
      void split(Evidence &set1, Evidence &set2, double perc);
	  void deterministicsplit(Evidence &set1, Evidence &set2);
      int getFactor() { return factor_; }
	  bool ifSickTraj(){
		  return rawTimestamps.size() > 0? 1:0;
	  }
   protected:
      int factor_;
      vector<vector<pair<int, int> > > traces;
	  // timestamp for grid traj
      vector<vector<double> > timestamps;
      vector<vector<double> > rawTimestamps;
	  vector<vector<pair<double,double> > > rawTrajs;
      Grid &grid;
	  /**Adding evidence of velocities***/
	  vector<vector<double> > velocities; 
	  vector<pair<int,int> > robots;
	  vector<pair<double,double> > r_robots;
};

void copy3DPrior(Grid &grid,int V,vector<vector<vector<double> > > &prior3D, vector<vector<double> > &prior2D);
 
void generatePrior(Grid &grid, Evidence &evid, vector<vector<double> > &prior,
      vector<double> radiusWeight=vector<double>(1,1.0), int factor=1);

void reducePrior(vector<vector<double> > &origPrior, 
      vector<vector<double> > &newPrior, int factor);

vector<int> randomPerm(int size);

#endif
