#include <map>
#include <algorithm>
#include <cassert>
#include "main.h"
#include "evidence.h"
#include "grid.h"

Evidence::Evidence(string filename, Grid &_grid, int factor) 
   : grid(_grid), factor_(factor) {

   if (strstr(filename.c_str(), ".trajectory")) {
      cout << "   .trajectory file: "<<filename.c_str()<<endl;
      loadTrajectory(filename, factor);
   }
   else if (strstr(filename.c_str(), ".track")) {
      cout << "Loading .track file: "<<filename.c_str()<<endl;
      loadTrack(filename, factor);
   }
   else if (strstr(filename.c_str(), ".sicktraj")) {
      cout << "Loading .sicktraj file: "<<filename.c_str()<<endl;
      loadSICKTraj(filename);
   }
   else {
      cout << "Unexpected file format: "<<filename<<endl;
      cout << "Expected either .trajectory or .track"<<endl;
      exit(0);
   }
}

void Evidence::loadTrack(string filename, int factor) {
   ifstream file(filename.c_str());

   char buf[1024*64];

   map<int, vector<double> > personData;

   int lines=0;

   while (file.getline(buf, 1024*64) > 0) {
      double tick, x, y, dx, dy;
      int uid;
      sscanf(buf, "%lf,%d,%lf,%lf,%lf,%lf", &tick, &uid, &y, &x, &dy, &dx);

      personData[uid].push_back(tick);
      personData[uid].push_back(x);
      personData[uid].push_back(y-3.82);

      lines++;
   }

   map<int, vector<double> >::iterator iter = personData.begin();

   int tracks = 0; 
   int skipped = 0;

   while (iter != personData.end()) {

      cout << "TRACK:"<<endl;

      vector<double> rawData = iter->second;

      vector<pair<int, int> > trace;
      vector<double> timestamp;

      for (int i=0; i*3 < rawData.size(); i+=3) {
         double tick = rawData.at(i);
         double x = rawData.at(i+1);
         double y = rawData.at(i+2);

         pair<int, int> point = grid.realToGrid(x, y, factor);

         cout <<point.first<<"  "<<point.second<<"   "<<x<<"  "<<y<<endl;


         if (trace.size() > 0) {
            int x1 = trace.back().first;
            int y1 = trace.back().second;
            int x2 = point.first;
            int y2 = point.second;
            int dist1 = (int)fabs(x2 - x1);
            int dist2 = (int)fabs(y2 - y1);

            //cout << "      "<<dist1<<" "<<dist2<<endl;

            // need to add extra points here
            if (dist1 > 1 || dist2 > 1) {
               //cout << "JOIN: "<<x1<<" "<<y1<<"   "<<x2<<" "<<y2<<endl; 


               int dist = max(dist1, dist2);


               //cout << "CHECK: "<<dist<<" "
               //   << (float)(x2-x1)/dist<<"  "<<(float)(y2-y1)/dist<<endl;

               double stepX = (x2-x1) != 0 ? ((float)(x2-x1)/(float)dist) : 0;
               double stepY = (y2-y1) != 0 ? ((float)(y2-y1)/(float)dist) : 0;

               //cout << "STEPS: "<<stepX<<" "<<stepY<<endl;

               for (int d=1; d <= dist; d++) {
                  //cout << "PT: "<<floor((float)x1+(float)d*stepX+.5)
                  //   <<"  "<< floor((float)y1+(float)d*stepY+.5)<<endl;
                  point.first = (int)floor((float)x1+(float)d*stepX+.5);
                  point.second = (int)floor((float)y1+(float)d*stepY+.5);
                  if (trace.back().first == point.first && 
                        trace.back().second == point.second) continue; 
                  trace.push_back(point);
                  cout << "    INSERTING: "<<point.first
                     << " "<<point.second<<endl;
               }
            }
            else if (dist1 == 0 && dist2 == 0) {
               double dT = tick-timestamp.back();
               if (dT > 0.3) {
                  cout << "ABORT! "<<dT<<endl;
                  skipped++;
                  goto skipToNext;
               }

               continue;
            }
            else {
               trace.push_back(point); 
               
               if (timestamp.size() > 0) {
                  double dT = tick-timestamp.back();
                  cout << "dT = "<<dT<<endl; 
               }
               timestamp.push_back(tick);

            }

         }
         else {
            trace.push_back(point);
            timestamp.push_back(tick);
         }
      }
      traces.push_back(trace); 
      tracks++;
skipToNext:
      iter++;
   }
   cout << tracks << " tracks ("<<skipped<<" skipped) in " << lines 
      << " lines processed"<<endl; 

}

void Evidence::loadTrajectory(string filename, int factor) {
   ifstream file(filename.c_str());

   int size;

   // change this if for observation time interval
   double deltat = 0.0267;

   char buf[1024*64];

   pair<int, int> dims = grid.dims();

   while (file.getline(buf, 1024*64) > 0) {
      vector<double> valueVec;
      convert(string(buf), " ", valueVec);
      vector<pair<int, int> > trace;
	  vector<pair<double,double> > raw;
      vector<double> timestamp;
	  vector<double> velocity;
	  pair<double,double> robot_real(valueVec.at(valueVec.size()-2),
				  valueVec.at(valueVec.size()-1));
	  pair<int,int> robot_grid = grid.realToGrid(valueVec.at(valueVec.size()-2),
				  valueVec.at(valueVec.size()-1),factor);
	  valueVec.pop_back();
	  valueVec.pop_back();
	  if (valueVec.size()<20)
	//	continue;
	  if (robot_grid.first < 0 || robot_grid.second < 0 
               || robot_grid.first >= dims.first || robot_grid.second >= dims.second){
	        cout<<"abandon"<<endl;
            continue;
	  }


      for (int i=0; (2*i+2) < valueVec.size(); i++) {
		 pair<double,double> coord(valueVec.at(2*i+1),valueVec.at(2*i+2));
		 //cout<<"x "<<coord.first<<" y: "<<coord.second<<endl; 
         raw.push_back(coord);
         pair<int, int> point = grid.realToGrid(
               valueVec.at(2*i+1), valueVec.at(2*i+2), factor);
		 double v;
/* 
 * The current assumption of the tracking data is that the time interval is the same 
 * and is set to 0.0267. Later when the discrete points are added, the timestamps 
 * are generated for each interpolation point. Potential contradiction may exist.
 * Need to be examed later.
 * The velocities for points interpolated between two actual data points are assumed 
 * the same.
 */
		 if (i > 0 && (2*i+2+2) < valueVec.size()){
			  v  = (valueVec.at(2*i+3)-valueVec.at(2*i+1))*
				 (valueVec.at(2*i+3)-valueVec.at(2*i+1))+
				 (valueVec.at(2*i+4)-valueVec.at(2*i+2))*
			 (valueVec.at(2*i+4)-valueVec.at(2*i+2));
			 v  = sqrt(v)/deltat;
		 }

         if (point.first < 0 || point.second < 0 
               || point.first >= dims.first || point.second >= dims.second)
            continue;
         //cout << "POINT: "<<point.first<<" "<<point.second<<endl;

         if (trace.size() > 0) {
            int x1 = trace.back().first;
            int y1 = trace.back().second;
            int x2 = point.first;
            int y2 = point.second;
            int dist1 = (int)fabs(x2 - x1);
            int dist2 = (int)fabs(y2 - y1);
            // need to add extra points heresplit
            if (dist1 > 1 || dist2 > 1) {
               //cout << "JOIN: "<<x1<<" "<<y1<<"   "<<x2<<" "<<y2<<endl; 

              
               int dist = max(dist1, dist2);


               //cout << "CHECK: "<<dist<<" "
               //   << (float)(x2-x1)/dist<<"  "<<(float)(y2-y1)/dist<<endl;

               double stepX = (x2-x1) != 0 ? ((float)(x2-x1)/(float)dist) : 0;
               double stepY = (y2-y1) != 0 ? ((float)(y2-y1)/(float)dist) : 0;

               //cout << "STEPS: "<<stepX<<" "<<stepY<<endl;
                
               for (int d=1; d <= dist; d++) {
                  //cout << "PT: "<<floor((float)x1+(float)d*stepX+.5)
                  //   <<"  "<< floor((float)y1+(float)d*stepY+.5)<<endl;
                  point.first = (int)floor((float)x1+(float)d*stepX+.5);
                  point.second = (int)floor((float)y1+(float)d*stepY+.5);
                  //cout << "PT: "<<x1 <<"  "<<y1<<"  "<<point.first
                  //   <<"  "<<point.second<<endl;
                  if (trace.back().first == point.first && 
                        trace.back().second == point.second) continue;
				  // push v
				  velocity.push_back(v);
                  trace.push_back(point);
                  timestamp.push_back(i*0.0267);
                  //cout << "added: "<<trace.back().first<<"  "
                  //   <<trace.back().second<<endl;
               }
            }else if (dist1 == 0 && dist2 == 0){ 
               continue;
			}else {
			   // push v
			   velocity.push_back(v);
               trace.push_back(point);
               timestamp.push_back(i*0.0267);
            }
         }
         if (trace.size() > 0 && trace.back().first == point.first &&
               trace.back().second == point.second) continue;
         // push v
		 velocity.push_back(v);
		 trace.push_back(point);
         timestamp.push_back(i*0.0267);
      }
      // check the length of the trajectory and the velocity sequence
	  if (trace.size()!=velocity.size()){
	     cout<<"traj length not equal to vel length"<<endl;
	  }
      velocities.push_back(velocity);
	  traces.push_back(trace);
      timestamps.push_back(timestamp);
	  robots.push_back(robot_grid);
	  r_robots.push_back(robot_real);
	  rawTrajs.push_back(raw);
      //cout<<"check    end:"<<trace.back().first<<" "<<trace.back().second<<" rob: "<<robot_grid.first<<" "<<robot_grid.second<<endl;
   }

   cout << "   loaded: "<<traces.size()<<" trajs "<<velocities.size()<<" vel seqs "
	   <<robots.size()<<" robot positions "<<rawTrajs.size()<<" raw trajs "<<
	   r_robots.size()<<" real-value robots pos"<<endl;
}


void Evidence::loadSICKTraj(string filename){
   ifstream file(filename.c_str());

   int size;
   char buf[1024*64];

   pair<int, int> dims = grid.dims();

   while (file.getline(buf, 1024*64) > 0) {
      vector<double> valueVec;
      convert(string(buf), " ", valueVec);
      vector<pair<int, int> > trace;
	  vector<pair<double,double> > raw;
      vector<double> timestamp;
      vector<double> rawTimestamp;
	  vector<double> velocity;
	  pair<double,double> robot_real(valueVec.at(valueVec.size()-2),
				  valueVec.at(valueVec.size()-1));
	  pair<int,int> robot_grid = grid.real2Grid(valueVec.at(valueVec.size()-2),
				  valueVec.at(valueVec.size()-1));
	  valueVec.pop_back();
	  valueVec.pop_back();
	  if (robot_grid.first < 0 || robot_grid.second < 0 
               || robot_grid.first >= dims.first || robot_grid.second >= dims.second){
	        cout<<"abandon"<<endl;
            continue;
	  }

      for (int i=0; (3*i+3) < valueVec.size(); i++) {
		 double tick = valueVec.at(3*i+1);
		 rawTimestamp.push_back(tick);		
		 pair<double,double> coord(valueVec.at(3*i+2),valueVec.at(3*i+3));
		 //cout<<"x "<<coord.first<<" y: "<<coord.second<<endl; 
         raw.push_back(coord);
         pair<int, int> point = grid.real2Grid(coord.first,coord.second);
		 double v;
		 if ((3*i+3+3) < valueVec.size()){
			 double interval = valueVec.at(3*i+4)-valueVec.at(3*i+1);

			 v  = (valueVec.at(3*i+5)-coord.first)*
				  (valueVec.at(3*i+5)-coord.first)+
				  (valueVec.at(3*i+6)-coord.second)*
				  (valueVec.at(3*i+6)-coord.second);
			 v  = sqrt(v)/interval;
		 }

         if (point.first < 0 || point.second < 0 
               || point.first >= dims.first || point.second >= dims.second)
            continue;
         //cout << "POINT: "<<point.first<<" "<<point.second<<endl;

         if (trace.size() > 0) {
            int x1 = trace.back().first;
            int y1 = trace.back().second;
            int x2 = point.first;
            int y2 = point.second;
            int dist1 = (int)fabs(x2 - x1);
            int dist2 = (int)fabs(y2 - y1);
            if (dist1 > 1 || dist2 > 1) {
               //cout << "JOIN: "<<x1<<" "<<y1<<"   "<<x2<<" "<<y2<<endl; 

              
               int dist = max(dist1, dist2);


               //cout << "CHECK: "<<dist<<" "
               //   << (float)(x2-x1)/dist<<"  "<<(float)(y2-y1)/dist<<endl;

               double stepX = (x2-x1) != 0 ? ((float)(x2-x1)/(float)dist) : 0;
               double stepY = (y2-y1) != 0 ? ((float)(y2-y1)/(float)dist) : 0;

               //cout << "STEPS: "<<stepX<<" "<<stepY<<endl;
                
               for (int d=1; d <= dist; d++) {
                  //cout << "PT: "<<floor((float)x1+(float)d*stepX+.5)
                  //   <<"  "<< floor((float)y1+(float)d*stepY+.5)<<endl;
                  point.first = (int)floor((float)x1+(float)d*stepX+.5);
                  point.second = (int)floor((float)y1+(float)d*stepY+.5);
                  //cout << "PT: "<<x1 <<"  "<<y1<<"  "<<point.first
                  //   <<"  "<<point.second<<endl;
                  if (trace.back().first == point.first && 
                        trace.back().second == point.second) continue;
				  // push v
				  velocity.push_back(v);
                  trace.push_back(point);
                  timestamp.push_back(tick);
                  //cout << "added: "<<trace.back().first<<"  "
                  //   <<trace.back().second<<endl;
               }
            }else if (dist1 == 0 && dist2 == 0){ 
               continue;
			}else {
			   // push v
			   velocity.push_back(v);
               trace.push_back(point);
               timestamp.push_back(tick);
            }
         }
         if (trace.size() > 0 && trace.back().first == point.first &&
               trace.back().second == point.second) continue;
         // push v
		 velocity.push_back(v);
		 trace.push_back(point);
         timestamp.push_back(tick);
      }
      // check the length of the trajectory and the velocity sequence
	  if (trace.size()!=velocity.size()){
	     cout<<"traj length not equal to vel length"<<endl;
	  }
      velocities.push_back(velocity);
	  traces.push_back(trace);
      timestamps.push_back(timestamp);
      rawTimestamps.push_back(rawTimestamp);
	  robots.push_back(robot_grid);
	  r_robots.push_back(robot_real);
	  rawTrajs.push_back(raw);
      //cout<<"check    end:"<<trace.back().first<<" "<<trace.back().second<<" rob: "<<robot_grid.first<<" "<<robot_grid.second<<endl;
   }

   cout << "   loaded: "<<traces.size()<<" trajs "<<velocities.size()<<" vel seqs "
	   <<robots.size()<<" robot positions "<<rawTrajs.size()<<" raw trajs "<<
	   r_robots.size()<<" real-value robots pos"<<endl;
}

void Evidence::write(string filename) {
   ofstream file(filename.c_str());

   for (int i=0; i < traces.size(); i++) {
	   file << "**************Print traj "<<i<<endl;
      for (int j=0; j < traces.at(i).size(); j++) {
         file << timestamps.at(i).at(j)<< " "
            << traces.at(i).at(j).first<< " "
            << traces.at(i).at(j).second<<" "
			<< velocities.at(i).at(j)<<endl; 
      } 
   }

   file.close();
}

void Evidence::writeRaw(string filename) {
   ofstream file(filename.c_str());

   for (int i=0; i < rawTrajs.size(); i++) {
	   file << "**************Print traj "<<i<<endl;
      for (int j=0; j < rawTrajs.at(i).size(); j++) {
         file << rawTimestamps.at(i).at(j)<< " "
            << rawTrajs.at(i).at(j).first<< " "
            << rawTrajs.at(i).at(j).second<<endl; 
      } 
   }

   file.close();
}

void Evidence::split(Evidence &set1, Evidence &set2, double perc) {

  int n = traces.size();
   
  vector<int> randOrder = randomPerm(n);

  int i=0;
  int lim=max(0,min(n,(int)floor(perc*n)));

  for (i=0; i < lim; i++) {
    set1.traces.push_back(traces.at(i));
    set1.timestamps.push_back(timestamps.at(i));
	set1.velocities.push_back(velocities.at(i));
	set1.robots.push_back(robots.at(i));
	set1.r_robots.push_back(r_robots.at(i));
	set1.rawTrajs.push_back(rawTrajs.at(i));
//	set1.rawTimestamps.push_back(rawTimestamps.at(i));
  }
  for (;i < n; i++) {
    set2.traces.push_back(traces.at(i));
    set2.timestamps.push_back(timestamps.at(i));
	set2.velocities.push_back(velocities.at(i));
	set2.robots.push_back(robots.at(i));
	set2.r_robots.push_back(r_robots.at(i));
	set2.rawTrajs.push_back(rawTrajs.at(i));
//	set1.rawTimestamps.push_back(rawTimestamps.at(i));
  } 
  set1.factor_ = factor_;
  set2.factor_ = factor_;
}

void Evidence::deterministicsplit(Evidence &set1, Evidence &set2) {
 
  double perc = 0.5; 
  int n = traces.size();
   
  vector<int> randOrder = randomPerm(n);

  int i=0;
  int lim=max(0,min(n,(int)floor(perc*n)));

  for (i=0; i < lim; i++) {
    set1.traces.push_back(traces.at(i));
    set1.timestamps.push_back(timestamps.at(i));
  }
  //single trajectory in testSet
  set2.traces.push_back(traces.at(n-1));
  set2.timestamps.push_back(timestamps.at(n-1));  
  
  set1.factor_ = factor_;
  set2.factor_ = factor_;
}

vector<int> randomPerm(int size) {

   vector<pair<int, int> > randomList;
 
   for (int i=0; i < size; i++)  
      randomList.push_back(pair<int, int>(rand(), i));
  
   sort(randomList.begin(), randomList.end());

   vector<int> res;

   for (int i=0; i < size; i++) {
      res.push_back(randomList.at(i).second); 
   }
 
   return res;
}

void copy3DPrior(Grid &grid,int V,vector<vector<vector<double> > > &prior3D, vector<vector<double> > &prior2D){
   pair<int, int> dims = grid.dims();
  
   prior3D.resize(dims.first,vector<vector<double> > (dims.second, 
				   vector<double> (V,-HUGE_VAL)));
	   
   for (int x = 0;x<dims.first;x++){
	   for(int y= 0;y<dims.second;y++){
		   for (int v=0;v<V;v++)
			 prior3D.at(x).at(y).at(v) = prior2D.at(x).at(y) -log(3);
	   }
   }
}

void generatePrior(Grid &grid, Evidence &evid, vector<vector<double> > &prior,
      vector<double> radiusWeight, int factor) {

   pair<int, int> dims = grid.dims();

   if (radiusWeight.size() < 2) {
      prior.resize(dims.first, vector<double>(dims.second, -HUGE_VAL)); 
      
      int N = evid.size()*2;

      for (int i=0; i < evid.size(); i++) {
         pair<int, int> start = evid.at(i).front();
         pair<int, int> end = evid.at(i).back();


         start.first*=factor;
         start.second*=factor;
         end.first*=factor;
         end.second*=factor;

         int x, y;
         x = min(start.first, dims.first-1);
         y = min(start.second, dims.second-1);
         prior.at(x).at(y) = log(exp(prior.at(x).at(y))*N+1) - log(N);
         if (isnan(prior.at(x).at(y)))
            prior.at(x).at(y) = -log(N);

         x = max(min(end.first, dims.first-1),0);
         y = max(min(end.second, dims.second-1),0);

         prior.at(x).at(y) = log(exp(prior.at(x).at(y))*N+1) - log(N); 
         if (isnan(prior.at(x).at(y)))
            prior.at(x).at(y) = -log(N); 
      } 
   }
   else {
      int R = radiusWeight.size();
      prior.resize(dims.first, vector<double>(dims.second, 
               0.1/(dims.first*dims.second)));

      for (int i=0; i < grid.dims().first; i++) 
         for (int j=0; j < grid.dims().second; j++) 
            if (grid.at(i,j))
               prior.at(i).at(j) = 0;
         
      
      for (int i=0; i < evid.size(); i++) {

         pair<int, int> start = evid.at(i).front();
         pair<int, int> end = evid.at(i).back();

         start.first*=factor;
         start.second*=factor;
         end.first*=factor;
         end.second*=factor;

         int x, y;
         x = min(start.first, dims.first-1);
         y = min(start.second, dims.second-1);
         
         for (int x1=max(x-R+1, 0); x1 <= min(x+R-1, dims.first-1); x1++) {
            int slack = R-abs(x1-x);
            for (int y1=max(y-slack+1, 0); y1 <= min(y+slack-1, dims.second-1); 
                  y1++) {
               int dist = abs(x1-x) + abs(y1-y);
               if (grid.at(x1,y1)) continue;
               prior.at(x1).at(y1) += radiusWeight.at(dist);
            }
         } 
         x = min(end.first, dims.first-1);
         y = min(end.second, dims.second-1);
         
         for (int x1=max(x-R+1, 0); x1 <= min(x+R-1, dims.first-1); x1++) {
            int slack = R-abs(x1-x);
            for (int y1=max(y-slack+1, 0); y1 <= min(y+slack-1, dims.second-1); 
                  y1++) {
               int dist = abs(x1-x) + abs(y1-y);
               if (grid.at(x1,y1)) continue;
               prior.at(x1).at(y1) += radiusWeight.at(dist);
            }
         } 

      } 

      double sum = 0.0;
      for (int x=0; x < dims.first; x++) 
         for (int y=0; y < dims.second; y++) 
            sum += prior.at(x).at(y);
         
      for (int x=0; x < dims.first; x++)
         for (int y=0; y < dims.second; y++) 
            prior.at(x).at(y) = log(prior.at(x).at(y))-log(sum); 
   }
}

void reducePrior(vector<vector<double> > &origPrior, 
      vector<vector<double> > &newPrior, int factor) {

   pair<int, int> dims((int)ceil((float)origPrior.size()/factor),
                  (int)ceil((float)origPrior.at(0).size()/factor));


   newPrior.clear();
   newPrior.resize(dims.first, vector<double>(dims.second, -HUGE_VAL));

   for (int i=0; i < origPrior.size(); i++) {
      for (int j=0; j < origPrior.at(i).size(); j++) {
         newPrior.at(i/factor).at(j/factor) = LogAdd(
               newPrior.at(i/factor).at(j/factor), origPrior.at(i).at(j));
      }
   }
   
}
