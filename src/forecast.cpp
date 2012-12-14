#include "main.h"
#include "options.h"
#include "evidence.h"
#include "grid.h"
#include "visualize.h"
#include "features.h"
#include "inference.h"
#include "localOptimalIOC.h"
#include <math.h>


double entropy(vector<vector<double> >& P){
    double H = 0.0;
	for (int x =0;x<P.size();x++)
        for (int y=0;y<P.at(0).size();y++)
		   H-=exp(P.at(x).at(y))*P.at(x).at(y)/log(2);
	return H;
}

int main(int argc, char **argv) {
   OptionParser opts;

   string mapFile, evidFile;

   int factor;

   opts.addOption(new StringOption("map", 
            "--map <filename>                 : map file",
            "../input/grid.bmp", mapFile, false));

   opts.addOption(new StringOption("evidence", 
            "--evidence <filename>            : evidence file",
            "", evidFile, true));

   opts.addOption(new IntOption("factor",
            "--factor <int>                   : scaling factor",
            1, factor, true));

   opts.parse(argc,argv);
   JetColorMap jet;
   RGBTRIPLE black = {0,0,0};

   RGBTRIPLE white = {255,255,255};

   RGBTRIPLE red;
   red.R = 255;
   red.G = 0;
   red.B = 0;

   RGBTRIPLE blue;
   blue.R = 0;
   blue.G = 0;
   blue.B = 255;

   RGBTRIPLE green;
   green.R = 0;
   green.G = 255;
   green.B = 0; 

   RGBTRIPLE initialColor;
   initialColor.R = 111; 
   initialColor.G = 49;
   initialColor.B = 152;
//   initialColor.G = 152;
//   initialColor.B = 49;


   RGBTRIPLE currentColor;
   currentColor.R = 181;
   currentColor.G = 165;
   currentColor.B = 213;
//   currentColor.G = 213;
//   currentColor.B = 165;


   RGBTRIPLE magenta;
   magenta.R = 255;
   magenta.G = 0;
   magenta.B = 255;

   RGBTRIPLE cyan;
   cyan.R = 0;
   cyan.G = 255;
   cyan.B = 255;

   RGBTRIPLE yellow;
   yellow.R = 255;
   yellow.G = 255;
   yellow.B = 0;

   BMPFile bmpFile(mapFile);

   Grid grid(bmpFile, black);

   
   Evidence testSet(evidFile, grid, factor);
 /* 
   if (1) { 
	   evid.split(trainSet, testSet, 0.8);
   }else{
	   evid.deterministicsplit(trainSet, testSet);
   }*/

#if 0 
   cout << "Creating Markov Model"<<endl;
   MarkovModel markmodel(grid, trainSet);

   double totalObj = 0.0;

   for (int i=0; i < testSet.size(); i++) {
      vector<pair<int, int> > path = testSet.at(i);
      cout << "Calling eval"<<endl;
      double obj = markmodel.eval(path);
      cout << "OBJ: "<<i<<" "<<obj<<endl;
	
      totalObj += obj;
   }

   cout << "TOTAL OBJ: "<<totalObj<<endl;

   cout << "AVERAGE OBJ: "<<totalObj/testSet.size()<<endl;
   return 0;
#endif
   vector<PosFeature> features;

   cout << "Constant Feature"<<endl;

   ConstantFeature constFeat(grid);
   features.push_back(constFeat);

   cout << "Obstacle Feature"<<endl;

   ObstacleFeature obsFeat(grid);
   features.push_back(obsFeat);

   for (int i=1; i < 5; i++) {
      cout << "Blur Feature "<<i<<endl;
      ObstacleBlurFeature blurFeat(grid, 5*i);
      features.push_back(blurFeat);
   }

   cout << "Creating feature array"<<endl;
   FeatureArray featArray2(features);

   cout << "Creating lower resolution feature array"<<endl;
   FeatureArray featArray(featArray2, factor);

   pair<int, int> dims = grid.dims();
   pair<int, int> lowDims((int)ceil((float)dims.first/factor),
         (int)ceil((float)dims.second/factor));

   vector<double> weights(features.size(), -0.0);
   weights.at(1) = -6.2;
   //for (int i=2; i < weights.size(); i++)
   //   weights.at(i) = -1.0;
   weights.at(0) = -2.23;//-2.23
   weights.at(2) = -0.35;
   weights.at(3) = -2.73;
   weights.at(4) = -0.92;
   weights.at(5) = -0.26;
   Parameters params(weights);

   OrderedWaveInferenceEngine engine(InferenceEngine::GRID8);

   vector<vector<double> > prior(dims.first,vector<double> (dims.second,0.0));
/*
   double divide = 1.0;
   vector<double> radiusWeight;
   for (int i=0; i < 20; i++) {
      radiusWeight.push_back(1.0/divide);
      divide*=2;
   }
   generatePrior(grid, trainSet, priorOrig, radiusWeight, factor);
 
   reducePrior(priorOrig, prior, factor);
*/

   vector<vector<vector<double> > > partition, backpartition;

   int time0 = time(0);

   BMPFile gridView(dims.first, dims.second);



   RewardMap rewards(featArray, params); 

   vector<double> sums(params.size(),0.00001);
      
   vector<vector<double> > occupancy;

   Predictor predictor(grid, rewards, engine); 
   
   predictor.setPrior(prior);


   cout << testSet.size() <<" Examples"<<endl;

   for (int i=0; i < testSet.size(); i++) {

      int index = 0;


      vector<pair<int, int> > traj = testSet.at(i);
      vector<double> times = testSet.getTimes(i); 
      pair<int, int> initial = traj.front();
	  pair<int,int> & botinGrid = testSet.at_bot(i); 
	  pair<double,double>& botinPoint = testSet.at_rbot(i);
	  pair<double,double>& end = testSet.at_raw(i).back();

      predictor.setStart(initial); 

      double thresh = -20.0;
	  double startTime = times.front();

      char buf[1024];
      sprintf(buf, "../output/pppredict%03d.dat", i);
      ofstream file(buf);

      for (double tick = startTime; index < traj.size(); tick+=0.4) {

         for ( ; index < traj.size() && times.at(index) < tick; index++); 

         if (index == traj.size() ) break;
 
         cout << "Evidence: "<<i<<"  timestep: "<<tick
            <<"   index: "<<index<<endl;
         predictor.predict(traj.at(index), occupancy);

         cout << "SIZE: "<<prior.size()<<endl;
		 vector<vector<double> >  pos 
            = predictor.getPosterior();

         gridView.addBelief(pos, -30.0, 0.0,jet);

         grid.addObstacles(gridView, black);
         gridView.addLabel(botinGrid,green);
         vector<pair<int, int> > subTraj;

         subTraj.insert(subTraj.end(), traj.begin(), traj.begin()+index);

         gridView.addVector(subTraj, red, factor);

         sprintf(buf, "../compare/pp%03d-%03f.bmp", i, tick-startTime); 
         gridView.write(buf);
		 //pair<double,double> values = predictor.check(traj.back());
		 double cost = 0.0;
		 for(int itr = 0;itr<index;itr++)
		   cost +=rewards.at(traj[itr].first,traj[itr].second);

		 cout<<i<<" Normalizer: "<<predictor.getNormalizer(traj.back())<<
			 " path cost: "<<cost<<" Probability:  "<<cost+predictor.getNormalizer(traj.back())<<endl;

         vector<vector<vector<double> > > timeOcc 
            = predictor.getTimeOccupancy();

		 vector<vector<double > > posterior  = predictor.getPosterior();
		 double maxV = -HUGE_VAL;
		 pair<int,int> predestGrid;
		 pair<double,double> predestPoint;

         for (int ii=0; ii< dims.first; ii++) { 
            for (int jj=0; jj < dims.second; jj++) {
			   if(posterior[ii][jj]>maxV){
				   predestGrid.first = ii;
				   predestGrid.second = jj;
			   }
               maxV  = max(maxV, posterior.at(ii).at(jj));
            }
         }
		 predestPoint = grid.grid2Real(predestGrid.first,predestGrid.second);
		 double dist = sqrt((end.first-predestPoint.first)*(end.first-predestPoint.first)
			 +(end.second-predestPoint.second)*(end.second-predestPoint.second));

		 double logloss = entropy(posterior);

		 cout<<"final belief: "<<posterior.at(traj.back().first).at(traj.back().second)
			 <<" max: "<<maxV
			 <<" logloss: "<<logloss<<endl; 
		 cout<<botinGrid.first<<" "<<botinGrid.second
			 <<" "<<predestGrid.first<<" "<<predestGrid.second<<endl;
		 file<<tick-startTime
			 <<" "<<logloss
			 <<" "<<posterior.at(botinGrid.first).at(botinGrid.second)
			 <<" "<<posterior.at(traj.back().first).at(traj.back().second)
			 <<" "<<maxV<<" "<<dist<<endl;

      } 
      file.close();
   }

}


