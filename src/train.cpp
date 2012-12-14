#include "main.h"
#include "options.h"
#include "grid.h"
#include "evidence.h"
#include "visualize.h"
#include "features.h"
#include "inference.h"
#include "optimization.h"
#include <math.h>


int main(int argc, char **argv) {
   OptionParser opts;

   string mapFile, evidFile;

   int factor = 1, itrs, write;
   double acc, rate;

   opts.addOption(new StringOption("map", 
            "--map <filename>                 : map file",
            "../input/grid.bmp", mapFile, false));

   opts.addOption(new StringOption("evidence", 
            "--evidence <filename>            : evidence file",
            "", evidFile, true));

   opts.addOption(new DoubleOption("rate", 
            "--rate <double>                   : learning rate",
            0.0001,rate, true));

   opts.addOption(new IntOption("itrs", 
            "--itrs <int>                   : iteration times",
            100,itrs, true));

   opts.addOption(new DoubleOption("acc", 
            "--acc <double>                   : accuracy",
            0.0001,acc, true));

   opts.addOption(new IntOption("w", 
            "--w <int>                   : write signal",
            0, write, true));



   opts.parse(argc,argv);

   cout << "Loading Map File"<<endl;
   BMPFile bmpFile(mapFile); 
   Grid grid(bmpFile, black);

   cout << "Loading Evidence"<<endl;
   Evidence evid(evidFile, grid, factor);

   Evidence trainSet(grid), testSet(grid);

   //evid.split(testSet, trainSet, 0.5);

   evid.split(trainSet, testSet, 1.0);


#if 0 
   for (int i=0; i < evid.size(); i++) {
      cout << "Evid "<<i<<endl;
      vector<pair<int, int> > traj = evid.at(i);
      vector<double> timestamps = evid.getTimes(i);

      cout << timestamps.size()<<"  "<<traj.size()<<endl;

      for (int j=0; j < traj.size(); j++) {
         cout << timestamps.at(j)<<"  "<<traj.at(j).first
            << "  "<<traj.at(j).second<<endl;
      } 
   }
#endif
//   testSet.write("testTraj.data");

   cout << "Generating Feature Set"<<endl;

   vector<PosFeature> features;

   cout << "   Constant Feature"<<endl;

   ConstantFeature constFeat(grid);
   features.push_back(constFeat);

   cout << "   Obstacle Feature"<<endl;

   ObstacleFeature obsFeat(grid);
   features.push_back(obsFeat);

   for (int i=1; i < 5; i++) {
      cout << "   Blur Feature "<<i<<endl;
      ObstacleBlurFeature blurFeat(grid, 8*i);
      features.push_back(blurFeat);
   }
#if 0
   for (int i=2; i < 25; i++) {
      cout << "Max Feature "<<i<<endl;
      ObstacleMaxFeature maxFeat(grid, i);
//      features.push_back(maxFeat);
   }
#endif

   cout << "   Creating feature array"<<endl;
   FeatureArray featArray2(features);

   cout << "   Creating lower resolution feature array"<<endl;
   FeatureArray featArray(featArray2, factor);


   pair<int, int> dims = grid.dims();


   vector<double> weights(features.size(), -0.0);
   //for (int i=2; i < weights.size(); i++)
   //   weights.at(i) = -1.0;
   weights.at(0) = -8.46;
   weights.at(1) = -6.18;
   weights.at(2) = -1.34;
   weights.at(3) = -2.69;
   weights.at(4) = -1.90;
   weights.at(5) = -1.25;
   Parameters params(weights);

   OrderedWaveInferenceEngine engine(InferenceEngine::GRID8);


   cout<<"Optimize over "<<trainSet.size()<<" examples"<<endl;
   trainSet.write("./traj.txt");
   trajectoryOptimizer optimizer(grid, trainSet, features, engine);

   optimizer.optimize(params,rate,itrs,acc,OPT_EXP);
	
   if(write){
		ofstream outfile("../params/ppp.dat");
		for(int i=0;i<params.size();i++)
		  outfile<<params.at(i)<<" ";
		outfile<<endl;
		outfile.close();
	}

   return 0;
}
