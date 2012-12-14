#include "options.h"
#include "visualize.h"
#include "optimization.h"
#include <math.h>


int main(int argc, char **argv) {
   OptionParser opts;

   string mapFile, evidFile;//interactFile,ignoreFile;

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

   cout << "Loading Map File"<<endl;
   BMPFile bmpFile(mapFile); 
   Grid grid(bmpFile, black);
//   cout << "xdim: "<<grid.dims().first<<" yDim: "<<grid.dims().second<<endl;
   cout << "Loading Evidence"<<endl;
   //Evidence trainSet(evidFile, grid, factor);
   /* used when need to train two seperate models
   Evidence evid_int(interactFile, grid, factor);
   Evidence evid_ig(ignoreFile, grid, factor);
   Evidence train_int(grid),test_int(grid),train_ig(grid), test_ig(grid);
   evid_int.split(train_int, test_int, 0.05);
   evid_ig.split(train_ig, test_ig, 0.05);
   */
   Evidence evid(evidFile,grid,factor);
   Evidence trainSet(grid),testSet(grid);
   evid.split(trainSet,testSet,0.05);
   cout<<"Optimize over "<<trainSet.size()<<" examples"<<endl;
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
      ObstacleBlurFeature blurFeat(grid, 5*i);
      features.push_back(blurFeat);
   }

   /*
   cout << "    Robot Feature"<<endl;
   RobotGlobalFeature robglobal(grid,snackbot,factor);
   features.push_back(robglobal);
   //  robot local blurres features
   for (int i=1; i < 5; i++) {
      cout << "  RobotBlur Feature "<<i<<endl;
      RobotLocalBlurFeature robblur(grid,snackbot,5*i,factor);
      features.push_back(robblur);
   }
	
   */
 
   /* 
   cout << "   Creating feature array"<<endl;
   FeatureArray featArray2(features);

   cout << "   Creating lower resolution feature array"<<endl;
   FeatureArray featArray(featArray2, factor);
   */

   cout << " Speed Feature"<<endl;
   vector<double> speedTable(2,0.0);
   speedTable.at(1) = 0.75;
   //speedTable.at(2) = 1.1;
   DisVecSeqFeature speedfeat(speedTable);


   /* Robset training weights: 
	* -3.83 -8.35991 -2.6512 -5.43475 -3.15203 -3.29758
	*  0.596987 0.439284
	* 0.589445 -0.82448
	* Non-robot-ending trainng weights:
	* -4.57257  -6.2 -0.3537 -2.7385 -0.9357 -0.2797
	* -0.495205 -0.2863
	* -1.2225 0.43993
	*/
   vector<double> weights(6+2+2, -0.0);
   weights.at(0) = -25;	
   weights.at(1) = -8.36;
   weights.at(2) = -2.65;
   weights.at(3) = -5.43;
   weights.at(4) = -3.17;
   weights.at(5) = -3.34;
   
   weights.at(6) = 0.5; // robot feature
   weights.at(7) = 0.3; // robot feature
  
   weights.at(8) = -0.29;  // velocity feature
   weights.at(9) = -1.11; // velocity feature

   //weights.push_back(1.5);//the last parameter is for velocity feature
   Parameters params(weights);

   DisSeqOrderInferEngine engine(8,InferenceEngine::GRID8);

   trajOptimizerplus optimizer(grid,trainSet,features,speedfeat,engine);

   optimizer.optimize(params,0.005,1000,1.0,OPT_EXP);

   return 0;

}
