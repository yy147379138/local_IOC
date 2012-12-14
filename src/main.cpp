#include "main.h"
#include "options.h"
#include "grid.h"
#include "evidence.h"
#include "visualize.h"
#include "features.h"
#include "inference.h"

int main(int argc, char **argv) {
   OptionParser opts;

   string mapFile, evidFile;

   opts.addOption(new StringOption("map", 
            "--map <filename>                 : map file",
            "../input/grid.bmp", mapFile, false));

   opts.addOption(new StringOption("evidence", 
            "--evidence <filename>            : evidence file",
            "", evidFile, true));


   BMPFile bmpFile("problem01.bmp");

   opts.parse(argc,argv);

   JetColorMap jet;

#if 0 
   Grid grid(mapFile);
#else
   Grid grid(bmpFile, black);

   vector<pair<int, int> > initialCells = bmpFile.find(initialColor);
   vector<pair<int, int> > currentCells = bmpFile.find(currentColor);

   if (initialCells.size() != 1 && currentCells.size() != 1) {
      cout << "Initial: "<<initialCells.size()<<endl;
      cout << "Current: "<<currentCells.size()<<endl;
      cout << "Incorrect number of initial and current cells "
         "in the maps"<<endl;
      exit(0); 
   }

   pair<int, int> initial = initialCells.at(0);
   pair<int, int> current = currentCells.at(0);
#endif 

   Evidence evid(evidFile, grid);


   cout << "EVIDENCE SIZE: "<<evid.size()<<endl;

   vector<PosFeature> features;

   cout << "Constant Feature"<<endl;

   ConstantFeature constFeat(grid);
   features.push_back(constFeat);

   cout << "Obstacle Feature"<<endl;

   ObstacleFeature obsFeat(grid);
   features.push_back(obsFeat);

   for (int i=1; i < 5; i++) {
      cout << "Blur Feature "<<5*i<<endl;
      ObstacleBlurFeature blurFeat(grid, 5*i);
      features.push_back(blurFeat);
   }
#if 0
   for (int i=2; i < 25; i++) {
      cout << "Max Feature "<<i<<endl;
      ObstacleMaxFeature maxFeat(grid, i);
//      features.push_back(maxFeat);
   }
#endif

   cout << "Creating feature array"<<endl;
   FeatureArray featArray(features);

   cout << "Creating Prior"<<endl;
   vector<vector<double> > prior;
   vector<double> radiusWeight;

   double divide = 1.0;
   for (int i=0; i < 20; i++) {
      radiusWeight.push_back(1.0/divide);
      divide*=2.0;
   }



   generatePrior(grid, evid, prior, radiusWeight);


   

   pair<int, int> dims = grid.dims();

   BMPFile gridView(dims.first,dims.second);

   grid.addObstacles(gridView, black);

   gridView.write("map.bmp"); 

   for (int i=0; i< evid.size(); i++) {
      vector<pair<int, int> > evidVec = evid.at(i);
      gridView.addVector(evidVec, blue);
   }

   gridView.write("evidence.bmp");

   gridView.addBelief(prior, -25.0, -2.0, jet, false);

   grid.addObstacles(gridView, black);

   gridView.write("prior.bmp");


   for (int k=0; k < featArray.size(); k++) {
      vector<vector<double> > vals(dims.first, 
            vector<double>(dims.second, 0.0));
      for (int i=0; i < dims.first; i++) {
         for (int j=0; j < dims.second; j++) {
            vals.at(i).at(j) = featArray.at(i,j).at(k); 
         }
      }
      gridView.addBelief(vals, 0.0, 1.0, jet, false);
      //grid.addObstacles(gridView, black);
      char buf[1024];
      sprintf(buf, "features%02d.bmp", k); 
      gridView.write(buf); 
   }


   cout << "INFERENCE"<<endl;

   vector<double> weights(features.size(), -0.0);

#if 0
   weights.at(1) = -1000;
   for (int i=2; i < 15; i++)
      weights.at(i) = -2;
   weights.at(0) = -6;
#endif
   weights.at(1) = -106.2;
   //for (int i=2; i < weights.size(); i++)
   //   weights.at(i) = -1.0;
   weights.at(0) = -2.23;
   weights.at(2) = -0.35;
   weights.at(3) = -2.73;
   weights.at(4) = -0.92;
   weights.at(5) = -0.26;

   Parameters params(weights);

   OrderedWaveInferenceEngine engine(grid, featArray, InferenceEngine::GRID8);

   vector<vector<vector<double> > > partition, backpartition;

   int time0 = time(0);
   
   RewardMap rewards(featArray, params);

   vector<vector<double> > costVals(dims.first, 
         vector<double>(dims.second, 0.0));

   double maxV = -HUGE_VAL;
   double minV = HUGE_VAL;
   for (int i=0; i < dims.first; i++) { 
      for (int j=0; j < dims.second; j++) {
         costVals.at(i).at(j) = -rewards.at(i,j); 
         maxV = max(maxV, -rewards.at(i,j));
         minV = min(minV, -rewards.at(i,j));
      }
   }

   cout << "MAXV: "<<maxV<<"  MINV: "<<minV<<endl;
   gridView.addBelief(costVals, minV, maxV-2.0, jet, false);
   gridView.write("costmap.bmp");


   vector<vector<double> > costs;
   vector<pair<int, int> > order;

   cout << "DIJKSTRAS"<<endl;
   double maximum;

   rewards.dijkstras(initial, costs, order, maximum);

   BMPFile optCost(dims.first, dims.second);

   optCost.addBelief(costs, 0.00, maximum, white, red, false);

   optCost.write("optcost.bmp");

   vector<pair<int, int> > traj = evid.at(28);

   Predictor predictor(grid, rewards, engine);

   for (int i=0; i < traj.size(); i++) { 


      cout << "SETTING START"<<endl;

      predictor.setStart(traj.at(0));

      cout << "SETTING PRIOR"<<endl;

      predictor.setPrior(prior);

      cout << "PREDICTING"<<endl;

      vector<vector<double> > occupancy;

      int tick = time(0);

      int N = 1;

      for (int t=0; t < N; t++)
         predictor.predict(traj.at(i), occupancy);

      cout << "  completed in: "<< (float)(time(0) - tick)/N << " seconds"<<endl;


      BMPFile gridView2(dims.first,dims.second);


      gridView2.addBelief(occupancy, -30.0, 5.0, jet);

      grid.addObstacles(gridView2, black);

      vector<pair<int, int> > subtraj(traj.begin(), traj.begin()+i+1);
      gridView2.addVector(subtraj, white, 1, 2);
      char buf[1024];
      sprintf(buf, "occup%03d.bmp", i);
      gridView2.write(buf);
   }

   vector<vector<vector<double> > > & timeOcc = predictor.getTimeOccupancy();

   for (int t=0; t < timeOcc.size(); t++) {
      //gridView.addBelief(timeOcc.at(t), -20.0, 0.0, white, red);

      gridView.addBelief(timeOcc.at(t), -20.0, 0.0, jet);
      grid.addObstacles(gridView, black);

      char buf[1024];
      sprintf(buf, "occ%d.bmp", t);
      gridView.write(buf);
   }

#if 1 
   vector<vector<double> > posterior = predictor.getPosterior();
   gridView.addBelief(posterior, -20.0, 0.0, white, red);
   //grid.addObstacles(gridView, black);
   gridView.write("destposterior.bmp");
#else
   double scale = .1;

   engine.forward(pair<int, int>(150,110), 100, rewards, partition);

   vector<vector<double> > posterior = partition.at(20);
  
   engine.flattenPartitions(partition, posterior);
#endif 

/*    for (int i=100;i<200;i++) {
      cout << "VAL CHECK: "<<posterior.at(250).at(i)<<"  "
         <<posterior.at(250).at(i+1)-posterior.at(250).at(i)<<endl;
      
   } */


/*  
   for (int i=0; i < 490; i++) {
      for (int j=0; j < 321; j++) {
         int value = grid.at(i,j);
         int pixel=min(255,max(0, 
                  (int)floor(10.0*posterior.at(i).at(j)*scale)+255)); 

         //if (pixel == 255) continue;

         //cout << "PIXEL: "<<pixel<< "   "<<posterior.at(i).at(j)<<endl;

         black.R = 255;
         black.G = 255-pixel;
         black.B = 255-pixel; 

         gridView.setPixel(i,j,black);
         //if (value == 1) gridView.setPixel(i,j,black);
         //if (value == 5) gridView.setPixel(i,j,red);
      }
   }

   gridView.setPixel(initial.first,initial.second,initialColor);
   gridView.setPixel(current.first,current.second,currentColor);

   for (int i=0; i < 490; i++) {
      for (int j=0; j < 321; j++) {
         int value = grid.at(i,j);
       
         int pixel = (int)floor(255-
               255*(min(1.0*features.at(1).at(i,j),1.0)/1.0));

         if (pixel == 255) continue;

         //cout << "PIXEL: "<<pixel<< "   "<<prior.at(i).at(j)<<endl;

         black.R = pixel;
         black.G = pixel;
         black.B = pixel; 

         gridView.setPixel(i,j,black);
         //if (value == 1) gridView.setPixel(i,j,black);
         //if (value == 5) gridView.setPixel(i,j,red);
      }
   }


   gridView.write("destposterior.bmp"); 
*/
   exit(0);

   //for (int i=0; i < N; i++) {
      pair<int, int> pos(100,100);
      engine.forward(pos, 5, rewards, partition);
      //engine.backward(pos, 5, rewards, backpartition);
      for (int x=0; x < partition.at(0).size(); x++) {
         for (int y=0; y < partition.at(0).at(x).size(); y++) {
            cout << x << "  "<<y<<"  "<<partition.at(0).at(x).at(y)<< "  "
               <<  backpartition.at(3).at(x).at(y)<<endl;
         }
      }
   //}
  // cout << "Run time: "<<(float)(time(0)-time0)/N<<endl;
}

#if 0
void convert(string str, string delim, vector<int> &results) {
   int next;
   char buf[20];
   while ( (next=str.find_first_of(delim)) != str.npos) {
      if (next > 0) 
         results.push_back(atoi(str.substr(0,next).c_str()));
         
      str = str.substr(next+1); 
   }
} 

void convert(string str, string delim, vector<double> &results) {
   int next;
   char buf[20];
   while ( (next=str.find_first_of(delim)) != str.npos) {
      if (next > 0) 
         results.push_back(atof(str.substr(0,next).c_str()));
         
      str = str.substr(next+1); 
   }
} 
#endif

