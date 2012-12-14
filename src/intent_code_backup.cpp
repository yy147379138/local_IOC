#include "main.h"
#include "options.h"
#include "grid.h"
#include "evidence.h"
#include "visualize.h"
#include "features.h"
#include "inference.h"
#include <math.h>
#include <queue>

#define NUMROBFEAT 0
#define VEL_DIM 2
#define NUMPOSFEAT 6

class Planner{
	public:
		Planner(Grid& _grid,RewardMap& _rewards):grid(_grid),
		    rewards(_rewards){
			
			
			}
		void simplePlan(vector<pair<int,int> >& pool,
					pair<int,int>& currentRob,int radius){
			pair<int,int> dims = grid.dims();
			int x = currentRob.first;
			int y = currentRob.second;
			int half = floor(radius/2);
			const int length = 8;
			int dists[length][2] = {
				{min(x+radius,dims.first-1),0},
				{max(x-radius,0),0},
				{0,min(y+radius,dims.second-1)},
				{0,max(y-radius,0)},
				{min(x+half,dims.first-1),min(y+half,dims.second-1)},
				{min(x+half,dims.first-1),max(y-half,0)},
				{max(x-half,0),min(y+half,dims.second-1)},
				{max(x-half,0),max(y-half,0)}};

			for(int itr=0;itr<length;itr++){
				if (grid.at(dists[itr][0],dists[itr][1]))
				  continue;
				pool.push_back(pair<int,int> (dists[itr][0],dists[itr][1]));

			}
		}

		void sociablePlan(vector<pair<int,int> >& pool,
               pair<int,int>& currentRob,int low, int high){
			cout<<"Generate plans"<<endl;
			vector<vector<double> > costs;
			vector<pair<int,int> > order;
			double maximum;
			rewards.dijkstras(currentRob,costs,order,maximum);
			pool.insert(pool.end(), order.begin()+low, order.begin()+high);
			//writePlans(pool);

		}

		double computeKLDivergence(vector<vector<double> >& novel,
			vector<vector<double> >& base){
		    double H = 0.0;
			for (int x =0;x<base.size();x++)
			    for (int y=0;y<base.at(0).size();y++)
				   H+=exp(novel.at(x).at(y))*(novel.at(x).at(y)-base.at(x).at(y));
			return H;
		}
		double computeEntropy(vector<vector<double> >& P){
		    double H = 0.0;
			for (int x =0;x<P.size();x++)
			    for (int y=0;y<P.at(0).size();y++)
				   H-=exp(P.at(x).at(y))*P.at(x).at(y);
			return H;
		}

		double computeEntropy(vector<vector<double> >& P, double& sum){
		    double H = 0.0;
			for (int x =0;x<P.size();x++)
			    for (int y=0;y<P.at(0).size();y++)
				   H-=exp(P.at(x).at(y)-sum)*(P.at(x).at(y)-sum);
			return H;
		}

		void planSimplePath(vector<pair<int,int> > & path, 
					pair<int,int> dest){
			int x1 = path.back().first;
			int y1 = path.back().second;
			int x2 = dest.first;
			int y2 = dest.second;
			int dist_x = x2-x1;
			int dist_y = y2-y1;
			pair<int,int> move(x1,y1);

			while(dist_x!=0||dist_y!=0){
				move.first+=signum(dist_x);
				move.second+=signum(dist_y);
				path.push_back(move);
				dist_x = x2-move.first;
				dist_y = y2-move.second;
			}
		}
	private:
		void writePlans(vector<pair<int,int> >& pool){
			 pair<int,int> dims = grid.dims();
			 BMPFile gridView(dims.first, dims.second);
			 char buf[512];
		      
             grid.addObstacles(gridView, black);
             gridView.addVector(pool, blue, 1);
             sprintf(buf, "../compare/plan.bmp"); 
             gridView.write(buf);
		}

		inline int signum(int x){
			  return (x>0)? 1:((x<0)? -1:0);
		}

		Grid& grid;
		RewardMap& rewards;
		priority_queue<pair<double,pair<int,int> >, 
			  vector<pair<double,pair<int,int> > >, 
			  greater<pair<double,pair<int,int> > > > PQ;
	


};

class IntentRecognizer{
	public:
		IntentRecognizer(Grid& _grid,Parameters& _p_Params,Parameters& 
			_r_PosParams,Parameters& _r_SeqParams,Parameters&
			_nr_PosParams,Parameters& _nr_SeqParams,DisVecSeqFeature&
			_seqFeat,OrderedWaveInferenceEngine engine_p,
            DisSeqOrderInferEngine engine_r, DisSeqOrderInferEngine
			engine_nr):
			grid(_grid),p_Params(_p_Params),r_PosParams(_r_PosParams),
		    r_SeqParams(_r_SeqParams),nr_PosParams(_nr_PosParams),
		    nr_SeqParams(_nr_SeqParams),seqFeat(_seqFeat),p_engine(engine_p),
			r_engine(engine_r),nr_engine(engine_nr){
				
              pair<int, int> dims = grid.dims();
              dimensions.push_back(dims.first);
              dimensions.push_back(dims.second);
              dimensions.push_back(VEL_DIM);

              cout << "Constant Feature"<<endl;
              ConstantFeature constFeat(grid);
              posFeatures.push_back(constFeat);
              cout << "Obstacle Feature"<<endl;
              ObstacleFeature obsFeat(grid);
              posFeatures.push_back(obsFeat);
			  for (int i=1; i < 5; i++) {
                 cout << "Blur Feature "<<i<<endl;
                 ObstacleBlurFeature blurFeat(grid, 5*i);
                 posFeatures.push_back(blurFeat);
              }

		}
		void forecast(vector<pair<int,int> >& traj, vector<double>& vels,
			 vector<double>& times, pair<int,int>& rob,int evid_i,
			 double interval){
			 cout<<"ROB: "<<rob.first<<" "<<rob.second<<endl; 
	         cout << "   Creating feature array 1"<<endl;
             FeatureArray p_featArray(posFeatures);

			 //  robot local blurres features
			  for (int r=1; r <= NUMROBFEAT; r++) {
		         cout << "Adding  Robot Feature "<<r<<endl;
		         RobotLocalBlurFeature robblurFeat(grid,rob,10*r);
   	             //	RobotGlobalFeature robFeat(grid,bot);
		         posFeatures.push_back(robblurFeat);
	           }
	
	          cout << "   Creating feature array 2"<<endl;
              FeatureArray intent_featArray(posFeatures);
	  
	          for (int rr=1;rr<= NUMROBFEAT;rr++)
		         posFeatures.pop_back();
	          
			  cout<<"Generating reward"<<endl;
              RewardMap p_rewards(p_featArray, p_Params);
              RewardMap r_rewards(intent_featArray,seqFeat,r_PosParams,r_SeqParams);
              cout<<"  Declare predictors "<<endl;
              //rewards towards other locations
              RewardMap nr_rewards(intent_featArray,seqFeat,nr_PosParams,nr_SeqParams); 
      
              cout<<"  Declare predictors "<<endl;
              Predictor p_predict(grid, p_rewards, p_engine); 
              DisSeqPredictor r_predict(grid, r_rewards, r_engine); 
              DisSeqPredictor nr_predict(grid, nr_rewards, nr_engine);
			  
			  double prior_weight = -log(dimensions[0]*dimensions[1]);
              vector<vector<double> > prior2D(dimensions[0],vector<double> 
						  (dimensions[1],prior_weight));//prior = 1/|G| number of goals
              cout<<" Set Priors"<<endl;
              p_predict.setPrior(prior2D);


              State initState(traj.front(), seqFeat.getFeat(vels.front()));
              State rState(rob,0);

	          cout<<" Set start "<<endl;
	          p_predict.setStart(traj.front());
	          r_predict.setStart(initState);
	          nr_predict.setStart(initState);
	  
              int index = 0;
        	  char buf[512];
	          sprintf(buf, "../output/beliefs%03d.dat", evid_i);
              ofstream outfile(buf);
	          for (double tick = 0.0; index < traj.size(); tick+=interval) {

					for ( ; index < traj.size() && times.at(index) < tick;
					       index++); 
			             if (index == traj.size() ) break;
 
	                cout << "Evidence: "<<evid_i<<"   timestep: "<<tick<<"  index: "<<index<<endl;
         
                    vector<double>  pathcost (3,0.0);
                    for (int j=0; j <=index; j++){
			             pathcost.at(0) = pathcost.at(0) +
				          p_rewards.at(traj.at(j).first, traj.at(j).second);
			             pathcost.at(1) = pathcost.at(1)+
				          r_rewards.at(traj.at(j).first, traj.at(j).second,
					         seqFeat.getFeat(vels.at(j)));
			             pathcost.at(2) = pathcost.at(2) +
				          nr_rewards.at(traj.at(j).first, traj.at(j).second,
					         seqFeat.getFeat(vels.at(j)));
     	             } 
                     cout<<"path cost "<<pathcost[0]<<" "<<pathcost[1]<<" "<<pathcost[2]<<endl;

		             cout<<" Prediction "<<endl; 
                     State currentState(traj.at(index),seqFeat.getFeat(vels.at(index)));

					 vector<vector<double> > occupancy2D; 
					 p_predict.predict(traj.at(index), occupancy2D);
                     r_predict.forwardInference(currentState);
                     nr_predict.forwardInference(currentState);

                     posterior2D = p_predict.getPosterior();
		             r_Distribute = r_predict.getObdistribute();
		             nr_Distribute = nr_predict.getObdistribute();
					 
					 genIntentPosterior(rob,pathcost);

					 cout<<"Compute beliefs"<<endl;
					 vector<double> beliefs(3,0.0);
					 beliefs.at(0) = pathcost[0]+p_predict.getNormalizer(rState.pos);
					 beliefs.at(1) = pathcost[1]+r_predict.getNormalizer(rState);
					 beliefs.at(2) = pathcost[2]+nr_predict.getNormalizer(rState);
					 
					 writeOutput(traj, index,beliefs, tick,rob,evid_i,outfile);

			  }

			  outfile.close();
		}
  	    
		void active_forecast(vector<pair<int,int> >& traj, vector<double>& vels,
			 vector<double>& times, pair<int,int>& rob,int evid_i,
			 double interval){
			 cout<<"ROB: "<<rob.first<<" "<<rob.second<<endl; 
             FeatureArray p_featArray(posFeatures);
	          
			  cout<<"Generating reward"<<endl;
              RewardMap p_rewards(p_featArray, p_Params);
              RewardMap r_rewards(p_featArray,seqFeat,r_PosParams,r_SeqParams);
              RewardMap nr_rewards(p_featArray,seqFeat,nr_PosParams,nr_SeqParams); 
              cout<<"  Declare predictors "<<endl;
      
              cout<<"  Declare Planner "<<endl;
			  Planner planner(grid,p_rewards);

              cout<<"  Declare predictors "<<endl;
              //Predictor p_predict(grid, p_rewards, p_engine); 
              DisSeqPredictor r_predict(grid, r_rewards, r_engine); 
              DisSeqPredictor nr_predict(grid, nr_rewards, nr_engine);
			  /*
              vector<vector<double> > prior2D(dimensions[0],vector<double> 
						  (dimensions[1],0.0));
              cout<<" Set Priors"<<endl;
              p_predict.setPrior(prior2D);
              */

              State initState(traj.front(), seqFeat.getFeat(vels.front()));
              State rState(rob,0);

	          cout<<" Set start "<<endl;
	          //p_predict.setStart(traj.front());
	          r_predict.setStart(initState);
	          nr_predict.setStart(initState);
	  
              int index = 0;
        	  char buf[512];
	          sprintf(buf, "../output/activeProbs%03d.dat", evid_i);
              ofstream outfile(buf);
			  
			  vector<pair<int,int> > robPath;
              robPath.push_back(rob);
	          for (double tick = 0.0; index < traj.size(); tick+=interval) {

					for ( ; index < traj.size() && times.at(index) < tick;
					       index++); 
			             if (index == traj.size() ) break;
 
	                cout << "Evidence: "<<evid_i<<"   timestep: "<<tick<<"  index: "<<index<<endl;
         
                    vector<double>  pathcost (3,0.0);
                    for (int j=0; j <=index; j++){
			             pathcost.at(0) = pathcost.at(0) +
				          p_rewards.at(traj.at(j).first, traj.at(j).second);
			             pathcost.at(1) = pathcost.at(1)+
				          r_rewards.at(traj.at(j).first, traj.at(j).second,
					         seqFeat.getFeat(vels.at(j)));
			             pathcost.at(2) = pathcost.at(2) +
				          nr_rewards.at(traj.at(j).first, traj.at(j).second,
					         seqFeat.getFeat(vels.at(j)));
     	             } 
                     cout<<"path cost "<<pathcost[0]<<" "<<pathcost[1]<<" "<<pathcost[2]<<endl;

		             cout<<" Prediction "<<endl; 
                     State currentState(traj.at(index),seqFeat.getFeat(vels.at(index)));

					 //vector<vector<double> > occupancy2D; 
					 //p_predict.predict(traj.at(index), occupancy2D);
                     r_predict.forwardInference(currentState);
                     nr_predict.forwardInference(currentState);

                     //posterior2D = p_predict.getPosterior();
		             r_Distribute = r_predict.getObdistribute();
		             nr_Distribute = nr_predict.getObdistribute();
					 
					 double posteriorSUM = pre_GenPosterior(pathcost);

					 vector<pair<int,int> > nextPosPool;
					 planner.sociablePlan(nextPosPool,rob,100,150);
					 pair<int,double> minimum(0,HUGE_VAL);
					 double current;
					 for (int n=0;n<nextPosPool.size();n++){
						 assign_Posterior(nextPosPool[n],pathcost[1],posteriorSUM);
						 current = planner.computeEntropy(flatposterior3D,posteriorSUM);
						 if (current<minimum.second){
							 minimum.first = n;
							 minimum.second = current;
						 }
						 dismiss_Posterior(nextPosPool[n],pathcost[2],posteriorSUM);
					 }
					 
					 pair<int,int> nextPos(0,0);
					 //genIntentPosterior(rob,pathcost);
					 assign_Posterior(rob,pathcost[1],posteriorSUM);
					 current = planner.computeEntropy(flatposterior3D,posteriorSUM);
					 if (minimum.second<current){
						 cout<<"Minimum entropy: "<<minimum.second<<endl;
						 nextPos.first = nextPosPool[minimum.first].first;
						 nextPos.second = nextPosPool[minimum.first].second;
					 }
					
					 vector<pair<int,int> > subTraj;
                     subTraj.insert(subTraj.end(), traj.begin(), traj.begin()+index);
      
					 writeOutput(subTraj,robPath,tick,rob,nextPos,posteriorSUM,evid_i,outfile);
					 if (nextPos.first!=0&&nextPos.second!=0){
						 rob.first = nextPos.first;
						 rob.second = nextPos.second;
					     planner.planSimplePath(robPath,nextPos);
					 }
					   
			  }

			  outfile.close();

		}

	private:
		double genIntentPosterior(pair<int,int>& robot, vector<double>& cost){
			
			cout<<"Generate Posterior"<<endl;
			double sum = -HUGE_VAL;
			double prior_weight = -log(dimensions[0]*dimensions[1]*dimensions[2]);
			actionDistribute.clear();
			actionDistribute.resize(dimensions[0], vector<vector<double> > (
					dimensions[1],vector<double>(dimensions[2],0.0)));

		    for(int x=0;x<dimensions[0];x++){
			   for(int y=0;y<dimensions[1];y++){
				 for(int v=0;v<dimensions[2];v++){
					 if (x == robot.first && y==robot.second){
						 actionDistribute.at(x).at(y).at(v) = cost[1]+
                           r_Distribute.at(x).at(y).at(v) + prior_weight; 
					 }else{
						 actionDistribute.at(x).at(y).at(v) = cost[2]+
                           nr_Distribute.at(x).at(y).at(v) + prior_weight;
					 } 
					 sum = LogAdd(sum,actionDistribute.at(x).at(y).at(v));
				 }
			   }
		    }
		 
            //cout<<"Observation SUM: "<<sum<<endl;
		    //cout<<" 3D posterior"<<endl;
            flatposterior3D.clear();
		    flatposterior3D.resize(dimensions[0],vector<double> (dimensions[1],-HUGE_VAL));

		    for(int x=0;x<dimensions[0];x++){
			   for(int y=0;y<dimensions[1];y++){
				  for(int v=0;v<dimensions[2];v++){
					 actionDistribute.at(x).at(y).at(v) -= sum;
					 if (isnan( actionDistribute.at(x).at(y).at(v) )) 
					   continue;
					 if ( actionDistribute.at(x).at(y).at(v)==-HUGE_VAL ) 
					   continue; 

					 flatposterior3D.at(x).at(y) = LogAdd(flatposterior3D.at(x).at(y),
								 actionDistribute.at(x).at(y).at(v));
				  }  
			    }
		    }

			return sum;
		}

    	double pre_GenPosterior(vector<double>& cost){
			
			cout<<"Preprocess for posterior"<<endl;
			double sum = -HUGE_VAL;
			double prior_weight = -log(dimensions[0]*dimensions[1]*dimensions[2]);
			actionDistribute.clear();
			actionDistribute.resize(dimensions[0], vector<vector<double> > (
					dimensions[1],vector<double>(dimensions[2],0.0)));

		    for(int x=0;x<dimensions[0];x++){
			   for(int y=0;y<dimensions[1];y++){
				 for(int v=0;v<dimensions[2];v++){
						 actionDistribute.at(x).at(y).at(v) = cost[2]+
                           nr_Distribute.at(x).at(y).at(v) + prior_weight;
					 sum = LogAdd(sum,actionDistribute.at(x).at(y).at(v));
				 }
			   }
		    }
		 
            //cout<<"Observation SUM: "<<sum<<endl;
		    //cout<<" 3D posterior"<<endl;
            flatposterior3D.clear();
		    flatposterior3D.resize(dimensions[0],vector<double> (dimensions[1],-HUGE_VAL));

		    for(int x=0;x<dimensions[0];x++){
			   for(int y=0;y<dimensions[1];y++){
				  for(int v=0;v<dimensions[2];v++){
					 flatposterior3D.at(x).at(y) = LogAdd(flatposterior3D.at(x).at(y),
								 actionDistribute.at(x).at(y).at(v));
				  }  
			    }
		    }

			return sum;
		}

	    inline void assign_Posterior(pair<int,int>& robot, double cost,
					double& sum){
			double r_prob = -HUGE_VAL;
			double prior_weight = -log(dimensions[0]*dimensions[1]*dimensions[2]);
			for(int v=0;v<dimensions[2];v++){
					 r_prob = LogAdd(r_prob,cost+
						r_Distribute.at(robot.first).at(robot.second).at(v))+prior_weight;
			}  
			sum = LogAdd(sum,r_prob);
			//sum = log(exp(sum)-exp(flatposterior3D.at(robot.first).at(robot.second)));
		    sum = LogSubtract(sum,flatposterior3D.at(robot.first).at(robot.second));
			flatposterior3D.at(robot.first).at(robot.second) = r_prob;

		}
		    
		inline void dismiss_Posterior(pair<int,int>& robot, double cost,
					double& sum){
			double nr_prob = -HUGE_VAL;
			double prior_weight = -log(dimensions[0]*dimensions[1]*dimensions[2]);
			for(int v=0;v<dimensions[2];v++){
					 nr_prob = LogAdd(nr_prob,cost+
						nr_Distribute.at(robot.first).at(robot.second).at(v))+prior_weight;
			}  
			sum = LogAdd(sum,nr_prob);
			//sum = log(exp(sum)-exp(flatposterior3D.at(robot.first).at(robot.second)));
			sum = LogSubtract(sum,flatposterior3D.at(robot.first).at(robot.second));
			flatposterior3D.at(robot.first).at(robot.second) = nr_prob;
		}	

		void writeOutput(vector<pair<int,int> >& traj, int index, 
					vector<double>&Beliefs, double tick,
					pair<int,int>& rob,int traj_ind,ofstream& file){  
			 
			  cout<<"Write output"<<endl;
			  vector<pair<int, int> > subTraj;
              subTraj.insert(subTraj.end(), traj.begin(), traj.begin()+index);
      
              BMPFile gridView(dimensions[0], dimensions[1]);
		      
			  char buf[512];
			  double maxV = -HUGE_VAL;
              double minV = HUGE_VAL;
		      /*
			  for(int ii=0; ii< dims.first; ii++) { 
                  for (int jj=0; jj < dims.second; jj++) {
					maxV = max(maxV, posterior2D.at(ii).at(jj));
					minV = min(minV, posterior2D.at(ii).at(jj));
                  }
			  }*/

		      gridView.addBelief(posterior2D, -50, -0.0, white,red);
              grid.addObstacles(gridView, black);
              gridView.addVector(subTraj, blue, 1);
		      gridView.addLabel(rob,green);
              sprintf(buf, "../compare/ppp%03d-%2f.bmp", traj_ind, tick); 
              gridView.write(buf);

			  /*
			   * maxV = -HUGE_VAL;
              minV = HUGE_VAL;
              for (int ii=0; ii < dims.first; ii++) { 
		  		for (int jj=0; jj < dims.second; jj++) {
					maxV = max(maxV, flatposterior3D.at(ii).at(jj));
					minV = min(minV, flatposterior3D.at(ii).at(jj));
				}
			  }
			  cout<<"max: "<<maxV<<" min: "<<minV<<endl;
			  */

		      gridView.addBelief(flatposterior3D, -50, -0.0, white,red);
              grid.addObstacles(gridView, black);
              gridView.addVector(subTraj, blue, 1);
		      gridView.addLabel(rob,green);
              sprintf(buf, "../compare/intent%03d-%2f.bmp", traj_ind, tick); 
              gridView.write(buf);


              cout<<" Write file "<<endl;
		      gridView.addLabel(rob,green);
              cout <<"BELIEFS: "<<index <<" "<<Beliefs[0]
			    <<" "<<Beliefs[1]<<" "<<Beliefs[2]
				<< " POSTEIOR: "<<posterior2D.at(rob.first).at(rob.second)
				<<" "<<flatposterior3D.at(rob.first).at(rob.second)<<endl;

		      file <<tick<<" "<<posterior2D.at(rob.first).at(rob.second)
				<<" "<<flatposterior3D.at(rob.first).at(rob.second)<<endl;
			  
			  cout<<"DONE"<<endl;
		
		}
		
		void writeOutput(vector<pair<int,int> >& userTraj, vector<pair<int,int> >&
					robTraj, double tick,pair<int,int>& rob,pair<int,int>& next,
					double& sum,int traj_ind, ofstream& file){  
			 
			  cout<<"Write output"<<endl;
              BMPFile gridView(dimensions[0], dimensions[1]);
		      
			  char buf[512];
              grid.addObstacles(gridView, black);
              gridView.addVector(userTraj, blue, 1);
              gridView.addVector(robTraj, magenta, 1);
		      gridView.addLabel(rob,green);
			  if (next.first!=0&&next.second!=0)
                 gridView.addLabel(next,red);
              sprintf(buf, "../compare/activeintent%03d-%2f.bmp", traj_ind, tick); 
              gridView.write(buf);


              cout<<" POSTEIOR: "<<flatposterior3D.at(rob.first).at(rob.second)-sum<<endl;
		      file <<tick<<" "<<flatposterior3D.at(rob.first).at(rob.second)-sum<<endl;
			  
			  cout<<"DONE"<<endl;
		
		}


		/* Data */
	    Grid& grid;
		vector<vector<double> > posterior2D;
        vector<vector<vector<double> > > actionDistribute;
        vector<vector<vector<double> > > r_Distribute;
        vector<vector<vector<double> > > nr_Distribute;
        vector<vector<double> > flatposterior3D;
		//pedestrian prediction model
        OrderedWaveInferenceEngine& p_engine;
        //intent model towards robot
        DisSeqOrderInferEngine& r_engine;
        //intent model towards other locations
        DisSeqOrderInferEngine& nr_engine;               
		vector<PosFeature> posFeatures;
	    DisVecSeqFeature &seqFeat;
		Parameters& p_Params;
		Parameters& r_PosParams;
		Parameters& r_SeqParams;
		Parameters& nr_PosParams;
		Parameters& nr_SeqParams;   
		vector<int> dimensions;
};


int main(int argc, char **argv) {
   OptionParser opts;

   string mapFile,trainFile,testFile;

   int factor;

   opts.addOption(new StringOption("map", 
            "--map <filename>                 : map file",
            "../input/grid.bmp", mapFile, false));
/*
   opts.addOption(new StringOption("trainevid", 
            "--trainevid <filename>            : evidence file",
            "", trainFile, true));
*/
   opts.addOption(new StringOption("evidence", 
            "--test evidence <filename>            : evidence file",
            "", testFile, true));

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

   
   Evidence testSet(testFile, grid, factor);
 //  Evidence trainSet(trainFile, grid, factor);

   pair<int, int> dims = grid.dims();
   pair<int, int> lowDims((int)ceil((float)dims.first/factor),
         (int)ceil((float)dims.second/factor));
   
   cout << " Speed Feature"<<endl;
   vector<double> speedTable(VEL_DIM,0.0);
   speedTable.at(1) = 0.75;
   DisVecSeqFeature speedfeat(speedTable);

   vector<int> dimensions;
   dimensions.push_back(dims.first);
   dimensions.push_back(dims.second);
   dimensions.push_back(3);
   
   vector<double> p_weights(NUMPOSFEAT,-0.0);
   p_weights.at(0) = -1.0; //-2.23 for PPP forecast
   p_weights.at(1) = -6.2;
   p_weights.at(2) = -0.35;
   p_weights.at(3) = -2.73;
   p_weights.at(4) = -0.92;
   p_weights.at(5) = -0.26;
   vector<double> r_PosWeights(NUMPOSFEAT+NUMROBFEAT, -0.0);
   r_PosWeights.at(0) = -3.83;
   r_PosWeights.at(1) = -8.36;
   r_PosWeights.at(2) = -2.65;
   r_PosWeights.at(3) = -5.43;
   r_PosWeights.at(4) = -3.15;
   r_PosWeights.at(5) = -3.30;
   //r_PosWeights.at(6) =  0.60;
   //r_PosWeights.at(7) =  0.45;

   vector<double> nr_PosWeights(NUMPOSFEAT+NUMROBFEAT, -0.0);
   nr_PosWeights.at(0) = -4.51;
   nr_PosWeights.at(1) = -6.2;
   nr_PosWeights.at(2) = -0.35;
   nr_PosWeights.at(3) = -2.73;
   nr_PosWeights.at(4) = -0.93;
   nr_PosWeights.at(5) = -0.28;
   //nr_PosWeights.at(6) = -0.50;
   //nr_PosWeights.at(7) = -0.286;
             
   vector<double> r_SeqWeights(VEL_DIM, -0.0);
   r_SeqWeights.at(0) = 0.59;
   r_SeqWeights.at(1) = -0.83;
   vector<double> nr_SeqWeights(VEL_DIM, -0.0);
   nr_SeqWeights.at(0) = -1.21;
   nr_SeqWeights.at(1) = 0.49;

   Parameters p(p_weights);
   Parameters r_Pos(r_PosWeights);
   Parameters nr_Pos(nr_PosWeights);
   Parameters r_Seq(r_SeqWeights);
   Parameters nr_Seq(nr_SeqWeights);


   OrderedWaveInferenceEngine engine_p(grid, InferenceEngine::GRID8);
   DisSeqOrderInferEngine engine_intent_r(8,InferenceEngine::GRID8);
   DisSeqOrderInferEngine engine_intent_nr(8,InferenceEngine::GRID8);

   IntentRecognizer IR(grid,p,r_Pos,r_Seq,nr_Pos,nr_Seq,speedfeat,
			   engine_p,engine_intent_r,engine_intent_nr);

   cout << testSet.size() <<" Examples"<<endl;

   for (int i=0; i < testSet.size(); i++) {

      vector<pair<int, int> > & traj = testSet.at(i);
	  vector<double> & vels = testSet.at_v(i);
      vector<double> times = testSet.getTimes(i); 
	  pair<int,int> & snackbot = testSet.at_bot(i);
      
	  IR.forecast(traj,vels,times,snackbot,i,1.0);
      
   }
}


