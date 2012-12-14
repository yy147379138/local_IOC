#ifndef INTENT_H__
#define INTENT_H__

#include "main.h"
#include "visualize.h"
#include "inference.h"
#include "linearquadratic.h"
#include <queue>


#ifndef VEL_DIM
#define VEL_DIM 2
#endif

#ifndef NUMROBFEAT
#define NUMROBFEAT 0
#endif

#ifndef PI
#define PI 3.1415927
#endif

class Planner{
	public:
		Planner(Grid& _grid,RewardMap& _rewards):grid(_grid),
		    rewards(_rewards){}
		void simplePlan(vector<pair<int,int> >& pool,
					pair<int,int>& currentRob,int radius);
		void sociablePlan(vector<pair<int,int> >& pool,
               pair<int,int>& currentRob,int low, int high);
		double computeKLDivergence(vector<vector<double> >& novel,
			vector<vector<double> >& base);
		double computeEntropy(vector<vector<double> >& P);
		double computeEntropy(vector<vector<double> >& P, double& sum);
		void planSimplePath(vector<pair<int,int> > & path, 
					pair<int,int> dest);
	private:
		void writePlans(vector<pair<int,int> >& pool);
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
#if 0
		IntentRecognizer(Grid& _grid,Parameters& _p_Params,Parameters& 
			_r_PosParams,Parameters& _r_SeqParams,Parameters&
			_nr_PosParams,Parameters& _nr_SeqParams,DisVecSeqFeature&
			_seqFeat)
            :grid(_grid),p_Params(_p_Params),r_PosParams(_r_PosParams), 
            r_SeqParams(_r_SeqParams),nr_PosParams(_nr_PosParams),
            nr_SeqParams(_nr_SeqParams),seqFeat(_seqFeat),A(initM1),
            B(initM2),Sigma(initM1),r_M(initM1),nr_M(initM1){

            initialize();

            dimensions.push_back(seqFeat.num_V());
            pp_engine = new OrderedWaveInferenceEngine(InferenceEngine::GRID8);
            mdpr_engine = new DisSeqOrderInferEngine(8,InferenceEngine::GRID8);
            mdpnr_engine = new DisSeqOrderInferEngine(8,InferenceEngine::GRID8);
		}
#endif
		IntentRecognizer(Grid& _grid,Parameters& _p_Params,Parameters& 
			_r_PosParams,Parameters& _r_SeqParams,Parameters&
			_nr_PosParams,Parameters& _nr_SeqParams,DisVecSeqFeature&
			_seqFeat,OrderedWaveInferenceEngine& _pp_engine,
			DisSeqOrderInferEngine& _mdpr_engine, 
			DisSeqOrderInferEngine& _mdpnr_engine, 
			LQControlInference& _lq_engine):grid(_grid),
			p_Params(_p_Params),r_PosParams(_r_PosParams), 
            r_SeqParams(_r_SeqParams),nr_PosParams(_nr_PosParams),
            nr_SeqParams(_nr_SeqParams),seqFeat(_seqFeat),
			pp_engine(_pp_engine),mdpr_engine(_mdpr_engine),
			mdpnr_engine(_mdpnr_engine),lq_engine(_lq_engine){

            initialize();
            dimensions.push_back(seqFeat.num_V());
            //lq_engine = new LQControlInference(A,B,Sigma,r_M,*convertor);
            //lqnr_engine = new LQControlInference(A,B,Sigma,nr_M,*convertor);
            //lqnr_engine->valueInference();
		}

		void discrtForecast(vector<pair<int,int> >& traj, vector<double>& vels,
			 vector<double>& times, pair<int,int>& rob,int evid_i,
			 double interval);
		void hybridForecast(vector<pair<int,int> >& traj,vector<double>& vels, 
             vector<pair<double,double> >& rawObs,vector<double>& times,
			 vector<double>& rawTimes,pair<int,int>& borinGrid,
			 pair<double,double>& botinReal,int evid_i,double interval);
		void combineForecast(vector<pair<int,int> >& traj,vector<double>& vels, 
             vector<pair<double,double> >& rawObs,vector<double>& times,
			 vector<double>& rawTimes,pair<int,int>& borinGrid,
			 int evid_i,double interval);
		void active_forecast(vector<pair<int,int> >& traj, vector<double>& vels,
			 vector<double>& times, pair<int,int>& rob,int evid_i,
			 double interval);

        enum{PP,DIS,CONT,HYBRID};
	private:
        /* Methods */
        void initialize(){
            pair<int, int> dims = grid.dims();
            dimensions.push_back(dims.first);
            dimensions.push_back(dims.second);
            /* Initialize position-based features */
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
	    bool turnningDetector(vector<pair<double,double> >& obs, int index,
					int prev_index, int restart);
		inline double product(pair<double,double>& v1, pair<double,double>& v2){
			return (v1.first*v2.first+v1.second*v2.second)/
				sqrt((v1.first*v1.first+v1.second*v1.second)*
							(v2.first*v2.first+v2.second*v2.second));
		}
        double genPosterior(vector<vector<double> >& prior,
        vector<vector<double> >& posterior,vector<vector<double> >& support);
		double genIntentPosterior(pair<int,int>& robot, vector<double>& cost);
    	double pre_GenPosterior(vector<double>& cost);
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

		double entropy(vector<vector<double> >& P);
		void writeOutput(vector<pair<int,int> >& traj, int index, 
					double tick,pair<int,int>& rob,double V,double D,int traj_ind,ofstream& file);
		void writeOutput(vector<pair<int,int> >& userTraj, vector<pair<int,int> >&
					robTraj, double tick,pair<int,int>& rob,pair<int,int>& next,
					double& sum,int traj_ind, ofstream& file);

		/* Data */
	    Grid& grid;
		vector<vector<double> > posterior2D;
        vector<vector<vector<double> > > actionDistribute;
        vector<vector<vector<double> > > r_Distribute;
        vector<vector<vector<double> > > nr_Distribute;
        vector<vector<double> > flatposterior3D;
        vector<vector<double> > lqposterior;
        vector<vector<double> > hybridposterior;
		vector<PosFeature> posFeatures;
        //pedestrian prediction model
        OrderedWaveInferenceEngine& pp_engine;
        //Discrete intent model
        DisSeqOrderInferEngine& mdpr_engine;
        //Discrete non-robot intent model 
        DisSeqOrderInferEngine& mdpnr_engine;
        ContinuousState* convertor;
        //Continuous robot intent model
        LQControlInference& lq_engine;
        //Continuous non-robot model
        //LQControlInference* lqnr_engine;
	    DisVecSeqFeature &seqFeat;
        /*Discrete parameters*/
		Parameters& p_Params;
		Parameters& r_PosParams;
		Parameters& r_SeqParams;
		Parameters& nr_PosParams;
		Parameters& nr_SeqParams;   
		vector<int> dimensions;
		JetColorMap jet;
};


#endif
