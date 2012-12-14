#ifndef LINEARQUADRATIC_H__
#define LINEARQUADRATIC_H__

#include <map>
#include "main.h"
#include "evidence.h"
#include "mymath.h"
#include <cassert>


class ContinuousState{
	public:
		ContinuousState():last_time(0.0){}
		void convertState(vector<VectorXd>& states,
					vector<pair<double,double> >& rawdata,
					const pair<double,double>& target);
		void convertState(vector<VectorXd>& states,
					vector<pair<double,double> >& rawdata, 
					const pair<double,double>& target,
					pair<double,double>& origin);
		void empiricalExpectation(vector<VectorXd>& seq, M_6& EE);
		void empiricalExpectation(vector<pair<double,double> >& rawTraj,
				pair<double,double>& target,M_6& EE);
	private:
		double last_time;
};

class LQControlInference{
	public:
		LQControlInference(M_6& _A, M_6_2& _B, M_6& _Sigma, M_6& _M,
				ContinuousState& _cState):A(_A), B(_B),Sigma(_Sigma),
		        M(_M),cState(_cState),valid_params(false){
				C.setZero();
				D.setZero();
				G.setZero();
				F.setZero();
				resetInference();
		}
		~LQControlInference(){
			resetDistribution();
		}
		void valueInference(int T=50);
		void setM(M_6& _M){ 
			valid_params = false;
			M =_M;
			valueInference(50);
		}
		void setA(M_6& _A){
			valid_params = false;
			A =_A;
		}
		void setB(M_6_2& _B){
			valid_params = false;
			B =_B;
		}
		void setSigma(M_6& _S){ 
			valid_params = false;
			Sigma =_S;
		}
		M_6& getM(){ return M;}
		M_6& getA(){ return A;}
		M_6_2& getB(){ return B;}
		M_6& getSigma(){ return Sigma;}

		inline double getQ(VectorXd& a, VectorXd& s);
		inline double getV(VectorXd& s);
		void computeDistribution(VectorXd& s0, int T);
		double cumulativeCosts(vector<VectorXd>& s_sequence,int index,
					M_6& quadraticSum);
		double costTransformation(M_6& quadraticSum, Col_6& linearSum,
					M_6& R, Col_6& d,int length);
		void quadraticExpectation(M_6& E);
		void forwardInference(vector<pair<double,double> >& traj,
				int prev,int current, map<pair<double,double>,
				double>& likelihoods, int start = 0);
		double forwardInference(vector<VectorXd>& externalStates);
		double forwardInference(vector<pair<double,double> >& traj,
			   int prev,int current,pair<double,double>& point);
		void stateMoments(int prev);
		void stateMoments(vector<VectorXd>& states);
		double sequenceLikelihood(const pair<double,double>& from, 
				const pair<double,double>& to,int length);
		double sequenceLikelihood(vector<VectorXd>& states);
		double determinantLikelihood(vector<VectorXd>& states);

		void setOrigin(pair<double,double>& _origin){
			Origin = make_pair(_origin.first,_origin.second);
			resetInference();
		}
		/*Check the state transformation from target to another*/
        void checkStateTransform(vector<VectorXd>& fromStates,
					vector<VectorXd>& toStates, 
					const pair<double,double>& from,
					const pair<double,double>& to);
		ContinuousState& cState; 
	protected:
		/*  linear dynamics parameters */
		M_6& A;
		M_6_2& B;
        M_6& Sigma;
        /* value parameters */
		M_8 C;
		M_6 D;
		M_6& M;
		Col_6 G;
		Col_8 F;
		bool valid_params;
		/* Distributions  */
		vector<Gaussian* > state_distribution;
		vector<Gaussian* > action_distribution;
        /* State relevent data */
	  /* converting function for my data */
		M_6 stateQuadrSum;
		Col_6 stateLinSum;
		M_2 actionQuadrSum;
		Col_2 actionLinSum;
		M_2_6 crossSum; /* SUM_t at*st^T */
		vector<VectorXd> firstState_seq;
		/** The origin of the traj, set before inference
		 * by the setOrigin from outer level predictor */
		pair<double,double> Origin;

		void resetInference();
		void resetDistribution();

};

class LQControlOptimizer{
	public:
		enum{STOCHASTIC_EXP,BATCH_EXP,SMALL_BATCH_EXP,
			STOCHASTIC_LINEAR};
		LQControlOptimizer(LQControlInference& _lq,ContinuousState& _cs,
					Evidence& _evid):lq(_lq),cs(_cs),evid(_evid),
		            maxLength(0){
			for(int i=0;i<evid.size();i++){
				vector<VectorXd> s_traj;
				cs.convertState(s_traj,evid.at_raw(i),
							evid.at_raw(i).back());
				statesSet.push_back(s_traj);
				if(s_traj.size()>maxLength)
				  maxLength = s_traj.size();
			}
			cout<<"Optimize over: "<<statesSet.size()
				<<" examples"<<endl;
		}
		
		/* Varying learning rate */
		double optimize_v(double step, int itrTimes, double thresh,
					int mode = STOCHASTIC_EXP);
		/*  backtrack function */
		double backtrack(int itr_times, double thresh, 
					double alpha, double beta, double mini_step, 
					int mode = STOCHASTIC_EXP);
	private:
		void testSeqLikelihood(vector<VectorXd>& states);
		void get_gradient(M_6& gradient, vector<VectorXd>& states);
		void update_M(M_6& gradient, M_6& diff_M, double step);
		void get_next_M(M_6& gradient, M_6& original_M, M_6& next_M, double step);
		void write_obj(double obj, int itr, ofstream& file);

		LQControlInference& lq;
		ContinuousState& cs;
		Evidence& evid;
		vector<vector<VectorXd> > statesSet;
		int maxLength;

};


class LQContinuousPredictor{
	public:
		LQContinuousPredictor(LQControlInference& _lqEngine, Grid& _grid):
			lqEngine(_lqEngine),grid(_grid){
			dims = grid.dims();
			for(int x=0;x<dims.first;x++){
				for(int y = 0;y<dims.second;y++){
					pair<double,double> realVals = 
						grid.grid2Real(x,y);
					continuousLikelihoods[realVals] = 0.0;
					mapping[make_pair(x,y)] = realVals;
				}
			
			}
			assert(continuousLikelihoods.size()==dims.first*dims.second);
			assert(mapping.size()==dims.first*dims.second);
			prior.resize(dims.first,vector<double>(dims.second,
						-HUGE_VAL));
			posterior.resize(dims.first,vector<double>(dims.second,
							-HUGE_VAL));
			gridLikelihoods.resize(dims.first,
						vector<double>(dims.second,-HUGE_VAL));
		}
		vector<vector<double> >& getPosterior(){ return posterior; }
		vector<vector<double> >& getLikelihoods(){ 
			return gridLikelihoods;}
        void setPrior(vector<vector<double> > &_prior);
		void predictAll(vector<pair<double,double> >& observation,
					int prev, int current, int start = 0);
		double predictPoint(vector<pair<double,double> >& observation,
					pair<double,double>& target,int prev, int current);
		void setOriginWrapper(pair<double,double>& _origin){
			lqEngine.setOrigin(_origin);
		}

		/*Check the optimized computation of sequence likelihoods
		 * for all targets in the set by comparing with brute-force
		 * method
		 * */
		void test(vector<pair<double,double> >& obs);
		void testAfterPredict(vector<pair<double,double> >& obs,
					int current);
		
	private:
		Grid& grid;
		LQControlInference& lqEngine;
		vector<vector<double> > prior, posterior,gridLikelihoods;
		map<pair<double,double>,double> continuousLikelihoods;
		map<pair<int,int>,pair<double,double> > mapping;
		pair<int,int> dims;
		
};

#endif
