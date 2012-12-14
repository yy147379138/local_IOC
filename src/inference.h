#ifndef INFERENCE_H__
#define INFERENCE_H__

#include "main.h"
#include "grid.h"
#include "features.h"
#include "evidence.h"

class State{
	public:
		State(pair<int,int> _pos, int _disV)
			:pos(_pos),disV(_disV){}
		State(int x,int y, int _disV):disV(_disV){
		   pos.first = x;
		   pos.second = y;
		}
		State (const State& _s):pos(_s.pos),disV(_s.disV){}
		int x(){return pos.first;}
		int y(){return pos.second;}
		pair<int,int> pos;
		int disV;
	private:

};

class Parameters {
   public:
      Parameters(vector<double> &_weights);
      Parameters(){}
      double reward(vector<double> &features);
      double reward(vector<int> &features);
      double & at(int i) { return weights.at(i); }
	  double & back() { return weights.back(); }
      int size() { return weights.size(); }
      operator vector<double>&()  { return weights; }
   protected:
      vector<double> weights;
};

class RewardMap {
   public:
      RewardMap(FeatureArray &posfeatures, Parameters &params);
      RewardMap(FeatureArray &posfeatures, SeqFeature 
				  &seqfeat, Parameters &params);
      RewardMap(FeatureArray &posfeatures, SeqFeature 
				  &seqfeat, Parameters &p_params, Parameters &s_params);
      double at(int i, int j) { return posvalues.at(i).at(j); }
      // reward for time step i
	  //double at(int i){ return seqvalues.at(i); }
	  // reward for discrete variable i
	  double reward(int i){ return lookupTable.at(i); }
	  // reward for a complete state
	  double at(State &s){ 
		  return at(s.pos.first,s.pos.second)+reward(s.disV); }
	  double at(State *s){ 
		  return at(s->pos.first,s->pos.second)+reward(s->disV); }
	  double at(int x, int y, int v){
		  return at(x,y)+reward(v);
	  }
      // order without considering the sequential feature
      void dijkstras(pair<int, int> origin, vector<vector<double> > &costs,
            vector<pair<int, int> > &order, double & maximum);
      void dijkstras(State& origin, vector<vector<vector<double> > > &costs,
            vector<State> &order, double & maximum);
	  pair<int, int> dims() { 
         pair<int, int> _dims(0,0);
         if (posvalues.size() > 0) {
            _dims.first = posvalues.size();
            _dims.second = posvalues.at(0).size();
         }
         return _dims;
      }
	  int V_dim(){
		  return lookupTable.size();
	  }
   protected:
	  /*  The total reward of posfeaturs for every location in 
	   *  the environment */
      vector<vector<double> > posvalues;
	  /* rewards lookuptable for decrete sequential variable*/
	  vector<double> lookupTable;
};

class InferenceEngine {
   public:
      // TODO:  Implement GRID4 version
      enum {GRID4, GRID8};

      InferenceEngine(int _connections=GRID8) 
         : connections(_connections) { }

      virtual void forward(pair<int, int> pos, int T, RewardMap &rewards, 
            vector<vector<vector<double> > > &partition);

      virtual void backward(pair<int, int> pos, int T, RewardMap &rewards,
            vector<vector<double> > &initial,
            vector<vector<vector<double> > > &partition); 

      inline void selectProp(int x, int y, pair<int, int> dim, 
            vector<vector<double> > &from,
            vector<vector<double> > &to1, 
            vector<vector<double> > &to2, 
            vector<vector<bool> > &mask, double reward);

      inline void selectBackProp(int x, int y, pair<int, int> dim, 
            vector<vector<double> > &from,
            vector<vector<double> > &to1, 
            vector<vector<double> > &to2, 
            vector<vector<bool> > &mask, RewardMap &rewards);

      inline void propagate(int x, int y, pair<int, int> dim, 
            vector<vector<double> > &from,
            vector<vector<double> > &to, double reward); 

      inline void invpropagate(int x, int y, pair<int, int> dim, 
            vector<vector<double> > &from,
            vector<vector<double> > &to, RewardMap &rewards);

      void flattenPartitions(vector<vector<vector<double> > > &origPartition,
            vector<vector<double> > &collapsePartition);

      void cummulativeFlattenPartitions(
            vector<vector<vector<double> > > &origPartition,
            vector<vector<vector<double> > > &collapsePartition);

      void combinePartitions(
            vector<vector<vector<double> > > &partition1,
            vector<vector<vector<double> > > &partition2,
            vector<vector<double> > &partitionOut);


      void edgeprop(vector<vector<double> > &from,
            vector<vector<double> > &to,
            int x0, int y0, int x1, int y1, double reward);

      void timePartitions(vector<vector<double> > &frequencies,
            vector<vector<vector<double> > > &timeFrequencies,
            vector<vector<double> > &optCost,
            double veloc, double sigma0, double sigma1, int T);

   protected:
      int connections;
};


class WaveInferenceEngine : public InferenceEngine {
   public:
      WaveInferenceEngine(int _connections) 
         : InferenceEngine(_connections) { }

      virtual void forward(pair<int, int> pos, int T, RewardMap &rewards,
            vector<vector<vector<double> > > &partition);

      virtual void backward(pair<int, int> pos, int T, RewardMap &rewards,
            vector<vector<double> > &initial,
            vector<vector<vector<double> > > &partition);

   protected:
};

class OrderedWaveInferenceEngine : public WaveInferenceEngine {
   public:
      OrderedWaveInferenceEngine(int _connections)
         : WaveInferenceEngine(_connections) { }

      void forward(pair<int, int> pos, int T, RewardMap &rewards,
            vector<vector<vector<double> > > &partition);

      void backward(pair<int, int> pos, int T, RewardMap &rewards,
            vector<vector<double> > &initial,
            vector<vector<vector<double> > > &partition); 

};

class Predictor {
   public:
      Predictor(Grid &_grid, RewardMap &_rewards, InferenceEngine &_engine);
      void setStart(pair<int, int> _start);
      void setPrior(vector<vector<double> > &_prior);
      double predict(pair<int, int> current, 
            vector<vector<double> > &occupancy);
      vector<vector<double> > & getPosterior() { return posterior; }
      vector<vector<vector<double> > > & getTimeOccupancy() { 
         return timeOccupancy; }
	  double getNormalizer(pair<int,int> & dest){
		  return forwardFlatPartitionB.at(dest.first).at(dest.second)-
			  forwardFlatPartitionA.at(dest.first).at(dest.second);
	  }
	  pair<double,double> check(pair<int,int> & dest){
		  pair<double,double> v (forwardFlatPartitionB.at(dest.first).at(dest.second),
			  forwardFlatPartitionA.at(dest.first).at(dest.second));
          return v;
	  }
   protected:
      Grid &grid;
      RewardMap &rewards;
      InferenceEngine &engine; 

      vector<vector<double> > prior, posterior;
      pair<int, int> start;
      vector<vector<vector<double> > > forwardPartitionA;
      vector<vector<double> > forwardFlatPartitionA;
      vector<vector<vector<double> > > forwardPartitionB; 
      vector<vector<double> > forwardFlatPartitionB; 
      vector<vector<vector<double> > > backwardPartition; 

      vector<vector<vector<double> > > timeOccupancy;
};

class DisSeqOrderInferEngine {
   public:
      // TODO:  Implement GRID4 version
      enum {GRID4, GRID8};

      DisSeqOrderInferEngine(int _itrTimes, int _connections=GRID8)
		  :itrTimes(_itrTimes),connections(_connections){}

      void forward(State &state, RewardMap &rewards, 
           vector<vector<vector<vector<double> > > > &partition);

      void backward(State &state,  RewardMap &rewards,
           vector<vector<vector<double> > > &initial,
           vector<vector<vector<vector<double> > > > &partition); 

      inline void selectProp(State& s, 
           vector<vector<vector<double> > > &from,
           vector<vector<vector<double> > > &to1, 
           vector<vector<vector<double> > > &to2, 
           vector<vector<vector<bool> > > &mask, RewardMap &rewards);

      inline void selectBackProp(State& s, 
           vector<vector<vector<double> > > &from,
           vector<vector<vector<double> > > &to1, 
           vector<vector<vector<double> > > &to2, 
           vector<vector<vector<bool> > > &mask, RewardMap &rewards);
      
	  void flattenPartitions
		  (vector<vector<vector<vector<double> > > > &origPartition,
           vector<vector<vector<double> > > &collapsePartition);

      void cummulativeFlattenPartitions(
			vector<vector<vector<vector<double> > > > &origPartition,
            vector<vector<vector<vector<double> > > > &collapsePartition);

      void combinePartitions(
            vector<vector<vector<vector<double> > > > &partition1,
            vector<vector<vector<vector<double> > > > &partition2,
            vector<vector<vector<double> > > &partitionOut);


      void timePartitions(vector<vector<vector<double> > > &frequencies,
           vector<vector<vector<vector<double> > > > &timeFrequencies,
           vector<vector<double> > &optCost,
            double veloc, double sigma0, double sigma1, int T);

   protected:
      int connections;
	  int itrTimes;
};

class DisSeqPredictor {
   public:
      DisSeqPredictor(Grid &_grid, RewardMap &_rewards, DisSeqOrderInferEngine &_engine);
	  /*
	  ~DisSeqPredictor(){
		  cout<<"Clear vectors"<<endl;
		  dimensions.clear();
		  prior.clear();
		  posterior.clear();
		  obdistribute.clear();
		  forwardFlatPartitionA.clear();
		  forwardFlatPartitionB.clear();
		  backwardPartition.clear();
		  cummBackPartition.clear();
		  initBackWeight.clear();
		  forwardPartitionA.clear();
		  forwardPartitionB.clear();
      }*/
      void setStart(State & start);
      void setPrior(vector<vector<vector<double> > > &_prior);
      double forwardBackwardInference(State &current, 
            vector<vector<vector<double> > > &occupancy);
	  void forwardInference(State &current);

      vector<vector<vector<double> > > & getPosterior() { return posterior; }
      vector<vector<vector<double> > > & getObdistribute() { return obdistribute; }
      vector<vector<double> > getFlatPosterior();
      //vector<vector<vector<double> > > & getTimeOccupancy() { 
        // return timeOccupancy; }
      void flattenOccupancy (vector<vector<vector<double > > >
		&origOccupancy,vector<vector<double> > &flatOccupancy);
	  double getNormalizer (State & dest){
		  double norm = -HUGE_VAL;
		  for (int v=0;v < dimensions[2];v++){
			  norm = LogAdd(norm,
			  forwardFlatPartitionB.at(dest.x()).at(dest.y()).at(v)-
			  forwardFlatPartitionA.at(dest.x()).at(dest.y()).at(v));
		  }
		  return norm;
	  }
   protected:
      Grid &grid;
      RewardMap &rewards;
      DisSeqOrderInferEngine &engine;  
	  vector<int> dimensions;

      vector<vector<vector<double> > > prior, posterior;
      //State start;
	  //Partition is actually logZ (log partition) i.e. state value
	  /* Forward propagation is from s0 to goal (everywhere) along
	   * the optimal paths. The updating rule is 
	   * V(s_t+1) = V(s_t) + reward(s_t)
	   * The value represent the maximum rewards obtained by a trajectory 
	   * reaching state s
	   *
	   * Backward propagation is from far points backward to s0
	   * The updating rule is V(s_t-1) = V(s_t) + reward(s_t-1)
	   * Normal value iteration is backwards
	
	   * */
	  // FOUR dimension
      vector<vector<vector<vector<double> > > > forwardPartitionA;
	  /* PartitionA is from s1(trajectory start state) to everywhere
	   * i.e. V(s1->G) */
      vector<vector<vector<double> > > forwardFlatPartitionA;
      vector<vector<vector<vector<double> > > > forwardPartitionB; 
      /* PartitionB is from st(current predicition state) to 
	   * everywhere, i.e. V(st->G) */
	  vector<vector<vector<double> > > forwardFlatPartitionB;
	  //FOUR dimension
      vector<vector<vector<vector<double> > > > backwardPartition; 
      //vector<vector<vector<double> > > timeOccupancy;
	  vector<vector<vector<vector<double> > > > cummBackPartition;
	  vector<vector<vector<double> > > initBackWeight;
	  /* Observation distribution, which is the normalizer i.e.
	   * V(S_T->G)-V(S_1->G)
	   * */
	  vector<vector<vector<double> > > obdistribute;
	
	
};

class MarkovModel {
   public:
      MarkovModel(Grid &grid, Evidence &evidence);

      double eval(vector<pair<int, int> > &path);

      double evalNext(vector<pair<int, int> > &path, pair<int, int> next,
            int histSize=10);
   protected:
      bool matchLast(vector<pair<int, int> > &a, 
            vector<pair<int, int> > &b, int N); 
      vector<vector<vector<pair<int, int> > > > instances;
      vector<vector<pair<int, int> > > paths;
};


#endif
