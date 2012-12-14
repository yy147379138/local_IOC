#include "inference.h"
#include <queue>
#include <cassert>
#include <new>

void DisSeqOrderInferEngine::selectProp(State& s,
      vector<vector<vector<double> > > &from, vector<vector<vector<double> > > &to1, 
      vector<vector<vector<double> > > &to2, vector<vector<vector<bool> > > &mask,
      RewardMap &rewards) {
      // called in forward();
	  // selectProp(x0,i,partition.at(t),partition.at(t),partition.at(t+1),mask,rewards.at(x0).at(i))
   double diagM = 1.41;
   pair<int,int> dim = rewards.dims();
   int D = rewards.V_dim();
   int x = s.x();
   int y = s.y();
   int v = s.disV;

   if (x < 0 || y < 0 || x >= dim.first || y >= dim.second)
      return; 

   //cout << "PROP: "<<x<<" "<<y<<endl;

   if (x > 0 && y > 0) {
	   for (int vv = 0; vv < D; vv++){
		   if (mask.at(x-1).at(y-1).at(vv))
			 to2.at(x-1).at(y-1).at(vv) = LogAdd(to2.at(x-1).at(y-1).at(vv),
						 from.at(x).at(y).at(v)+diagM*rewards.at(x,y)+rewards.reward(v));
           else
			 to1.at(x-1).at(y-1).at(vv) = LogAdd(to1.at(x-1).at(y-1).at(vv),
						 from.at(x).at(y).at(v)+diagM*rewards.at(x,y)+rewards.reward(v));
	   }
   }

   if (x > 0) {
	   for (int vv = 0; vv < D; vv++){
		   if (mask.at(x-1).at(y).at(vv))
			 to2.at(x-1).at(y).at(vv) = LogAdd(to2.at(x-1).at(y).at(vv),
						 from.at(x).at(y).at(v)+rewards.at(s));
           else
			 to1.at(x-1).at(y).at(vv) = LogAdd(to1.at(x-1).at(y).at(vv),
						 from.at(x).at(y).at(v)+rewards.at(s));
	   }
   }

   if (x > 0 && (y+1) < dim.second) {
	   for (int vv = 0; vv < D; vv++){
		   if (mask.at(x-1).at(y+1).at(vv))
			 to2.at(x-1).at(y+1).at(vv) = LogAdd(to2.at(x-1).at(y+1).at(vv),
						 from.at(x).at(y).at(v)+diagM*rewards.at(x,y)+rewards.reward(v));
           else
			 to1.at(x-1).at(y+1).at(vv) = LogAdd(to1.at(x-1).at(y+1).at(vv),
						 from.at(x).at(y).at(v)+diagM*rewards.at(x,y)+rewards.reward(v));
	   }
   }

   if (y > 0) {
	   for (int vv = 0; vv < D; vv++){
		   if (mask.at(x).at(y-1).at(vv))
			 to2.at(x).at(y-1).at(vv) = LogAdd(to2.at(x).at(y-1).at(vv),
						 from.at(x).at(y).at(v)+rewards.at(s));
           else
			 to1.at(x).at(y-1).at(vv) = LogAdd(to1.at(x).at(y-1).at(vv),
						 from.at(x).at(y).at(v)+rewards.at(s));
	   }
   }

   if ((y+1) < dim.second) {
	   for (int vv = 0; vv < D; vv++){
		   if (mask.at(x).at(y+1).at(vv))
			 to2.at(x).at(y+1).at(vv) = LogAdd(to2.at(x).at(y+1).at(vv),
						 from.at(x).at(y).at(v)+rewards.at(s));
           else
			 to1.at(x).at(y+1).at(vv) = LogAdd(to1.at(x).at(y+1).at(vv),
						 from.at(x).at(y).at(v)+rewards.at(s));
	   }
   }

   if ((x+1) < dim.first && y > 0) {
	   for (int vv = 0; vv < D; vv++){
		   if (mask.at(x+1).at(y-1).at(vv))
			 to2.at(x+1).at(y-1).at(vv) = LogAdd(to2.at(x+1).at(y-1).at(vv),
						 from.at(x).at(y).at(v)+diagM*rewards.at(x,y)+rewards.reward(v));
           else
			 to1.at(x+1).at(y-1).at(vv) = LogAdd(to1.at(x+1).at(y-1).at(vv),
						 from.at(x).at(y).at(v)+diagM*rewards.at(x,y)+rewards.reward(v));
	   }
   }

   if ((x+1) < dim.first) {
	   for (int vv = 0; vv < D; vv++){
		   if (mask.at(x+1).at(y).at(vv))
			 to2.at(x+1).at(y).at(vv) = LogAdd(to2.at(x+1).at(y).at(vv),
						 from.at(x).at(y).at(v)+rewards.at(s));
           else
			 to1.at(x+1).at(y).at(vv) = LogAdd(to1.at(x+1).at(y).at(vv),
						 from.at(x).at(y).at(v)+rewards.at(s));
	   }
   }

   if ((x+1) < dim.first && (y+1) < dim.second) {
	   for (int vv = 0; vv < D; vv++){
		   if (mask.at(x+1).at(y+1).at(vv))
			 to2.at(x+1).at(y+1).at(vv) = LogAdd(to2.at(x+1).at(y+1).at(vv),
						 from.at(x).at(y).at(v)+diagM*rewards.at(x,y)+rewards.reward(v));
           else
			 to1.at(x+1).at(y+1).at(vv) = LogAdd(to1.at(x+1).at(y+1).at(vv),
						 from.at(x).at(y).at(v)+diagM*rewards.at(x,y)+rewards.reward(v));
	   }
   }
}

void DisSeqOrderInferEngine::selectBackProp(State& s,
     vector<vector<vector<double> > > &from, vector<vector<vector<double> > > &to1, 
     vector<vector<vector<double> > > &to2, vector<vector<vector<bool> > > &mask,
      RewardMap &rewards) {

   double diagM = 1.41;
   pair<int, int> dims = rewards.dims();
   int D = rewards.V_dim();
   int x = s.x();
   int y = s.y();
   int v = s.disV;
   
   //cout << "PROP: "<<x<<" "<<y<<endl;
   if (x < 0 || y < 0 || x >= dims.first || y >= dims.second)
      return; 


   if (x > 0 && y > 0) {
	   for (int vv = 0; vv < D; vv++){
		   if (mask.at(x-1).at(y-1).at(vv))
			 to2.at(x-1).at(y-1).at(vv) = LogAdd(to2.at(x-1).at(y-1).at(vv),
					from.at(x).at(y).at(v)+diagM*rewards.at(x-1,y-1)+rewards.reward(vv));
           else
			 to1.at(x-1).at(y-1).at(vv) = LogAdd(to1.at(x-1).at(y-1).at(vv),
				    from.at(x).at(y).at(v)+diagM*rewards.at(x-1,y-1)+rewards.reward(vv));
	   }
   }

   if (x > 0) {
	   for (int vv = 0; vv < D; vv++){
		   if (mask.at(x-1).at(y).at(vv))
			 to2.at(x-1).at(y).at(vv) = LogAdd(to2.at(x-1).at(y).at(vv),
						from.at(x).at(y).at(v)+rewards.at(x-1,y)+rewards.reward(vv));
           else
			 to1.at(x-1).at(y).at(vv) = LogAdd(to1.at(x-1).at(y).at(vv),
						 from.at(x).at(y).at(v)+rewards.at(x-1,y)+rewards.reward(vv));
	   }
   }

   if (x > 0 && (y+1) < dims.second) {
	   for (int vv = 0; vv < D; vv++){
		   if (mask.at(x-1).at(y+1).at(vv))
			 to2.at(x-1).at(y+1).at(vv) = LogAdd(to2.at(x-1).at(y+1).at(vv),
				  from.at(x).at(y).at(v)+diagM*rewards.at(x-1,y+1)+rewards.reward(vv));
           else
			 to1.at(x-1).at(y+1).at(vv) = LogAdd(to1.at(x-1).at(y+1).at(vv),
				  from.at(x).at(y).at(v)+diagM*rewards.at(x-1,y+1)+rewards.reward(vv));
	   }
   }

   if (y > 0) {
	   for (int vv = 0; vv < D; vv++){
		   if (mask.at(x).at(y-1).at(vv))
			 to2.at(x).at(y-1).at(vv) = LogAdd(to2.at(x).at(y-1).at(vv),
						 from.at(x).at(y).at(v)+rewards.at(x,y-1)+rewards.reward(vv));
           else
			 to1.at(x).at(y-1).at(vv) = LogAdd(to1.at(x).at(y-1).at(vv),
						 from.at(x).at(y).at(v)+rewards.at(x,y-1)+rewards.reward(vv));
	   }
   }

   if ((y+1) < dims.second) {
	   for (int vv = 0; vv < D; vv++){
		   if (mask.at(x).at(y+1).at(vv))
			 to2.at(x).at(y+1).at(vv) = LogAdd(to2.at(x).at(y+1).at(vv),
						 from.at(x).at(y).at(v)+rewards.at(x,y+1)+rewards.reward(vv));
           else
			 to1.at(x).at(y+1).at(vv) = LogAdd(to1.at(x).at(y+1).at(vv),
						 from.at(x).at(y).at(v)+rewards.at(x,y+1)+rewards.reward(vv));
	   }
   }

   if ((x+1) < dims.first && y > 0) {
	   for (int vv = 0; vv < D; vv++){
		   if (mask.at(x+1).at(y-1).at(vv))
			 to2.at(x+1).at(y-1).at(vv) = LogAdd(to2.at(x+1).at(y-1).at(vv),
				    from.at(x).at(y).at(v)+diagM*rewards.at(x+1,y-1)+rewards.reward(vv));
           else
			 to1.at(x+1).at(y-1).at(vv) = LogAdd(to1.at(x+1).at(y-1).at(vv),
					from.at(x).at(y).at(v)+diagM*rewards.at(x+1,y-1)+rewards.reward(vv));
	   }
   }

   if ((x+1) < dims.first) {
	   for (int vv = 0; vv < D; vv++){
		   if (mask.at(x+1).at(y).at(vv))
			 to2.at(x+1).at(y).at(vv) = LogAdd(to2.at(x+1).at(y).at(vv),
					from.at(x).at(y).at(v)+rewards.at(x+1,y)+rewards.reward(vv));
           else
			 to1.at(x+1).at(y).at(vv) = LogAdd(to1.at(x+1).at(y).at(vv),
					from.at(x).at(y).at(v)+rewards.at(x+1,y)+rewards.reward(vv));
	   }
   }

   if ((x+1) < dims.first && (y+1) < dims.second) {
	   for (int vv = 0; vv < D; vv++){
		   if (mask.at(x+1).at(y+1).at(vv))
			 to2.at(x+1).at(y+1).at(vv) = LogAdd(to2.at(x+1).at(y+1).at(vv),
					from.at(x).at(y).at(v)+diagM*rewards.at(x+1,y+1)+rewards.reward(vv));
           else
			 to1.at(x+1).at(y+1).at(vv) = LogAdd(to1.at(x+1).at(y+1).at(vv),
					from.at(x).at(y).at(v)+diagM*rewards.at(x+1,y+1)+rewards.reward(vv));
	   }
   }

}


void DisSeqOrderInferEngine::flattenPartitions(
      vector<vector<vector<vector<double> > > > &origPartition,
      vector<vector<vector<double> > > &collapsePartition) {

   collapsePartition.clear();

   int T = origPartition.size();
   int x_d = origPartition.at(0).size();
   int y_d = origPartition.at(0).at(0).size();
   int v_d = origPartition.at(0).at(0).at(0).size();

   collapsePartition.resize(x_d, 
        vector<vector<double> > (y_d,vector<double>(v_d,-HUGE_VAL)));

   for (int t=0; t < T; t++) {
      for (int i=0; i < x_d; i++) {
         for (int j=0; j < y_d; j++) {
			 for (int k=0; k < v_d; k++){
				//cout << i <<"  "<<j<<" !"<<endl;
				//cout << collapsepartition.size() << "  "
				//   << collapsepartition.at(0).size()<<endl;
				collapsePartition.at(i).at(j).at(k) = LogAdd(
                   collapsePartition.at(i).at(j).at(k), 
                   origPartition.at(t).at(i).at(j).at(k));
			 }
         }
      }
   }
}

void DisSeqOrderInferEngine::cummulativeFlattenPartitions(
     vector<vector<vector<vector<double> > > > &origPartition,
     vector<vector<vector<vector<double> > > > &cummFlattenPartition) {

   //cummFlattenPartition.clear();
   int T = origPartition.size();
   int x_d = origPartition.at(0).size();
   int y_d = origPartition.at(0).at(0).size();
   int v_d = origPartition.at(0).at(0).at(0).size();
   
   //cummFlattenPartition.resize(T, vector<vector<vector<double> > > 
//			   (x_d,vector<vector<double> > (y_d,vector<double>(v_d,-HUGE_VAL))));


   for (int t=0; t < origPartition.size(); t++) {
      for (int i=0; i < x_d; i++) {
         for (int j=0; j < y_d; j++) { 
			 for (int k=0; k < v_d; k++) {
                if (t > 0)
                    cummFlattenPartition.at(t).at(i).at(j).at(k) = 
                         cummFlattenPartition.at(t-1).at(i).at(j).at(k);
                cummFlattenPartition.at(t).at(i).at(j).at(k) = LogAdd(
                   cummFlattenPartition.at(t).at(i).at(j).at(k),
                   origPartition.at(t).at(i).at(j).at(k));
			 }
         }
      }
   }
}



void DisSeqOrderInferEngine::forward(State &state,RewardMap &rewards,
			vector<vector<vector<vector<double> > > > &partition) {
   partition.clear();

   pair<int, int> dims = rewards.dims();
   int v_dim = rewards.V_dim();
   try{
       partition.resize(itrTimes, vector<vector<vector<double> > >(dims.first, 
           vector<vector<double> >(dims.second, vector<double>(v_dim,-HUGE_VAL))));
   }catch(bad_alloc &ba){
	   cerr<<"forward: "<<itrTimes<<" "<<dims.first<<" "<<dims.second<<
		   " "<<v_dim<<" "<<ba.what()<<endl;
   }


   assert(state.pos.first >= 0 && state.pos.first < dims.first);
   assert(state.pos.second >= 0 && state.pos.second < dims.second);
   assert(state.disV >= 0 && state.disV < v_dim);

   partition.at(0).at(state.pos.first).at(state.pos.second).at(state.disV) = 0.0;

   vector<vector<vector<double> > > costs;
   vector<State> order;
   double maximum;
   rewards.dijkstras(state, costs, order, maximum);
   int update_size = floor(order.size()/2);
//   cout << "Forward Iteration size: "<<order.size()<<endl;

   for (int t=0; t < (itrTimes-1); t++) {

      //cout << "Forward Iteration: "<<t<<endl;
      vector<vector<vector<bool> > > mask(dims.first, 
				  vector<vector<bool> > (dims.second, vector<bool>(v_dim,false)));
      
      for (int i=0; i < order.size(); i++) { 
		    mask.at(order.at(i).x()).at(order.at(i).y()).at(order.at(i).disV) 
				= true;
            selectProp(order.at(i), partition.at(t), partition.at(t), 
               partition.at(t+1), mask, rewards); 
         //cout << "FORWARD: "<<x<<" "<<y<<" "<<partition.at(t).at(x).at(y)<<endl;
      } 
   }
   
   order.clear();
   costs.clear();
   //cout << "foward inference done"<<endl;
   return;
}

void DisSeqOrderInferEngine::backward(State &state,
      RewardMap &rewards, vector<vector<vector<double> > > &initial, 
      vector<vector<vector<vector<double> > > > &partition) {

   partition.clear();

   pair<int, int> dims = rewards.dims();
   int v_dim = rewards.V_dim();
   try{
	   partition.resize(itrTimes, vector<vector<vector<double> > >(dims.first, 
           vector<vector<double> >(dims.second, vector<double>(v_dim,-HUGE_VAL))));
	   partition.at(0)=initial;

   }catch(bad_alloc &ba){
	   cerr<<"backward: "<<itrTimes<<" "<<dims.first<<" "<<dims.second<<
		   " "<<v_dim<<" "<<ba.what()<<endl;
   }

   vector<vector<vector<double> > > costs;
   vector<State> order;
   double maximum;
   rewards.dijkstras(state, costs, order, maximum); 
   int update_size = floor(order.size()/2);
//   cout << "Backward Iteration size: "<<order.size()<<endl;
   
   for (int t=0; t < (itrTimes-1); t++) {

      vector<vector<vector<bool> > > mask(dims.first, 
				  vector<vector<bool> > (dims.second, vector<bool>(v_dim,false)));
      
      //cout << "Backward Iteration: "<<t<<endl;

      for (int i=order.size()-1; i >= 0; i--) {
		    mask.at(order.at(i).x()).at(order.at(i).y()).at(order.at(i).disV)
				= true;
            selectBackProp(order.at(i), partition.at(t), partition.at(t), 
               partition.at(t+1), mask, rewards); 
		 }
      mask.clear();
   } 

   //cout << "done"<<endl;
   costs.clear();
   order.clear();
   return;
}


void DisSeqOrderInferEngine::combinePartitions(
      vector<vector<vector<vector<double> > > > & partition1, 
      vector<vector<vector<vector<double> > > > & partition2,
      vector<vector<vector<double> > > & partitionOut) {

   //engine.combinePartitions(forwardPartitionB, cummBackPartition, occupancy);

   int x_d = partition1.at(0).size();
   int y_d = partition1.at(0).at(0).size();
   int v_d = partition1.at(0).at(0).at(0).size();

   for (int t=0; t < min(partition1.size(), partition2.size()); t++) {

      for (int i=0; i < x_d; i++) {
         assert(partition1.at(t).at(i).size() == partition2.at(t).at(i).size());
         assert(partitionOut.at(i).size() == partition1.at(t).at(i).size());

         for (int j=0; j < y_d; j++) {
			 for (int k=0; k < v_d; k++){
                  partitionOut.at(i).at(j).at(k) = LogAdd(
                  partitionOut.at(i).at(j).at(k), 
				  partition1.at(t).at(i).at(j).at(k) +
                  partition2.at(t).at(i).at(j).at(k));
			 } 
         }
      }
   }
}



DisSeqPredictor::DisSeqPredictor(Grid &_grid, RewardMap &_rewards, 
      DisSeqOrderInferEngine &_engine) 
      : grid(_grid), rewards(_rewards), engine(_engine) {

   dimensions.push_back(grid.dims().first);
   dimensions.push_back(grid.dims().second);
   dimensions.push_back(rewards.V_dim());
   cout<<"Dimensions: "<<dimensions[0]<<" "<<dimensions[1]<<" "
	   <<dimensions[2]<<endl;

   double weight = 1.0/(dimensions[0]*dimensions[1]*dimensions[2]);
   try{
   prior.resize(dimensions[0], vector<vector<double> >(dimensions[1],
				   vector<double>(dimensions[2], weight)));
   }catch(bad_alloc &ba){
	   cerr<<"prior: "<<dimensions[0]<<" "<<dimensions[1]<<" "<<dimensions[2]<<
		   " "<<ba.what()<<endl;
   }
}

void DisSeqPredictor::setStart(State & start) { 
  // start = _start;

   cout << "Start Point Inference"<<endl;

   cout << "   Forward Inference"<<endl;
   engine.forward(start, rewards, forwardPartitionA); 
   // 10 is propogation times
   cout << "   Flattening partitions"<<endl;
   engine.flattenPartitions(forwardPartitionA, forwardFlatPartitionA);
   forwardPartitionA.clear();

}

void DisSeqPredictor::setPrior(vector<vector<vector<double> > > &_prior) {
   prior = _prior;
}

vector<vector<double> > DisSeqPredictor::getFlatPosterior(){
	vector<vector<double> > flatposterior;
	flatposterior.resize(dimensions.at(0),
				vector<double> (dimensions.at(1),-HUGE_VAL));
	for(int i=0;i<dimensions.at(0);i++){
		for(int j=0;j<dimensions.at(1);j++){
			for(int k=0;k<dimensions.at(2);k++){
				flatposterior.at(i).at(j) = LogAdd(flatposterior.at(i).at(j),
							posterior.at(i).at(j).at(k));
			}
		}
	}
    return flatposterior;
}

void DisSeqPredictor::flattenOccupancy(vector<vector<vector<double> > > &origOccupancy,
			vector<vector<double> > &flatOccupancy){
	flatOccupancy.resize(dimensions.at(0),
				vector<double> (dimensions.at(1),-HUGE_VAL));
	for(int i=0;i<dimensions.at(0);i++){
		for(int j=0;j<dimensions.at(1);j++){
			for(int k=0;k<dimensions.at(2);k++){
				flatOccupancy.at(i).at(j) = LogAdd(flatOccupancy.at(i).at(j),
							origOccupancy.at(i).at(j).at(k));
			}
		}
	}
}


double DisSeqPredictor::forwardBackwardInference(State &current, 
     vector<vector<vector<double> > > &occupancy) {

   cout << "Future Prediction"<<endl;

   occupancy.clear();
   try{
        occupancy.resize(dimensions[0], 
          vector<vector<double> >(dimensions[1],vector<double> (dimensions[2], -HUGE_VAL))); 
   }catch(bad_alloc &ba){
	    cerr<<"occupancy: "<<dimensions[0]<<" "<<dimensions[1]<<" "<<dimensions[2]<<
		   " "<<ba.what()<<endl;
   }

   cout << "   Forward Inference"<<endl;

   engine.forward(current, rewards, forwardPartitionB); 

   cout << "   Flattening partitions"<<endl;
   
   engine.flattenPartitions(forwardPartitionB, forwardFlatPartitionB);

   double sum = -HUGE_VAL;

   double normalizer = -HUGE_VAL;

   posterior.clear();
   try{
         posterior.resize(dimensions[0], 
            vector<vector<double> >(dimensions[1], vector<double>(
				dimensions[2],-HUGE_VAL)));
   }catch(bad_alloc &ba){
	     cerr<<"posterior: "<<dimensions[0]<<" "<<dimensions[1]<<" "<<dimensions[2]<<
		   " "<<ba.what()<<endl;
   }
   
   
   for (int i=0; i < dimensions[0]; i++) { 
      for (int j=0; j < dimensions[1]; j++) {
		  for(int k=0; k < dimensions[2]; k++){
			  //P(G)exp(V(st->G)-V(s0->G))
			  
			posterior.at(i).at(j).at(k) = prior.at(i).at(j).at(k) 
               + forwardFlatPartitionB.at(i).at(j).at(k)
			   - forwardFlatPartitionA.at(i).at(j).at(k);
            if (isnan(posterior.at(i).at(j).at(k))) continue;
            if (posterior.at(i).at(j).at(k) == -HUGE_VAL) continue;
            //cout << "POSTERIOR: "<<i<<"  "<<j<<"  "<<posterior.at(i).at(j)<<endl;
            sum = LogAdd(sum, posterior.at(i).at(j).at(k));
            if (prior.at(i).at(j).at(k) == 0.0) {
               //cout << "NORMALIZER: "<<forwardFlatPartitionA.at(i).at(j)<<endl;
			   //i,j,k is the goal state, so
			   //forwardFlatPartitionA.at(i).at(j).at(k) is V(s_1->G)
               normalizer = forwardFlatPartitionA.at(i).at(j).at(k);
            }
		  }
      }
   }
   cout <<"   Posterior Normalizer (" <<sum<<")"<<endl;

   initBackWeight.clear();
   try{
        initBackWeight.resize(dimensions[0],
           vector<vector<double> > (dimensions[1], vector<double>(
				dimensions[2],-HUGE_VAL)));
   }catch(bad_alloc &ba){
	    cerr<<"initBackWeight: "<<dimensions[0]<<" "<<dimensions[1]<<" "<<dimensions[2]<<
		   " "<<ba.what()<<endl;
   }

   for (int i=0; i < dimensions[0]; i++) {
      for (int j=0; j < dimensions[1]; j++) {
		  for (int k=0; k < dimensions[2]; k++){
              posterior.at(i).at(j).at(k) -= sum;
              //cout << "POST: "<<i << "  "<<j<<"  Ba"<<posterior.at(i).at(j)
              //   << "  "<<forwardFlatPartitionB.at(i).at(j)  
              //   << "  "<<forwardFlatPartitionA.at(i).at(j)<<endl;
              initBackWeight.at(i).at(j).at(k) = -forwardFlatPartitionB.at(i).at(j).at(k) +
              posterior.at(i).at(j).at(k);// - rewards.at(i,j);
              //if (initBackWeight.at(i).at(j) > -HUGE_VAL)
              // cout << "Init Back: "<<i<<" "<<j<<"  "
              //   <<initBackWBaeight.at(i).at(j)<<endl;
		  }
      }
   }
  
   
   cout << "BACKWARD"<<endl;
   // calculate backwards probability including prior
   engine.backward(current, rewards, initBackWeight, backwardPartition);

   cout << "CUMM"<<endl;
   cummBackPartition.clear();
   try{
         cummBackPartition.resize(10,vector<vector<vector<double> > >(
			   dimensions[0],vector<vector<double> > 
			   (dimensions[1], vector<double>(dimensions[2],-HUGE_VAL))));
   }catch(bad_alloc &ba){
	     cerr<<"cummBackParttion: "<<dimensions[0]<<" "<<dimensions[1]<<" "<<dimensions[2]<<
		   " "<<ba.what()<<endl;
   }

   engine.cummulativeFlattenPartitions(backwardPartition,cummBackPartition);
   backwardPartition.clear();
   cout << "Combine: "<<endl;

   occupancy.clear();
   try{
        occupancy.resize(dimensions[0], 
         vector<vector<double> >(dimensions[1],vector<double> (dimensions[2], -HUGE_VAL))); 
   }catch(bad_alloc &ba){
	    cerr<<"occupancy: "<<dimensions[0]<<" "<<dimensions[1]<<" "<<dimensions[2]<<
		   " "<<ba.what()<<endl;
   }

   engine.combinePartitions(forwardPartitionB, cummBackPartition, occupancy);

   //double norm = backwardPartition.at(0).at(current.x()).at(current.y()).at(current.disV);
    //  + rewards.at(current.first, current.second);

   //cout << "COMBINING: "<<norm<<endl; 

   //.cout << "NORM2: "<<
    //  cummBackPartition.back().at(current.x()).at(current.y()).at(current.disV)<<endl;

   //cout << backwardPartition.at(0).at(current.x()).at(current.y()).at(current.disV) << "  "
      //<< forwardPartitionB.at(0).at(current.x()).at(current.y()).at(current.disV)<<endl;
	  

   return normalizer;
}

void DisSeqPredictor::forwardInference(State &current) {

   cout << "   Forward Inference"<<endl;

   engine.forward(current, rewards, forwardPartitionB); 

   cout << "   Flattening partitions"<<endl;
   
   engine.flattenPartitions(forwardPartitionB, forwardFlatPartitionB);

   double sum = -HUGE_VAL;

   double normalizer = -HUGE_VAL;

   obdistribute.clear();
   try{
	     obdistribute.resize(dimensions[0], 
	         vector<vector<double> >(dimensions[1],vector<double> (dimensions[2], -HUGE_VAL))); 
	}catch(bad_alloc &ba){
		 cerr<<"obdistribute: "<<dimensions[0]<<" "<<dimensions[1]<<" "<<dimensions[2]<<
		   " "<<ba.what()<<endl;
	}
	
	cout<<"  Compute observation distribution"<<endl;
	for (int i=0; i < dimensions[0]; i++) { 
		  for (int j=0; j < dimensions[1]; j++) {
			  for(int k=0; k < dimensions[2]; k++){
				  //exp(V(st->G)-V(s0->G)
				obdistribute.at(i).at(j).at(k) = forwardFlatPartitionB.at(i).at(j).at(k)
				- forwardFlatPartitionA.at(i).at(j).at(k);
			  }
		  }
	 }

}
