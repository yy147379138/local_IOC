#include "inference.h"
#include <queue>
#include <cassert>

class Comparator{
	public:
		bool operator()(const pair<double,State>& a, 
					const pair<double,State> & b) {
			return a.first>b.first;
		}
	private:
};

Parameters::Parameters(vector<double> &_weights) : weights(_weights) {

}


double Parameters::reward(vector<double> &features) {
   double val = 0.0;
   for (int i=0; i < features.size(); i++) 
      val += features.at(i) * weights.at(i); 
   return val;
}

double Parameters::reward(vector<int> &features) {
   double val = 0.0;
   for (int i=0; i < features.size(); i++) 
      val += features.at(i) * weights.at(i); 
   return val;
}

RewardMap::RewardMap(FeatureArray &posfeatures, Parameters &params) {
   pair<int, int> dims = posfeatures.dims(); 
  
   posvalues.resize(dims.first, vector<double>(dims.second, 0.0));

   for (int i=0; i < dims.first; i++) 
      for (int j=0; j < dims.second; j++) 
         posvalues.at(i).at(j) = params.reward(posfeatures.at(i,j)); 
}


RewardMap::RewardMap(FeatureArray &posfeatures, SeqFeature &seqfeat,
			Parameters &params) {
   pair<int, int> dims = posfeatures.dims(); 
  
   posvalues.resize(dims.first, vector<double>(dims.second, 0.0));

   for (int i=0; i < dims.first; i++) 
      for (int j=0; j < dims.second; j++) 
         posvalues.at(i).at(j) = params.reward(posfeatures.at(i,j)); 


   int D  = seqfeat.num_V();
   for (int j=0; j < D; j++){
	 lookupTable.push_back(params.back()*j);
   }
}


RewardMap::RewardMap(FeatureArray &posfeatures, SeqFeature &seqfeat,
			Parameters &p_params, Parameters &s_params) {
   pair<int, int> dims = posfeatures.dims(); 
  
   posvalues.resize(dims.first, vector<double>(dims.second, 0.0));

   //cout<<"pos "<<posfeatures.size()<<endl;
   for (int i=0; i < dims.first; i++) 
      for (int j=0; j < dims.second; j++) 
         posvalues.at(i).at(j) = p_params.reward(posfeatures.at(i,j)); 
   //cout<<"seq "<<s_params.size()<<" "<<seqfeat.num_V()<<endl;

   int D  = seqfeat.num_V();
   for (int j=0; j < D; j++)
	 lookupTable.push_back(s_params.at(j));
}

void RewardMap::dijkstras(pair<int, int> origin, 
      vector<vector<double> > &costs,
      vector<pair<int, int> > &order,
      double &maximum) {


   pair<int, int> dims;
   dims.first = posvalues.size();
   dims.second = posvalues.at(0).size();

   priority_queue<pair<double, pair<int, int> >,
       vector<pair<double, pair<int, int> > >,
       greater<pair<double, pair<int, int> > > > pq;
   //prioiry_queue<class, Container, Compare>
   
   vector<vector<int> > explored(dims.first, 
         vector<int>(dims.second, 0));

   costs.resize(dims.first, 
         vector<double>(dims.second, HUGE_VAL));

   vector<vector<double> > activeCosts(dims.first,
         vector<double>(dims.second, HUGE_VAL));


   order.clear();

   pair<double, pair<int, int> > ele;
   pair<int, int> pt;
   pq.push(pair<double, pair<int, int> >(0.0, origin));

   while (pq.size() > 0) {
      ele = pq.top();
      pq.pop(); 

      int x = ele.second.first;
      int y = ele.second.second;

     // cout << "PT: "<<x<<"  "<<y<<"   "<<ele.first<<"  "
       //  <<explored.at(x).at(y)<<endl;

      if (explored.at(x).at(y)) continue; 

      order.push_back(ele.second);// push position
      
      explored.at(x).at(y) = 1; 
      costs.at(x).at(y) = ele.first;

      maximum = ele.first;

	  //When finding optimal paths, only position rewards are
	  //considered, ignoring the 
      double cost = -posvalues.at(x).at(y);

      double diagVal = ele.first+1.5*cost;
      double strVal = ele.first+cost;

      if (x > 0 && y > 0 && diagVal < activeCosts.at(x-1).at(y-1)) {
         pq.push(pair<double, pair<int, int> >(diagVal, 
                  pair<int,int>(x-1,y-1)));
         activeCosts.at(x-1).at(y-1) = diagVal;
      }
 
      if (x > 0 && strVal < activeCosts.at(x-1).at(y)) {
         pq.push(pair<double, pair<int, int> >(strVal, 
                  pair<int,int>(x-1,y))); 
         activeCosts.at(x-1).at(y) = strVal;
      }
      
      if (x > 0 && y < dims.second-1 && diagVal < activeCosts.at(x-1).at(y+1)){ 
         pq.push(pair<double, pair<int, int> >(diagVal,
                  pair<int,int>(x-1,y+1)));
         activeCosts.at(x-1).at(y+1) = diagVal;
      }
 
      if (y > 0 && strVal < activeCosts.at(x).at(y-1)) {
         pq.push(pair<double, pair<int, int> >(strVal, 
                  pair<int,int>(x,y-1)));
        
      }

      if (y < dims.second-1 && strVal < activeCosts.at(x).at(y+1)) {
         pq.push(pair<double, pair<int, int> >(strVal, 
                  pair<int,int>(x,y+1)));
         activeCosts.at(x).at(y+1) = strVal;
      }

      if (x < dims.first-1 && y > 0 && diagVal < activeCosts.at(x+1).at(y-1)) {
         pq.push(pair<double, pair<int, int> >(diagVal, 
                  pair<int, int>(x+1,y-1)));
         activeCosts.at(x+1).at(y-1) = diagVal;
      }

      if (x < dims.first-1 && strVal < activeCosts.at(x+1).at(y)) {
         pq.push(pair<double, pair<int, int> >(strVal, 
                  pair<int, int>(x+1,y)));
         activeCosts.at(x+1).at(y) = strVal;
      }

      if (x < dims.first-1 && y < dims.second-1 && 
            diagVal < activeCosts.at(x+1).at(y+1)) {
         pq.push(pair<double, pair<int, int> >(diagVal, 
                  pair<int, int>(x+1,y+1)));
         activeCosts.at(x+1).at(y+1) = diagVal;
      }
   }


}



void RewardMap::dijkstras(State& origin, 
      vector<vector<vector<double> > > &costs,
      vector<State> &order,double &maximum) {

   pair<int, int> dims;
   dims.first = posvalues.size();
   dims.second = posvalues.at(0).size();
   //pair<int, int> dims = dims();
   int v_dim = V_dim();
   /*
   priority_queue<pair<double, State>,
       vector<pair<double, State> >,
       greater<pair<double, State> > > pq;
   */
   priority_queue<pair<double, State>,
       vector<pair<double, State> >,Comparator > pq;

    //prioiry_queue<class, Container, Compare>
   
   vector<vector<vector<int> > > explored(dims.first, 
        vector<vector<int> >(dims.second,vector<int>(v_dim, 0)));

   costs.resize(dims.first, 
         vector<vector<double> >(dims.second, vector<double> (v_dim,HUGE_VAL)));
   
   vector<vector<vector<double> > > activeCosts(dims.first,
         vector<vector<double> >(dims.second, vector<double> (v_dim,HUGE_VAL)));


   order.clear();

   //pair<double, State> ele;
   pq.push(pair<double, State>(0.0, origin));

   while (pq.size() > 0) {
      //ele = pq.top();
      State s(pq.top().second);
	  double pq_cost = pq.top().first;
	 // State s(pq.top().second.x(),pq.top().second.y(),pq.top().second.disV);
	  int x = s.x();
      int y = s.y();
      int v = s.disV;

      pq.pop(); 
      //cout << "PT: "<<x<<"  "<<y<<"   "<<v<<" explored: "
        // <<explored.at(x).at(y).at(v)<<endl;

      if (explored.at(x).at(y).at(v)) continue; 

      order.push_back(s);// push state
      
      explored.at(x).at(y).at(v) = 1; 
      costs.at(x).at(y).at(v) = pq_cost; // ele.first;

      maximum =pq_cost; // ele.first;
      //cout<<"Dijkstra cost: "<<maximum<<endl;
	  //When finding optimal paths, only position rewards are
	  //considered, ignoring the 
      double movcost = -posvalues.at(x).at(y);
      double diagVal = pq_cost+1.5*movcost-lookupTable.at(v);
      double strVal = pq_cost+movcost-lookupTable.at(v);


	  for (int future_v = 0;future_v<v_dim;future_v++){

		if (x > 0 && y > 0 && diagVal < activeCosts.at(x-1).at(y-1).at(future_v)) {
			 State s(x-1,y-1,future_v);
			 pq.push(pair<double, State>(diagVal,s));
			activeCosts.at(x-1).at(y-1).at(future_v) = diagVal;

		}
 
		if (x > 0 && strVal < activeCosts.at(x-1).at(y).at(future_v)) {
			 State s(x-1,y,future_v);
			 pq.push(pair<double, State>(strVal,s)); 
			activeCosts.at(x-1).at(y).at(future_v) = strVal;
		}
      

		if (x > 0 && y < dims.second-1 && diagVal < 
					activeCosts.at(x-1).at(y+1).at(future_v)){
			 State s(x-1,y+1,future_v);
			 pq.push(pair<double, State>(strVal,s)); 
			activeCosts.at(x-1).at(y+1).at(future_v) = diagVal;
		}

 
		if (y > 0 && strVal < activeCosts.at(x).at(y-1).at(future_v)) {
			 State s(x,y-1,future_v);
			 pq.push(pair<double, State>(strVal,s)); 
		     activeCosts.at(x).at(y-1).at(future_v) = strVal;
		}

		if (y < dims.second-1 && strVal 
				  < activeCosts.at(x).at(y+1).at(future_v)) {
		      State s(x,y+1,future_v);
			  pq.push(pair<double, State>(strVal,s));
			  activeCosts.at(x).at(y+1).at(future_v) = strVal;
		}

		if (x < dims.first-1 && y > 0 && diagVal < 
					  activeCosts.at(x+1).at(y-1).at(future_v)) {
			   State s(x+1,y-1,future_v);
			   pq.push(pair<double, State>(strVal,s)); 
			   activeCosts.at(x+1).at(y-1).at(future_v) = diagVal;
		}

		if (x < dims.first-1 && strVal < 
					  activeCosts.at(x+1).at(y).at(future_v)) {
			   State s(x+1,y,future_v);
			   pq.push(pair<double, State>(strVal,s));
			   activeCosts.at(x+1).at(y).at(future_v) = strVal;
		}

		if (x < dims.first-1 && y < dims.second-1 && 
            diagVal < activeCosts.at(x+1).at(y+1).at(future_v)) {
				State s(x+1,y+1,future_v);
			    pq.push(pair<double, State>(strVal,s));
			    activeCosts.at(x+1).at(y+1).at(future_v) = diagVal;
		}
	  }
   }

   
}


void InferenceEngine::forward(pair<int, int> pos, int T, RewardMap &rewards, 
      vector<vector<vector<double> > > &partition) {//called in setstart forward(start, 10,reward, forwardpartitionA)
   partition.clear();

   pair<int, int> dims = rewards.dims();

//   cout << "Dimensions: "<<dims.first<<" "<<dims.second<<endl;

   partition.resize(T, vector<vector<double> >(dims.first, 
            vector<double>(dims.second, 0.0)));

   cout << "Obtaining rewards"<<endl;

   assert(pos.first >= 0 && pos.first < dims.first);

   assert(pos.second >= 0 && pos.second < dims.second);

   partition.at(0).at(pos.first).at(pos.second) = 0.0;

   double count = 0;
   cout << "Integrating"<<endl;
   for (int t=0; t < (T-1); t++) { 
      cout << "ITER: "<<t<<endl;

      /* (i-1), (j-1) */ 
      for (int i=1; i < dims.first; i++) { 
         for (int j=1; j < dims.second; j++) {
            partition.at(t+1).at(i-1).at(j-1)
               = LogAdd(partition.at(t+1).at(i-1).at(j-1), 
                     partition.at(t).at(i).at(j)+rewards.at(i, j));
         }
      }

      cout << "2"<<endl;
 
      /* (i), (j-1) */ 
      for (int i=0; i < dims.first; i++) { 
         for (int j=1; j < dims.second; j++) {
            partition.at(t+1).at(i).at(j-1)
               = LogAdd(partition.at(t+1).at(i).at(j-1), 
                     partition.at(t).at(i).at(j)+rewards.at(i, j));
 
         }
      }
      cout << "3"<<endl;
 
      /* (i+1), (j-1) */ 
      for (int i=0; i < (dims.first-1); i++) { 
         for (int j=1; j < dims.second; j++) {
            partition.at(t+1).at(i+1).at(j-1)
               = LogAdd(partition.at(t+1).at(i+1).at(j-1), 
                     partition.at(t).at(i).at(j)+rewards.at(i, j));
 
         }
      }
      cout << "4"<<endl;
 
      /* (i-1), (j) */ 
      for (int i=1; i < dims.first; i++) { 
         for (int j=0; j < dims.second; j++) {
            partition.at(t+1).at(i-1).at(j)
               = LogAdd(partition.at(t+1).at(i-1).at(j), 
                     partition.at(t).at(i).at(j)+rewards.at(i, j));
 
         }
      }

      cout << "5"<<endl;
 
      /* (i+1), (j) */ 
      for (int i=0; i < (dims.first-1); i++) { 
         for (int j=0; j < dims.second; j++) {
            partition.at(t+1).at(i+1).at(j)
               = LogAdd(partition.at(t+1).at(i+1).at(j), 
                     partition.at(t).at(i).at(j)+rewards.at(i, j));
 
         }
      }

      cout << "6"<<endl;
 
      /* (i-1), (j+1) */ 
      for (int i=1; i < dims.first; i++) { 
         for (int j=0; j < (dims.second-1); j++) {
            partition.at(t+1).at(i-1).at(j+1)
               = LogAdd(partition.at(t+1).at(i-1).at(j+1), 
                     partition.at(t).at(i).at(j)+rewards.at(i, j));
 
         }
      }
 
      cout << "7"<<endl;
 
      /* (i), (j+1) */ 
      for (int i=0; i < dims.first; i++) { 
         for (int j=0; j < (dims.second-1); j++) {
            partition.at(t+1).at(i).at(j+1)
               = LogAdd(partition.at(t+1).at(i).at(j+1), 
                     partition.at(t).at(i).at(j)+rewards.at(i, j));
 
         }
      }
 
      cout << "8"<<endl;
 
      /* (i+1), (j+1) */ 
      for (int i=0; i < (dims.first-1); i++) { 
         for (int j=0; j < (dims.second-1); j++) {
            partition.at(t+1).at(i+1).at(j+1)
               = LogAdd(partition.at(t+1).at(i+1).at(j+1), 
                     partition.at(t).at(i).at(j)+rewards.at(i, j)); 
         }
      }
      cout << "past"<<endl;
   } 
   cout << count<<endl;
}

void InferenceEngine::backward(pair<int, int> pos, int T, RewardMap &rewards,
      vector<vector<double> > &initial, 
      vector<vector<vector<double> > > &partition) {
}

void InferenceEngine::selectProp(int x, int y, pair<int, int> dim,
      vector<vector<double> > &from, vector<vector<double> > &to1, 
      vector<vector<double> > &to2, vector<vector<bool> > &mask,
      double reward) {
      // called in WaveInference::forward();
	  // selectProp(x0,i,partition.at(t),partition.at(t),partition.at(t+1),mask,rewards.at(x0).at(i))
   double diagM = 1.41;

   if (x < 0 || y < 0 || x >= dim.first || y >= dim.second)
      return; 

//   cout<<dim.first<<" "<<dim.second<<endl;

   //cout << "PROP: "<<x<<" "<<y<<endl;

   if (x > 0 && y > 0) {
      if (mask.at(x-1).at(y-1))
         to2.at(x-1).at(y-1) = LogAdd(to2.at(x-1).at(y-1), from.at(x).at(y)+
            diagM*reward);
      else
         to1.at(x-1).at(y-1) = LogAdd(to1.at(x-1).at(y-1), from.at(x).at(y)+
            diagM*reward);

   }

   if (x > 0) {
      if (mask.at(x-1).at(y)) 
         to2.at(x-1).at(y) = LogAdd(to2.at(x-1).at(y), from.at(x).at(y)+
            reward);
      else
         to1.at(x-1).at(y) = LogAdd(to1.at(x-1).at(y), from.at(x).at(y)+
            reward); 
   }

   if (x > 0 && (y+1) < dim.second) {
      if (mask.at(x-1).at(y+1))
         to2.at(x-1).at(y+1) = LogAdd(to2.at(x-1).at(y+1), 
               from.at(x).at(y)+diagM*reward);
      else
         to1.at(x-1).at(y+1) = LogAdd(to1.at(x-1).at(y+1), 
               from.at(x).at(y)+diagM*reward); 
   }

   if (y > 0) {
      if (mask.at(x).at(y-1))
         to2.at(x).at(y-1) = LogAdd(to2.at(x).at(y-1), 
               from.at(x).at(y)+reward);
      else
         to1.at(x).at(y-1) = LogAdd(to1.at(x).at(y-1), 
               from.at(x).at(y)+reward); 
   }

   if ((y+1) < dim.second) {
      if (mask.at(x).at(y+1))
         to2.at(x).at(y+1) = LogAdd(to2.at(x).at(y+1), 
               from.at(x).at(y)+reward); 
      else 
         to1.at(x).at(y+1) = LogAdd(to1.at(x).at(y+1), 
               from.at(x).at(y)+reward); 
   }
   if ((x+1) < dim.first && y > 0) {
      if (mask.at(x+1).at(y-1))
         to2.at(x+1).at(y-1) = LogAdd(to2.at(x+1).at(y-1), 
               from.at(x).at(y)+diagM*reward);
      else
         to1.at(x+1).at(y-1) = LogAdd(to1.at(x+1).at(y-1), 
               from.at(x).at(y)+diagM*reward);
   }

   if ((x+1) < dim.first) {
      if (mask.at(x+1).at(y))
         to2.at(x+1).at(y) = LogAdd(to2.at(x+1).at(y), 
               from.at(x).at(y)+reward); 
      else

         to1.at(x+1).at(y) = LogAdd(to1.at(x+1).at(y), 
               from.at(x).at(y)+reward); 
   }

   if ((x+1) < dim.first && (y+1) < dim.second) {
      if (mask.at(x+1).at(y+1))
         to2.at(x+1).at(y+1) = LogAdd(to2.at(x+1).at(y+1), from.at(x).at(y)+
               diagM*reward); 
      else
         to1.at(x+1).at(y+1) = LogAdd(to1.at(x+1).at(y+1), from.at(x).at(y)+
               diagM*reward); 
   }
}

void InferenceEngine::selectBackProp(int x, int y, pair<int, int> dim,
      vector<vector<double> > &from, vector<vector<double> > &to1, 
      vector<vector<double> > &to2, vector<vector<bool> > &mask,
      RewardMap &rewards) {

   double diagM = 1.41;

   if (x < 0 || y < 0 || x >= dim.first || y >= dim.second)
      return; 

   //cout << "PROP: "<<x<<" "<<y<<endl;

   if (x > 0 && y > 0) {
      if (mask.at(x-1).at(y-1))
         to2.at(x-1).at(y-1) = LogAdd(to2.at(x-1).at(y-1), from.at(x).at(y)+
            diagM*rewards.at(x-1,y-1));
      else
         to1.at(x-1).at(y-1) = LogAdd(to1.at(x-1).at(y-1), from.at(x).at(y)+
            diagM*rewards.at(x-1,y-1));

   }

   if (x > 0) {
      if (mask.at(x-1).at(y)) 
         to2.at(x-1).at(y) = LogAdd(to2.at(x-1).at(y), from.at(x).at(y)+
            rewards.at(x-1,y));
      else
         to1.at(x-1).at(y) = LogAdd(to1.at(x-1).at(y), from.at(x).at(y)+
            rewards.at(x-1,y)); 
   }

   if (x > 0 && (y+1) < dim.second) {
      if (mask.at(x-1).at(y+1))
         to2.at(x-1).at(y+1) = LogAdd(to2.at(x-1).at(y+1), 
               from.at(x).at(y)+diagM*rewards.at(x-1,y+1));
      else
         to1.at(x-1).at(y+1) = LogAdd(to1.at(x-1).at(y+1), 
               from.at(x).at(y)+diagM*rewards.at(x-1,y+1)); 
   }

   if (y > 0) {
      if (mask.at(x).at(y-1))
         to2.at(x).at(y-1) = LogAdd(to2.at(x).at(y-1), 
               from.at(x).at(y)+rewards.at(x,y-1));
      else
         to1.at(x).at(y-1) = LogAdd(to1.at(x).at(y-1), 
               from.at(x).at(y)+rewards.at(x,y-1)); 
   }

   if ((y+1) < dim.second) {
      if (mask.at(x).at(y+1))
         to2.at(x).at(y+1) = LogAdd(to2.at(x).at(y+1), 
               from.at(x).at(y)+rewards.at(x,y+1)); 
      else 
         to1.at(x).at(y+1) = LogAdd(to1.at(x).at(y+1), 
               from.at(x).at(y)+rewards.at(x,y+1)); 
   }
   if ((x+1) < dim.first && y > 0) {
      if (mask.at(x+1).at(y-1))
         to2.at(x+1).at(y-1) = LogAdd(to2.at(x+1).at(y-1), 
               from.at(x).at(y)+diagM*rewards.at(x+1,y-1));
      else
         to1.at(x+1).at(y-1) = LogAdd(to1.at(x+1).at(y-1), 
               from.at(x).at(y)+diagM*rewards.at(x+1,y-1));
   }

   if ((x+1) < dim.first) {
      if (mask.at(x+1).at(y))
         to2.at(x+1).at(y) = LogAdd(to2.at(x+1).at(y), 
               from.at(x).at(y)+rewards.at(x+1,y)); 
      else

         to1.at(x+1).at(y) = LogAdd(to1.at(x+1).at(y), 
               from.at(x).at(y)+rewards.at(x+1,y)); 
   }

   if ((x+1) < dim.first && (y+1) < dim.second) {
      if (mask.at(x+1).at(y+1))
         to2.at(x+1).at(y+1) = LogAdd(to2.at(x+1).at(y+1), from.at(x).at(y)+
               diagM*rewards.at(x+1,y+1)); 
      else
         to1.at(x+1).at(y+1) = LogAdd(to1.at(x+1).at(y+1), from.at(x).at(y)+
               diagM*rewards.at(x+1,y+1)); 
   }
}


void InferenceEngine::propagate(int x, int y, pair<int, int> dim,
      vector<vector<double> > &from, vector<vector<double> > &to, 
      double reward) {
	//called in WaveInference::forward
	//propagate(x,y,dims,partition.at(t), partition.at(t), rewards.at(x,y))

   double diagM = 1.0;

   if (x < 0 || y < 0 || x >= dim.first || y >= dim.second)
      return; 

   //cout << "PROP: "<<x<<" "<<y<<endl;

   if (x > 0 && y > 0) 

      to.at(x-1).at(y-1) = LogAdd(to.at(x-1).at(y-1), from.at(x).at(y)+
            diagM*reward);

   if (x > 0)
      to.at(x-1).at(y) = LogAdd(to.at(x-1).at(y), from.at(x).at(y)+
            reward);

   if (x > 0 && (y+1) < dim.second)
      to.at(x-1).at(y+1) = LogAdd(to.at(x-1).at(y+1), from.at(x).at(y)+
            diagM*reward);

   if (y > 0)
      to.at(x).at(y-1) = LogAdd(to.at(x).at(y-1), from.at(x).at(y)+reward);

   if ((y+1) < dim.second)
      to.at(x).at(y+1) = LogAdd(to.at(x).at(y+1), from.at(x).at(y)+reward);

   if ((x+1) < dim.first && y > 0)
      to.at(x+1).at(y-1) = LogAdd(to.at(x+1).at(y-1), from.at(x).at(y)+
            diagM*reward);

   if ((x+1) < dim.first)
      to.at(x+1).at(y) = LogAdd(to.at(x+1).at(y), from.at(x).at(y)+reward);

   if ((x+1) < dim.first && (y+1) < dim.second)
      to.at(x+1).at(y+1) = LogAdd(to.at(x+1).at(y+1), from.at(x).at(y)+
            diagM*reward); 
}

void InferenceEngine::invpropagate(int x, int y, pair<int, int> dim,
      vector<vector<double> > &from, vector<vector<double> > &to, 
      RewardMap &rewards) {



   if (x < 0 || y < 0 || x >= dim.first || y >= dim.second)
      return;

   if (x > 0 && y > 0) 
      to.at(x).at(y) = LogAdd(to.at(x).at(y), 
            from.at(x-1).at(y-1)+rewards.at(x-1,y-1));

   if (x > 0)
      to.at(x).at(y) = LogAdd(to.at(x).at(y), 
            from.at(x-1).at(y)+rewards.at(x-1,y));

   if (x > 0 && (y+1) < dim.second)
      to.at(x).at(y) = LogAdd(to.at(x).at(y), 
            from.at(x-1).at(y+1)+rewards.at(x-1,y+1));

   if (y > 0)
      to.at(x).at(y) = LogAdd(to.at(x).at(y), 
            from.at(x).at(y-1)+rewards.at(x,y-1));

   if ((y+1) < dim.second)
      to.at(x).at(y) = LogAdd(to.at(x).at(y), 
            from.at(x).at(y+1)+rewards.at(x,y+1));

   if ((x+1) < dim.first && y > 0)
      to.at(x).at(y) = LogAdd(to.at(x).at(y), 
            from.at(x+1).at(y-1)+rewards.at(x+1,y-1));

   if ((x+1) < dim.first)
      to.at(x).at(y) = LogAdd(to.at(x).at(y), 
            from.at(x+1).at(y)+rewards.at(x+1,y));

   if ((x+1) < dim.first && (y+1) < dim.second)
      to.at(x).at(y) = LogAdd(to.at(x).at(y), 
            from.at(x+1).at(y+1)+rewards.at(x+1,y+1)); 
}
 

void InferenceEngine::flattenPartitions(
      vector<vector<vector<double> > > &origPartition,
      vector<vector<double> > &collapsePartition) {

   collapsePartition.clear();

   collapsePartition.resize(origPartition.at(0).size(), 
         vector<double>(origPartition.at(0).at(0).size(), -HUGE_VAL));

   for (int t=0; t < origPartition.size(); t++) {
      for (int i=0; i < origPartition.at(t).size(); i++) {
         for (int j=0; j < origPartition.at(t).at(i).size(); j++) {
            //cout << i <<"  "<<j<<" !"<<endl;
            //cout << collap
            //   << collapsepartition.at(0).size()<<endl;
            collapsePartition.at(i).at(j) = LogAdd(
                   collapsePartition.at(i).at(j), 
                   origPartition.at(t).at(i).at(j));
         }
      }
   }
}

void InferenceEngine::cummulativeFlattenPartitions(
      vector<vector<vector<double> > > &origPartition,
      vector<vector<vector<double> > > &cummFlattenPartition) {

   cummFlattenPartition.clear();

   cummFlattenPartition.resize(origPartition.size(), 
         vector<vector<double> >(origPartition.at(0).size(), 
         vector<double>(origPartition.at(0).at(0).size(), -HUGE_VAL)));

   for (int t=0; t < origPartition.size(); t++) {
      for (int i=0; i < origPartition.at(t).size(); i++) {
         for (int j=0; j < origPartition.at(t).at(i).size(); j++) { 
            if (t > 0)
               cummFlattenPartition.at(t).at(i).at(j) = 
                  cummFlattenPartition.at(t-1).at(i).at(j);
            cummFlattenPartition.at(t).at(i).at(j) = LogAdd(
                   cummFlattenPartition.at(t).at(i).at(j),
                   origPartition.at(t).at(i).at(j));
         }
      }
   }

}

void InferenceEngine::combinePartitions(
      vector<vector<vector<double> > > & partition1, 
      vector<vector<vector<double> > > & partition2,
      vector<vector<double> > & partitionOut) {

   partitionOut.clear();

   partitionOut.resize(partition1.at(0).size(), 
         vector<double>(partition1.at(0).at(0).size(), -HUGE_VAL));

   for (int t=0; t < min(partition1.size(), partition2.size()); t++) {

      for (int i=0; i < partition1.at(t).size(); i++) {

         assert(partition1.at(t).at(i).size() == partition2.at(t).at(i).size());
         assert(partitionOut.at(i).size() == partition1.at(t).at(i).size());
         for (int j=0; j < partition1.at(t).at(i).size(); j++) {
            
            partitionOut.at(i).at(j) = LogAdd(
                  partitionOut.at(i).at(j), 
                  partition1.at(t).at(i).at(j) +
                  partition2.at(t).at(i).at(j)); 
         }
      }
   }
}


void InferenceEngine::edgeprop(vector<vector<double> > &from,
      vector<vector<double> > &to,
      int x0, int y0, int x1, int y1, double reward) {
   

   to.at(x1).at(y1) = LogAdd(to.at(x1).at(y1),
         from.at(x0).at(y0) + reward);
}

void InferenceEngine::timePartitions(vector<vector<double> > &frequencies,
      vector<vector<vector<double> > > &timeFrequencies,
      vector<vector<double> > &optCost, 
      double veloc, double sigma0, double sigma1, int T) {

   cout << "Time indexing occupancies"<<endl;

   timeFrequencies.resize(T, vector<vector<double> >(frequencies.size(),
            vector<double>(frequencies.at(0).size(), 0.0)));

   for (int x=0; x < frequencies.size(); x++) {

      for (int y=0; y < frequencies.at(x).size(); y++) {
         double norm = -HUGE_VAL;
         double opt = optCost.at(x).at(y);
         for (int t=0; t < T; t++) {
            double logprob = 
               -(veloc*t-opt)*(veloc*t-opt)/(2*(sigma0*sigma0+
                        t*sigma1*t*sigma1));
            norm = LogAdd(norm, logprob);
            //cout << "  NORM? "<<norm<<endl;
            timeFrequencies.at(t).at(x).at(y) = frequencies.at(x).at(y)
               +logprob;
            //cout << "LOGP : "<<logprob<<endl;
         }
         //cout << "NORM : "<<norm<<endl;
         for (int t=0; t < T; t++) {
            timeFrequencies.at(t).at(x).at(y) -= norm; 
            //cout << "  VAL: "<<timeFrequencies.at(t).at(x).at(y)<<endl;
         }
      } 
   } 
}


void WaveInferenceEngine::forward(pair<int, int> pos, int T, 
      RewardMap &rewards, vector<vector<vector<double> > > &partition) {

   partition.clear();

   pair<int, int> dims = rewards.dims();

   partition.resize(T, vector<vector<double> >(dims.first, 
            vector<double>(dims.second, -HUGE_VAL)));

   assert(pos.first >= 0 && pos.first < dims.first);
   assert(pos.second >= 0 && pos.second < dims.second);

   partition.at(0).at(pos.first).at(pos.second) = 0.0;

   /* Find radius that covers entire grid */
   int maxR = max(max(dims.first-pos.first, dims.first), 
         max(dims.second-pos.second, dims.second)); 

   for (int t=0; t < (T-1); t++) {

      vector<vector<bool> > mask(dims.first, vector<bool>(dims.second, false));
      int bnd, low;
      int x = pos.first;
      int y = pos.second;

      cout << "Propagate Iteration: "<<t<<endl;


      propagate(x, y, dims, partition.at(t), partition.at(t), 
            rewards.at(x, y));

      //cout << "Center: "<<partition.at(t).at(x).at(y)<<endl;

      mask.at(x).at(y) = true;

      for (int r=1; r <= maxR; r++) { 

         int x0 = x - r;
         int x1 = x + r;
         int y0 = y - r;
         int y1 = y + r; 

         /******************** 
          * PASS 1: Set mask *
          ********************/
         /**** WESTERN LINE (including corners) ****/
         if (x0 >= 0) {
            int low = max(y0, 0);
            int high = min(y1, dims.second-1); 
            for (int i=low; i <= high; i++) 
               mask.at(x0).at(i) = true; 
         }
         /**** NORTHERN LINE ****/
         if (y0 >= 0) {
            int low = max(x0, 0)+1;
            int high = min(x1, dims.first-1)-1; 
            for (int i=low; i <= high; i++) 
               mask.at(i).at(y0) = true; 
         }
         /**** EASTERN LINE (including corners) ****/ 
         if (x1 < dims.first) {
            int low = max(y0, 0);
            int high = min(y1, dims.second-1); 
            for (int i=low; i <= high; i++) 
               mask.at(x1).at(i) = true;
         }

         /**** SOUTHERN LINE ****/ 
         if (y1 < dims.second) {
            int low = max(x0, 0)+1;
            int high = min(x1, dims.first-1)-1; 
            for (int i=low; i <= high; i++) 
               mask.at(i).at(y1) = true; 
         }

         /********************* 
          * PASS 2: Propagate *
          *********************/ 
         /**** WESTERN LINE (including corners) ****/
         if (x0 >= 0) {
            int low = max(y0, 0);
            int high = min(y1, dims.second-1); 
            for (int i=low; i <= high; i++) 
               selectProp(x0, i, dims, partition.at(t), partition.at(t), 
                     partition.at(t+1), mask, rewards.at(x0, i)); 
         }

         /**** NORTHERN LINE ****/
         if (y0 >= 0) {
            int low = max(x0, 0)+1;
            int high = min(x1, dims.first-1)-1; 
            for (int i=low; i <= high; i++) 
               selectProp(i, y0, dims, partition.at(t), partition.at(t), 
                     partition.at(t+1), mask, rewards.at(i, y0)); 
         }
         
         /**** EASTERN LINE (including corners) ****/ 
         if (x1 < dims.first) {
            int low = max(y0, 0);
            int high = min(y1, dims.second-1); 
            for (int i=low; i <= high; i++) 
               selectProp(x1, i, dims, partition.at(t), partition.at(t), 
                     partition.at(t+1), mask, rewards.at(x1, i)); 
         }

        
         /**** SOUTHERN LINE ****/ 
         if (y1 < dims.second) {
            int low = max(x0, 0)+1;
            int high = min(x1, dims.first-1)-1; 
            for (int i=low; i <= high; i++) 
               selectProp(i, y1, dims, partition.at(t), partition.at(t), 
                     partition.at(t+1), mask, rewards.at(i, y1)); 
         }
      } 
   }
   cout << "done"<<endl;

   return;
}

void WaveInferenceEngine::backward(pair<int, int> pos, int T, 
      RewardMap &rewards, vector<vector<double> > &initial,
      vector<vector<vector<double> > > &partition) {

   partition.clear();

   pair<int, int> dims = rewards.dims();

   cout << "Dimensions: "<<dims.first<<" "<<dims.second<<endl;

   partition.resize(T, vector<vector<double> >(dims.first, 
            vector<double>(dims.second, -HUGE_VAL)));

   vector<vector<double> > curpartition(dims.first, 
         vector<double>(dims.second, 0.0));


   /* Find radius that covers entire grid */
   int maxR = max(dims.first-pos.first, dims.first) +
              max(dims.second-pos.second, dims.second);

   int bnd, low;
   int x = pos.first;
   int y = pos.second;


   for(int t=0; t < T; t++) { 
      cout << "ITERATION: "<<t<<endl;

      for (int r=maxR; r >= 1; r--) { 

         /* Propagate along line SW of origin */
         bnd = min(x, r);
         bnd -= (r <= x);
         low = max(0,y+r-(dims.second-1)); 
         for (int i=bnd; i >= low; i--) {
            cout << "BPT: "<<x-i<<"   "<<y+r-i<<endl;
            partition.at(t).at(x-i).at(y+r-i) = curpartition.at(x-i).at(y+r-i);
            propagate(x-i, y+r-i, dims, curpartition, curpartition, 
                  rewards.at(x-i, y+r-i)); 
            curpartition.at(x-i).at(y+r-i) = -HUGE_VAL; 
         }

         /* Propagate along line SE of origin */
         bnd = min(dims.second-1-y, r);
         bnd -= (r <= dims.second-1-y);
         low = max(0,x+r-(dims.first-1)); 
         for (int i=bnd; i >= low; i--) {
            cout << "BPT: "<<x+r-i<<"   "<<y+i<<endl; 
            propagate(x+r-i, y+i, dims, curpartition, curpartition, 
                  rewards.at(x+r-i,y+i));
            partition.at(t).at(x+r-i).at(y+i) = curpartition.at(x+r-i).at(y+i);
            curpartition.at(x+r-i).at(y+i) = -HUGE_VAL;
         }

         /* Propagate along line NE of ori
   occupancy.clear();
   occupancy.resize(prior.size(), 
         vector<double>(prior.at(0).size(), -HUGE_VAL)); 

   cout << "   Forward Inference"<<endl;

   engine.forward(current, 10, rewards, forwardPartitionB); 

    return flatposterior;
 gin */ 
         bnd = min(dims.first-1-x, r);
         bnd -= (r <= dims.first-1-x);
         low = max(0,r-y);
         for (int i=bnd; i >= low; i--) {
            cout << "BPT: "<<x+i<<"   "<<y-r+i<<endl; 
            propagate(x+i, y-r+i, dims, curpartition, curpartition, 
                  rewards.at(x+i, y-r+i));  
            partition.at(t).at(x+i).at(y-r+i) = curpartition.at(x+i).at(y-r+i);
            curpartition.at(x+i).at(y-r+i) = -HUGE_VAL;
         }
         
         /* Propagate along line NW of origin */
         bnd = min(y, r); 
         bnd -= (r <= y);
         low = max(0,r-x);
         for (int i=bnd; i >= low; i--) {
            cout << "BPT: "<<x-r+i<<"   "<<y-i<<endl; 
            propagate(x-r+i, y-i, dims, curpartition, curpartition, 
                  rewards.at(x-r+i,y-i));
            partition.at(t).at(x-r+i).at(y-i) = curpartition.at(x-r+i).at(y-i);
            curpartition.at(x-r+i).at(y-i) = -HUGE_VAL;
         }

      }


      /* Propagate from origin cell */
      propagate(x, y, dims, curpartition, curpartition, rewards.at(x,y)); 
      partition.at(t).at(x).at(y) = curpartition.at(x).at(y);
      curpartition.at(x).at(y) = -HUGE_VAL;


      for (int i=0; i < curpartition.size(); i++) { 
         for (int j=0; j < curpartition.at(i).size(); j++) {
            cout << "CURP: "<<i<<"  "<<j
               <<"  "<<curpartition.at(i).at(j)<<endl;
         }
      }
   }
}


void OrderedWaveInferenceEngine::forward(pair<int, int> pos, int T, 
      RewardMap &rewards, vector<vector<vector<double> > > &partition) {

   partition.clear();

   pair<int, int> dims = rewards.dims();
  // cout<<dims.first<<" "<<dims.second<<endl;

   partition.resize(T, vector<vector<double> >(dims.first, 
            vector<double>(dims.second, -HUGE_VAL)));


   assert(pos.first >= 0 && pos.first < dims.first);
   assert(pos.second >= 0 && pos.second < dims.second);

   partition.at(0).at(pos.first).at(pos.second) = 0.0;

   vector<vector<double> > costs;
   vector<pair<int, int> > order;
   double maximum;

   rewards.dijkstras(pos, costs, order, maximum);
   // this set order an vector contains position pairs in order of 
   // distances from the pos 

   for (int t=0; t < (T-1); t++) {

      vector<vector<bool> > mask(dims.first, vector<bool>(dims.second, false));
      
     // cout << "Forward Iteration: "<<t<<" size: "<<order.size()<<endl;

      for (int i=0; i < order.size(); i++) { 

         int x = order.at(i).first;
         int y = order.at(i).second;
         mask.at(x).at(y) = true; 
         selectProp(x, y, dims, partition.at(t), partition.at(t), 
               partition.at(t+1), mask, rewards.at(x, y)); 
         //cout << "FORWARD: "<<x<<" "<<y<<" "<<partition.at(t).at(x).at(y)<<endl;
      } 
   }
   //cout << "foward inference done"<<endl;

   return;
}

void OrderedWaveInferenceEngine::backward(pair<int, int> pos, int T, 
      RewardMap &rewards, vector<vector<double> > &initial, 
      vector<vector<vector<double> > > &partition) {

   partition.clear();

   pair<int, int> dims = rewards.dims();

   partition.resize(T, vector<vector<double> >(dims.first, 
            vector<double>(dims.second, -HUGE_VAL)));

   partition.at(0) = initial;

   assert(pos.first >= 0 && pos.first < dims.first);
   assert(pos.second >= 0 && pos.second < dims.second);

   //partition.at(0).at(pos.first).at(pos.second) = 0.0;

   vector<vector<double> > costs;
   vector<pair<int, int> > order;
   double maximum;

   rewards.dijkstras(pos, costs, order, maximum); 

   for (int t=0; t < (T-1); t++) {

      vector<vector<bool> > mask(dims.first, vector<bool>(dims.second, false));
      
     //cout << "Backward Iteration: "<<t<<" size: "<<order.size()<<endl;

      for (int i=order.size()-1; i >= 0; i--) { 

         int x = order.at(i).first;
         int y = order.at(i).second;

         mask.at(x).at(y) = true; 
         selectBackProp(x, y, dims, partition.at(t), partition.at(t), 
               partition.at(t+1), mask, rewards); 
      } 
   }
   cout << "done"<<endl; 
   
   return;
}

Predictor::Predictor(Grid &_grid, RewardMap &_rewards, 
      InferenceEngine &_engine) 
      : grid(_grid), rewards(_rewards), engine(_engine) {

   pair<int, int> dims = grid.dims();
  

   double weight = 1.0/(dims.first*dims.second);

   prior.resize(dims.first, vector<double>(dims.second, weight));
}

void Predictor::setStart(pair<int, int> _start) { 
   start = _start;

   cout << "Start Point Inference"<<endl;

   //cout << "   Forward Inference"<<endl;
   engine.forward(start, 10, rewards, forwardPartitionA); 
   // 10 is propogation times
   //cout << "   Flattening partitions"<<endl;
   engine.flattenPartitions(forwardPartitionA, forwardFlatPartitionA);

   //cout << "   Complete"<<endl;
}

void Predictor::setPrior(vector<vector<double> > &_prior) {
   prior = _prior;
}

 
double Predictor::predict(pair<int, int> current, 
      vector<vector<double> > &occupancy) {

   cout << "Future Prediction"<<endl;

   vector<vector<double> > initBackWeight(prior.size(),
         vector<double>(prior.at(0).size(), -HUGE_VAL));

   posterior.clear();
   posterior.resize(prior.size(), 
         vector<double>(prior.at(0).size(), -HUGE_VAL));

   occupancy.clear();
   occupancy.resize(prior.size(), 
         vector<double>(prior.at(0).size(), -HUGE_VAL)); 

  // cout << "   Forward Inference"<<endl;

   engine.forward(current, 10, rewards, forwardPartitionB); 

  // cout << "   Flattening partitions"<<endl;
   
   engine.flattenPartitions(forwardPartitionB, forwardFlatPartitionB);
   
   // compute posterior destination probability

   double sum = -HUGE_VAL;

   double normalizer = -HUGE_VAL;
   
   for (int i=0; i < prior.size(); i++) { 
      for (int j=0; j < prior.at(i).size(); j++) {
         posterior.at(i).at(j) = prior.at(i).at(j) 
            + forwardFlatPartitionB.at(i).at(j)
            - forwardFlatPartitionA.at(i).at(j);
         if (isnan(posterior.at(i).at(j))) continue;
         if (posterior.at(i).at(j) == -HUGE_VAL) continue;
         //cout << "POSTERIOR: "<<i<<"  "<<j<<"  "<<posterior.at(i).at(j)<<endl;
         sum = LogAdd(sum, posterior.at(i).at(j));
         if (prior.at(i).at(j) == 0.0) {
#if 0 
            cout << "NORMALIZER: "
				 <<i<<" "<<j<<" "
				 <<forwardFlatPartitionA.at(i).at(j)-
				   forwardFlatPartitionB.at(i).at(j)<<endl;
#endif
			/* MAIN CHANGE OF THE CODE:
			 * THE ORIGNIAL CODE RETURNS V(S1->G) AS NORMALIZER. 
			 * THE SENTENCE BELOW RETURNS 
			 * V(S1->G)-V(ST->G) TO GUARANTEE POSTIVE SEQUENCE
			 * LIKELIHOOD */
            normalizer = forwardFlatPartitionA.at(i).at(j)
				-forwardFlatPartitionB.at(i).at(j);
         }
      }
   }

  // cout <<"   Posterior Normalizer (" <<sum<<")"<<endl;

   for (int i=0; i < posterior.size(); i++) {
      for (int j=0; j < posterior.at(i).size(); j++) {
         posterior.at(i).at(j) -= sum;
         //cout << "POST: "<<i << "  "<<j<<"  "<<posterior.at(i).at(j)
         //   << "  "<<forwardFlatPartitionB.at(i).at(j)  
         //   << "  "<<forwardFlatPartitionA.at(i).at(j)<<endl; 
         initBackWeight.at(i).at(j) = -forwardFlatPartitionB.at(i).at(j) +
            posterior.at(i).at(j);// - rewards.at(i,j);
         //if (initBackWeight.at(i).at(j) > -HUGE_VAL)
           // cout << "Init Back: "<<i<<" "<<j<<"  "
            //   <<initBackWeight.at(i).at(j)<<endl;
      }
   }

 //  cout << "BACKWARD"<<endl;
   // calculate backwards probability including prior
   engine.backward(current, 10, rewards, initBackWeight, backwardPartition);


   vector<vector<vector<double> > > cummBackPartition(0);

//   cout << "CUMM"<<endl;
   engine.cummulativeFlattenPartitions(backwardPartition, cummBackPartition);

   vector<vector<double> > occupancy2;

 //  cout << "COMBINE:"<<endl;
   engine.combinePartitions(forwardPartitionB, cummBackPartition, occupancy);

   double norm = backwardPartition.at(0).at(current.first).at(current.second);
    //+ rewards.at(current.first, current.second);
#if 0
   cout << "COMBINING: "<<norm<<endl; 

   cout << "NORM2: "<<
      cummBackPartition.back().at(current.first).at(current.second)<<endl;

   cout << backwardPartition.at(0).at(current.first).at(current.second) << "  "
      << forwardPartitionB.at(0).at(current.first).at(current.second)<<endl;

   cout << backwardPartition.at(0).at(current.first+5).at(current.second+5) 
      << "  " 
      << forwardPartitionB.at(0).at(current.first+5).at(current.second+5)
      <<endl;
#endif
#if 0
   for (int i=0; i < backwardPartition.at(0).size(); i++) {
      for (int j=0; j < backwardPartition.at(0).at(i).size(); j++) {
         occupancy.at(i).at(j) = backwardPartition.at(0).at(i).at(j) + 
            forwardPartitionB.at(0).at(i).at(j)-norm;//+rewards.at(i,j);
         if (occupancy.at(i).at(j) > 0.0) {
            cout << "OCC: "<<i<<" "<<j<<"  "<<occupancy.at(i).at(j)<<endl;
            cout << "  "<<backwardPartition.at(0).at(i).at(j) << " "
               << forwardPartitionB.at(0).at(i).at(j)<<" "
               <<norm<<" "<<rewards.at(i,j)<<endl;

            cout << "  "<<initBackWeight.at(i).at(j)<<endl;
         }

         //cout << "OCC: "<<i<<"  "<<j<<"  "<<occupancy.at(i).at(j)<<endl;
      } 
   }
#endif

//   vector<vector<vector<double> > > timeOccupancy;

   vector<pair<int, int> > order;
   vector<vector<double> > optCosts;
   double maximum;



   //rewards.dijkstras(current, optCosts, order, maximum);

//   engine.timePartitions(occupancy, timeOccupancy, optCosts, 
  //       3.9, 1.7, 0.2, 40);

   //occupancy = backwardPartition.at(0);

   //occupancy = initBackWeight;
   //cout << "   Complete"<<endl;
   
   return normalizer;
}

MarkovModel::MarkovModel(Grid &grid, Evidence &evidence) {

   pair<int, int> dims = grid.dims();

   instances.resize(dims.first, 
         vector<vector<pair<int, int> > >(dims.second,
            vector<pair<int, int> >(0)));


   for (int i=0;  i < evidence.size(); i++) {

      vector<pair<int, int> > path = evidence.at(i);
      paths.push_back(path);

      for (int j=0; j < path.size(); j++) {

         int posX = path.at(j).first;
         int posY = path.at(j).second;

         instances.at(posX).at(posY).push_back(pair<int, int>(i,j));
      }
   }
}

double MarkovModel::eval(vector<pair<int, int> > &path) {

   double total = 0.0;
   for (int t=0; t < path.size(); t++) {
      cout << "before "<<t<<endl;
      vector<pair<int, int> > subpath = vector<pair<int, int> >(
            path.begin(), path.begin()+t);
      pair<int, int> next = (t < path.size() ? path.at(t) : 
         pair<int, int>(0,0)); 
      cout << "call"<<endl;
      total += evalNext(subpath, next); 
   }
   return total;
}

double MarkovModel::evalNext(vector<pair<int, int> > &path,
      pair<int, int> next, int histSize) {

   double val = 0.0; 
   double pseudo = 0.1;

   if (path.size() == 0) return 0.0;

   int x = path.back().first;
   int y = path.back().second;

   vector<pair<int, int> > examples = instances.at(x).at(y);

   vector<pair<pair<int, int>, int> > count;

   for (int i=0; i < examples.size(); i++) {

      int exInd = examples.at(i).first;

      vector<pair<int, int> > history = vector<pair<int, int> >(
            paths.at(exInd).begin(),
            paths.at(exInd).begin()+examples.at(i).second+1);

      if (matchLast(history, path, histSize)) {
         if (paths.at(exInd).size() <= examples.at(i).second+1) continue;
         pair<int, int> next = paths.at(exInd).at(examples.at(i).second+1);

         for (int j=0; j < count.size(); j++) {

            if (count.at(j).first == next) {
               count.at(j).second++;
               goto skipAdd;
            }
         }
         count.push_back(pair<pair<int, int>, int>(next, 1));
      } 
      skipAdd: ;
   } 

   int total = 0;
   bool match = false;
   // reason about matches
   for (int i=0; i < count.size(); i++) {
      if (count.at(i).first == next) {
         val += log(count.at(i).second+pseudo);
         match = true;
     }
      total += count.at(i).second;
   }

   if ((total < 1 && histSize > 1))  {
      return evalNext(path, next, histSize-1);
   }

   cout << "SHRINKAGE: "<<histSize<<endl;

   if (!match) 
      val += log(pseudo);
   double norm = total+8*pseudo;
   val -= log(norm);

   cout << "RETURNING: "<<val<<endl;

   return val;
}

bool MarkovModel::matchLast(vector<pair<int, int> > &a,
      vector<pair<int, int> > &b, int N) {
   for (int i=1, j=1; i <= a.size() && j <= b.size() && i <= N; i++, j++) {
      if (a.at(a.size()-i) != b.at(b.size()-j)) { 
         return false;
      }
   }
#if 0
   cout << "PATH: "<<endl;
   for (int i=0; i < a.size(); i++) 
      cout << a.at(i).first <<"  "<<a.at(i).second<<endl;
   cout << endl <<" PATH: "<<endl;
   for (int i=0; i < b.size(); i++) 
      cout << b.at(i).first<<"  "<<b.at(i).second<<endl;
#endif
   return true;
}
