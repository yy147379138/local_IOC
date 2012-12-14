#include "optimization.h"
#include <math.h>

#define NUMROBFEAT 2

double trajOptimizerplus::eval(vector<double> &params) {

   cout << "IN EVAL "<<itrcount++<<" "<<params.size()<<endl;


   
   for (int i=0; i < params.size(); i++) 
      cout << "PARAMS IN: "<<i<<" "<<params.at(i)<<endl;


   int factor = evidence.getFactor();

   pair<int, int> dims = grid.dims();
   int v_dim = seqFeat.num_V();

   /*
   pair<int, int> lowDims((int)ceil((float)dims.first/factor),
         (int)ceil((float)dims.second/factor));
   */
   vector<vector<vector<double> > >
	   prior(dims.first, vector<vector<double> >(dims.second, 
            vector<double> (v_dim,-HUGE_VAL))); 

   double obj = 0.0;
   vector<double> gradient(params.size(), 0.0); 
   vector<vector<vector<double> > > occupancy;
   vector<vector<double> > layerOccupancy;
   layerOccupancy.resize(dims.first,vector<double>(dims.second,-HUGE_VAL));
   vector<double> modelFeats, pathFeats;

   for (int i=0; i < evidence.size(); i++) {
      for (int j=0; j < params.size(); j++){ 
         cout << "  "<<j<<" "<<params.at(j);
	  }
	  cout<<endl;
   
	  cout << "Evidence #"<<i<<endl;
      vector<pair<int, int> >&  trajectory = evidence.at(i);
      vector<double>& velocityseq = evidence.at_v(i);
      pair<int,int>&  bot = evidence.at_bot(i);

	  //  robot local blurres features
      for (int r=1; r <= NUMROBFEAT; r++) {
		cout << "Adding  Robot Feature "<<r<<endl;
		RobotLocalBlurFeature robblurFeat(grid,bot,10*r);
   	    //	RobotGlobalFeature robFeat(grid,bot);
		posFeatures.push_back(robblurFeat);
	  }
	
	  cout << "   Creating feature array"<<endl;
      FeatureArray featArray2(posFeatures);
      FeatureArray featArray(featArray2, factor);
	  
	  for (int rr=1;rr<= NUMROBFEAT;rr++)
		posFeatures.pop_back();
 
      // split different posfeatures and seqfeature weights 
      vector<double> p_weights,s_weights;
      int itr = 0;
      for (;itr<featArray.size();itr++)
	      p_weights.push_back(params[itr]);
      for (;itr<params.size();itr++)
	      s_weights.push_back(params[itr]);

	  //cout<<"Params"<<endl;
      Parameters p_parameters(p_weights), s_parameters(s_weights);
/*    cout<<featArray.size()<<endl;
	  cout<<params.size()<<endl;
	  cout<<p_weights.size()<<endl;
	  cout<<s_weights.size()<<endl;
	  cout<<p_parameters.size()<<endl;
	  cout<<s_parameters.size()<<endl;
*/
      //cout<<"Reward"<<endl;
	  RewardMap rewards(featArray,seqFeat,p_parameters,s_parameters); 
      DisSeqPredictor predictor(grid, rewards, engine); 
      
	  // sum of reward along the trajectory
      double cost = 0.0;
	  //cout<< trajectory.size()<<endl;
      for (int j=0; j < trajectory.size(); j++){
		  //cout<<j<<" "<<trajectory.at(j).first<<" "<< trajectory.at(j).second<< " "<< seqFeat.getFeat(velocityseq.at(j))<<endl;
		  cost+=rewards.at(trajectory.at(j).first, trajectory.at(j).second, seqFeat.getFeat(velocityseq.at(j)));
	  }
      State initial(trajectory.front(),seqFeat.getFeat(velocityseq.front()));
      State destination(trajectory.back(),seqFeat.getFeat(velocityseq.back()));
	  //for (int k=0;k<v_dim;k++)
	  prior.at(destination.x()).at(destination.y()).at(destination.disV) = 0.0;

      cout << "Initial: "<<initial.x()<<"  "<<initial.y()<<"  "<<initial.disV<<endl;
      cout << "Destination: "<<destination.x()<<"  "
         <<destination.y()<<" "<<destination.disV<<endl;
      predictor.setStart(initial);
      predictor.setPrior(prior);

      double norm = predictor.forwardBackwardInference(initial, occupancy);

	  for (int l=0;l<v_dim;l++){
          BMPFile gridView(dims.first, dims.second);
		  for (int x= 0;x<dims.first;x++){
			  for(int y=0;y<dims.second;y++){
				  layerOccupancy.at(x).at(y) = occupancy.at(x).at(y).at(l);
			  }
		  }

          char buf[1024];
		  /* 
		  RobotGlobalFeature robblurFeat(grid,bot);
          gridView.addBelief(robblurFeat.getMap(), 0.0, 25, white, red);
          gridView.addVector(trajectory, blue, factor);
          gridView.addLabel(bot,green);
          sprintf(buf, "../figures/feat%04d_%d.bmp",i,l);
          gridView.write(buf);
          */

		  gridView.addBelief(layerOccupancy, -300.0, 5.0, white, red);
          //grid.addObstacles(gridView, black);
          gridView.addLabel(bot,green);
          gridView.addVector(trajectory, blue, factor);

          sprintf(buf, "../figures/train%04d_%d.bmp",i,l);
          gridView.write(buf);
      }


	  /*
      for (int i=0; i < occupancy.size(); i++)
         for (int j=0; j < occupancy.at(i).size(); j++) 
            if (occupancy.at(i).at(j) > -10)
               cout << i <<" "<<j<<"    "<<occupancy.at(i).at(j)<<endl; 
      */
      featArray.featureCounts(occupancy, modelFeats);

      featArray.featureCounts(trajectory, pathFeats);


	  seqFeat.featureCounts_vec(occupancy,modelFeats);
	  seqFeat.featureCounts_vec(velocityseq,pathFeats);

      for (int k=0; k < params.size(); k++) {
         double diff = pathFeats.at(k) - modelFeats.at(k);
         gradient.at(k) -= diff;
         cout <<" Gradient ("<< k << " -grad: "<< gradient.at(k) <<" -path: "<< 
			 pathFeats.at(k)<<" -model: "<< modelFeats.at(k)<<")";
      }
	  cout<<endl;
      cout << "OBJ: "<<cost-norm<< "  "<<cost<<"  "<<norm<<endl;
      obj += (cost - norm);
      /* obj is the path probability 
	   * cost is the sum of rewards: sum f(s,a)
	   * norm is V(s_1->G), since here s_T = G, V(s_T->G) = 0*/
      prior.at(destination.x()).at(destination.y()).at(destination.disV)
		  = -HUGE_VAL; 
   }

   cout << "RETURN OBJ: "<<-obj<<endl;

   params = gradient;

   return -obj;
}
 

