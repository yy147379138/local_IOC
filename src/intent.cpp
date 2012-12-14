#include "intent.h"

void Planner::simplePlan(vector<pair<int,int> >& pool,
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

void Planner::sociablePlan(vector<pair<int,int> >& pool,
    pair<int,int>& currentRob,int low, int high){
	cout<<"Generate plans"<<endl;
	vector<vector<double> > costs;
	vector<pair<int,int> > order;
	double maximum;
	rewards.dijkstras(currentRob,costs,order,maximum);
	pool.insert(pool.end(), order.begin()+low, order.begin()+high);
	//writePlans(pool);
}

double Planner::computeKLDivergence(vector<vector<double> >& novel,
	vector<vector<double> >& base){
	double H = 1.0;
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

double Planner::computeEntropy(vector<vector<double> >& P, double& sum){
    double H = 0.0;
	for (int x =0;x<P.size();x++)
	    for (int y=0;y<P.at(0).size();y++)
		   H-=exp(P.at(x).at(y)-sum)*(P.at(x).at(y)-sum);
    return H;
}

void Planner::planSimplePath(vector<pair<int,int> > & path, 
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

void Planner::writePlans(vector<pair<int,int> >& pool){
	pair<int,int> dims = grid.dims();
	BMPFile gridView(dims.first, dims.second);
	char buf[512];
     
    grid.addObstacles(gridView, black);
    gridView.addVector(pool, blue, 1);
    sprintf(buf, "../compare/plan.bmp"); 
    gridView.write(buf);
}

  	    
void IntentRecognizer::discrtForecast(vector<pair<int,int> >& traj, vector<double>& vels,
    vector<double>& times, pair<int,int>& rob,int evid_i,
	double interval){
	cout<<"ROB: "<<rob.first<<" "<<rob.second<<endl; 
#if 0
    //  robot local blurres features
    for (int r=1; r <= NUMROBFEAT; r++) {
        cout << "Adding  Robot Feature "<<r<<endl;
        RobotLocalBlurFeature robblurFeat(grid,rob,10*r);
     //	RobotGlobalFeature robFeat(grid,bot);
        posFeatures.push_back(robblurFeat);
    }
#endif

    cout << "   Creating feature array 2"<<endl;
    FeatureArray intent_featArray(posFeatures);
#if 0	  
    for (int rr=1;rr<= NUMROBFEAT;rr++)
		posFeatures.pop_back();
	#endif
    cout<<"Generating reward"<<endl;
    RewardMap r_rewards(intent_featArray,seqFeat,r_PosParams,r_SeqParams);
    RewardMap nr_rewards(intent_featArray,seqFeat,nr_PosParams,nr_SeqParams); 
      
    cout<<"  Declare predictors "<<endl;
    DisSeqPredictor r_predict(grid, r_rewards,  mdpr_engine); 
    DisSeqPredictor nr_predict(grid, nr_rewards,mdpnr_engine);
			  

    State initState(traj.front(), seqFeat.getFeat(vels.front()));
    State rState(rob,0);

	cout<<" Set start "<<endl;
	r_predict.setStart(initState);
	nr_predict.setStart(initState);
	  
    int index = 0;
    char buf[512];
	sprintf(buf, "../output/discrt%03d.dat", evid_i);
    ofstream outfile(buf);
	for (double tick = 0.0; index < traj.size(); tick+=interval) {

	    for ( ; index < traj.size() && times.at(index) < tick;
			index++); 
			if (index == traj.size() ) break;
 
	    cout << "Evidence: "<<evid_i<<"   timestep: "
            <<tick<<"  index: "<<index<<endl;
         
        vector<double>  pathcost (2,0.0);
        for (int j=0; j <=index; j++){
			pathcost.at(0) = pathcost.at(0)+
			    r_rewards.at(traj.at(j).first, traj.at(j).second,
				seqFeat.getFeat(vels.at(j)));
		    pathcost.at(1) = pathcost.at(1) +
				nr_rewards.at(traj.at(j).first, traj.at(j).second,
				seqFeat.getFeat(vels.at(j)));
     	 } 
         cout<<"path cost "<<pathcost[0]<<" "<<pathcost[1]<<endl;

		 cout<<" Prediction "<<endl; 
         State currentState(traj.at(index),seqFeat.getFeat(vels.at(index)));

         r_predict.forwardInference(currentState);
         nr_predict.forwardInference(currentState);
	  /* converting function for my data */

		 r_Distribute = r_predict.getObdistribute();
		 nr_Distribute = nr_predict.getObdistribute();
					 
		 genIntentPosterior(rob,pathcost);

	     cout<<"Compute beliefs"<<endl;
		 vector<double> beliefs(2,0.0);
		 beliefs.at(0) = pathcost[0]+r_predict.getNormalizer(rState);
		 beliefs.at(1) = pathcost[1]+nr_predict.getNormalizer(rState);
				 
		// writeOutput(DIS,traj, index,beliefs, tick,rob,evid_i,outfile);

	}

	outfile.close();
}


void IntentRecognizer::combineForecast(vector<pair<int, int> > & traj,
   vector<double>& vels,vector<pair<double, double> > & rawObs,
   vector<double>& times,vector<double>& rawTimes,pair<int,int>& 
   botinGrid,int evid_i,double interval){
 
    pair<int,int> start = traj.front();
	double startTime = times.front(); 
	pair<double,double>& end = rawObs.back();


	/* pedestrain prediction discrete model */
	cout << "   Creating feature array 1"<<endl;
    FeatureArray p_featArray(posFeatures);
    cout<<"Generating reward"<<endl;
    RewardMap p_rewards(p_featArray, p_Params);
    cout<<"  Declare predictors "<<endl;
    Predictor p_predict(grid, p_rewards, pp_engine); 
			  
    double prior_weight = -log(dimensions[0]*dimensions[1]);
	vector<vector<double> > occupancy;
    vector<vector<double> > prior2D(dimensions[0],vector<double> 
	    (dimensions[1],prior_weight));//prior = 1/|G| number of goals
    cout<<" Set Priors"<<endl;
    p_predict.setPrior(prior2D);
	cout<<" Set start "<<endl;
	p_predict.setStart(start);
	  
    /* linear-quadratic continuous model */
    LQContinuousPredictor lq_predictor(lq_engine,grid);
    lq_predictor.setPrior(prior2D);
    vector<vector<double> > likelihoods;
    lq_predictor.setOriginWrapper(rawObs.front());

    int rawIndex = 0;
    int prevRawIndex = 0;
    int gridIndex = 0;
    int restartIndex = 0;
	bool two_before = false; // Have seen turning two steps before
	bool one_before = false; // Have seen turning one step before
	bool turn_now = false;  // Seeing turning now
    char buf[512];
    sprintf(buf, "../output/cmb%03d.dat", evid_i);
    ofstream outfile(buf);
    for (double tick= startTime;rawIndex<rawObs.size();tick+=interval) {

		for (;gridIndex < traj.size()&&times.at(gridIndex) < tick;
					       gridIndex++);
			if (gridIndex == traj.size() ) break;
 
		for (;rawIndex < rawObs.size()&&rawTimes.at(rawIndex) < tick;
					       rawIndex++); 
			if (rawIndex == rawObs.size() ) break;

	    cout <<"Evidence: "<<evid_i<<" timestep: "
				<<tick<<" index: "<<rawIndex<<" previous: "<<prevRawIndex<<endl;
	
		turn_now = turnningDetector(rawObs,rawIndex,prevRawIndex,restartIndex);

		if(!turn_now && !one_before && two_before){
			restartIndex = rawIndex;
			prevRawIndex = rawIndex;
			lq_predictor.setOriginWrapper(rawObs.at(rawIndex));
			cout<<"***** Restart *********** Turnning: "<<rawIndex<<endl;
			}
		two_before = one_before;
		one_before = turn_now;


		/* PP PREDICTING...*/
		double  pathcost; 
        for (int j=0; j <= gridIndex; j++){
		    pathcost +=
		        p_rewards.at(traj[j].first, traj[j].second);
     	} 
       // cout<<"path cost "<<pathcost<<endl;

		p_predict.predict(traj.at(gridIndex), occupancy);
        posterior2D = p_predict.getPosterior();

		/* LQ PREDICTING ....*/
		lq_predictor.predictAll(rawObs,prevRawIndex,rawIndex,restartIndex);
        /*
		double botLikelihood = 
			lqr_predictor.predictPoint(rawObs,
						botinReal,prevIndex,rawIndex);
	    */
		prevRawIndex = rawIndex;
		likelihoods = lq_predictor.getLikelihoods(); 
		//likelihoods.at(botinGrid.first).at(botinGrid.second) = botLikelihood;
		genPosterior(prior2D,lqposterior,likelihoods);

		/* Combine the posterior */
        hybridposterior.resize(dimensions[0],vector<double>(
						dimensions[1],-HUGE_VAL));
		double sum = -HUGE_VAL;
	    for(int x=0;x<dimensions[0];x++){
			for(int y=0;y<dimensions[1];y++){
				hybridposterior.at(x).at(y) = posterior2D[x][y]
						+lqposterior[x][y];
		        sum = LogAdd(sum,hybridposterior.at(x).at(y));
	        }
		}
	 
		for(int x=0;x<dimensions[0];x++){
			for(int y=0;y<dimensions[1];y++){
				hybridposterior.at(x).at(y) -= sum;
			}
		}
		double maxV = -HUGE_VAL;
        double minV = HUGE_VAL;
		pair<int,int> predestGrid;
		pair<double,double> predestPoint;
    
		for(int ii=0; ii< dimensions[0]; ii++) { 
              for (int jj=0; jj < dimensions[1]; jj++) {
		
				  if(hybridposterior[ii][jj]>maxV){
						predestGrid.first = ii;
						predestGrid.second = jj;
				  }
				  maxV = max(maxV, hybridposterior.at(ii).at(jj));
				  minV = min(minV, hybridposterior.at(ii).at(jj));
              }
		}
			  
		predestPoint = grid.grid2Real(predestGrid.first,predestGrid.second);
		double dist = sqrt((end.first-predestPoint.first)*
						  (end.first-predestPoint.first)+
						  (end.second-predestPoint.second)*
						  (end.second-predestPoint.second));

				 
	    writeOutput(traj, gridIndex,tick-startTime,
					botinGrid,maxV,dist,evid_i,outfile);
     }

	outfile.close();
}

bool IntentRecognizer::turnningDetector(vector<pair<double,double> >& obs, 
			int index, int prev_index, int restart){

	for (int i = index; i >= prev_index; i--){
		if(i < restart+15){
			return false;
		}
		pair<double,double> pointS = obs.at(i-5);
		pair<double,double> pointM1 = obs.at(i-2);
		pair<double,double> pointM2 = obs.at(i-1);
		pair<double,double> pointE = obs.at(i);
	
		pair<double,double> vectorSM1 = make_pair(pointM1.first-pointS.first,
					pointM1.second-pointS.second);
		pair<double,double> vectorME1 = make_pair(pointE.first-pointM1.first,
					pointE.second-pointM1.second);
		pair<double,double> vectorSM2 = make_pair(pointM2.first-pointS.first,
					pointM2.second-pointS.second);
		pair<double,double> vectorME2 = make_pair(pointE.first-pointM2.first,
					pointE.second-pointM2.second);
		if ((product(vectorSM1,vectorME1)+product(vectorSM2,vectorME2))/2 < 0.20){
			//cout<<product(vector35,vector13)<<endl; 

			return true;
		}
	}
	return false;
}


void IntentRecognizer::active_forecast(vector<pair<int,int> >& traj, 
    vector<double>& vels,vector<double>& times, pair<int,int>& rob,
    int evid_i,double interval){
			 
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
    DisSeqPredictor r_predict(grid, r_rewards, mdpr_engine); 
    DisSeqPredictor nr_predict(grid, nr_rewards,mdpnr_engine);
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

double IntentRecognizer::genPosterior(vector<vector<double> >& prior,
			vector<vector<double> >& posterior,
			vector<vector<double> >& support){
			
	double sum = -HUGE_VAL;
	int width = prior.size();
	int height = prior.at(0).size();
	posterior.clear();
	posterior.resize(width,vector<double>(height,
			-HUGE_VAL));

    for(int x=0;x<width;x++){
	   for(int y=0;y<height;y++){
	       posterior.at(x).at(y) = prior.at(x).at(y)
			   +support.at(x).at(y);
		   sum = LogAdd(sum,posterior.at(x).at(y));
	   }
    }
	 
    for(int x=0;x<width;x++){
	   for(int y=0;y<height;y++){
	       posterior.at(x).at(y) -=sum;
	   }
    }
	return sum;
}


double IntentRecognizer::genIntentPosterior(pair<int,int>& robot, vector<double>& cost){
			
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

double IntentRecognizer::pre_GenPosterior(vector<double>& cost){
			
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


void IntentRecognizer::writeOutput(vector<pair<int,int> >& traj, int index, 
		double tick, pair<int,int>& rob,double V, double D,
		int traj_ind,ofstream& file){  
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

		gridView.addBelief(hybridposterior, -30, -0.0,jet);
		grid.addObstacles(gridView, black);
		gridView.addVector(subTraj, red, 1);
		gridView.addLabel(rob,green);
		sprintf(buf, "../compare/cmbpredict%03d-%03f.bmp", traj_ind, tick); 
		gridView.write(buf);
        
		gridView.addBelief(lqposterior, -30, -0.0,jet);
		grid.addObstacles(gridView, black);
		gridView.addVector(subTraj, blue, 1);
		gridView.addLabel(rob,green);
		sprintf(buf, "../compare/cmbpartial%03d-%03f.bmp", traj_ind, tick); 
		//gridView.write(buf);
        
		/* Format: time, hybrid prob. of trobot, hybrid prob. of the end
		 * max prob, distance between max point and groundtruth
		 */

	    double logloss = entropy(hybridposterior);
		cout <<tick<<" "
			 <<" POSTEIOR: "<<hybridposterior[rob.first][rob.second]
			 <<" "<<hybridposterior[traj.back().first][traj.back().second]
			 <<" log "<<logloss<<endl;

		file <<tick
			 <<" "<<logloss
			 <<" "<<hybridposterior[rob.first][rob.second]
			 <<" "<<hybridposterior[traj.back().first][traj.back().second]
			 <<" "<<V<<" "<<D<<endl;
		
}
		
void IntentRecognizer::writeOutput(vector<pair<int,int> >& userTraj, vector<pair<int,int> >&
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


double IntentRecognizer::entropy(vector<vector<double> >& P){
    double H = 0.0;
	for (int x =0;x<P.size();x++)
        for (int y=0;y<P.at(0).size();y++)
		   H-=exp(P.at(x).at(y))*P.at(x).at(y)/log(2);
	return H;
}
