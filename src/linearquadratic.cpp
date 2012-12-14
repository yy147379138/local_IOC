#include "linearquadratic.h"

template <typename T>
void expm(T& indice, T& Exp){
	EigenSolver<T> eigensolver(indice);
	T D = eigensolver.pseudoEigenvalueMatrix();
	T V = eigensolver.pseudoEigenvectors();
	for(int ii=0;ii<D.rows();ii++){
		D(ii,ii) = exp(D(ii,ii));
	}
	Exp = V*D*V.inverse();
}

template <typename T>
void logm(T& indice, T& Log){
	EigenSolver<T> eigensolver(indice);
	T D = eigensolver.pseudoEigenvalueMatrix();
	T V = eigensolver.pseudoEigenvectors();
	for(int ii=0;ii<D.rows();ii++){
		D(ii,ii) = log(D(ii,ii));
	}
	Log = V*D*V.inverse();
}


/* The method converts the raw observation to target related
 * states. The rawdata contains complete observation so that 
 * the first point is the origin. Since full observation 
 * is required this method is usually called in 
 * LQControlOptimizer or LQContinuousPredictor
 * VERSION: USING VARING TIME INTERVAL */
void ContinuousState::convertState(vector<VectorXd>& states,
	vector<pair<double,double> >& rawdata,
	const pair<double,double>& target){
    /*state: (x,y,x',y',x'',y'')*/	
	pair<double,double>& origin = rawdata.front();
//	cout<<rawdata.size()<<endl;
	double angle;
	if(target.first!=origin.first){
		angle = -atan2(target.second-origin.second,target.first
					-origin.first);
	}else{
		angle = target.second > origin.second? (-PI/2):(PI/2);
	}
	
	Rotation2D<double> rot(angle);
	VectorXd trans(2);
	trans<<-target.first,-target.second;
	trans = rot*trans;
	
	for(int i=0;i<rawdata.size();i++){
		VectorXd xy(2);
		xy<<rawdata.at(i).first,rawdata.at(i).second;
		xy = rot*xy+trans;
		VectorXd s(6);
		/* x and y */
		s.head(2) = xy;
		/* velocity  */
		if(!states.empty()){
			s.segment(2,2) = xy-states.back().head(2);
			s.tail(2) = s.segment(2,2)-states.back().segment(2,2);
		}else{
		   s.tail(4)<<0,0,0,0;
		}
		states.push_back(s);
		//cout<<"state: "<<endl<<s<<endl;
	}
}
	
/* The method converts the raw observation to target related
 * states. The rawdata contains partial observation. The
 * origin is passed by variable, usually the Origin set in
 * LQControlInference. So this method is usually called in 
 * LQControlInference
 * VERSION: USING VARING TIME INTERVAL*/
void ContinuousState::convertState(vector<VectorXd>& states,
	vector<pair<double,double> >& rawdata,
	const pair<double,double>& target,
	pair<double,double>& origin){
    /*state: (x,y,x',y',x'',y'')*/	
	double angle;
	if(target.first!=origin.first){
		angle = -atan2(target.second-origin.second,target.first
					-origin.first);
	}else{
		angle = target.second > origin.second? (-PI/2):(PI/2);
	}
	
	Rotation2D<double> rot(angle);
	VectorXd trans(2);
	trans<<-target.first,-target.second;
	trans = rot*trans;
	
	for(int i=0;i<rawdata.size();i++){
		VectorXd xy(2);
		xy<<rawdata.at(i).first,rawdata.at(i).second;
		xy = rot*xy+trans;
		VectorXd s(6);
		/* x and y */
		s.head(2) = xy;
		/* velocity  */
		if(!states.empty()){
			s.segment(2,2) = xy-states.back().head(2);
			s.tail(2) = s.segment(2,2)-states.back().segment(2,2);
		}else{
		   s.tail(4)<<0,0,0,0;
		}
		states.push_back(s);
	}
	
}

void ContinuousState::empiricalExpectation(vector<VectorXd>& seq, M_6& EE){
     EE.setZero();
	 int T = seq.size();
	 for(int t=0;t<T;t++){
		 EE.noalias() +=seq.at(t)*seq.at(t).transpose();
	 }
}

void ContinuousState::empiricalExpectation(vector<pair<double,double> >& 
			rawTraj,pair<double,double>& target,
			M_6& EE){
	vector<VectorXd> stateSeq;
	convertState(stateSeq,rawTraj,target);
	empiricalExpectation(stateSeq,EE);
}


void LQControlInference::valueInference(int T){
	D = M;
	G << 1,1,1,1,1,1;
	for(int t=0;t<T;t++){
		//cout<<"LQ value iteration: "<<t<<endl;
		M_2 Caa = B.transpose()*D*B;
		if (Caa.determinant()==0){
			cout<<"Matrix Caa is singular"<<endl;
			break;
		}
		M_2_6 Cas = B.transpose()*D*A;
		M_6 Css = A.transpose()*D*A;
		C.topLeftCorner<2,2>() = Caa;
		C.topRightCorner<2,6>() = Cas;
		C.bottomLeftCorner<6,2>() = Cas.transpose();
		C.bottomRightCorner<6,6>() = Css;

		Col_2 Fa = B.transpose()*G;
		Col_6 Fs = A.transpose()*G;
		F << Fa,Fs;
        D = Css + M - Cas.transpose()*Caa.inverse()*Cas;
		G = Fs - Cas.transpose()*Caa.inverse()*Fa;

	}
	//cout<<"C matrix: "<<endl<<C<<endl;
	//cout<<"D matrix: "<<endl<<D<<endl;
	//cout<<"G matrix: "<<endl<<G<<endl;
	valid_params = true;
}

void LQControlInference::computeDistribution(VectorXd& s0, int T){
	if(!valid_params){
		cout<<"The parameters are not updated. Please run value inference!"
			<<endl;
		return;
	}
	T = 1.1*T;
	resetDistribution();

    Gaussian* Ps0 = new Gaussian(s0,Sigma);
	state_distribution.push_back(Ps0);
	/* The mean of P(at|st) is a linear function of 
	 * st which is As*st+b 
	 * Covariance matrix L */
	M_2 L = 0.5*C.topLeftCorner<2,2>().inverse();
	M_2_6 As = -2*L*C.topRightCorner<2,6>();
    Col_2 b = -L*F.head(2);
    M_6_8 K;
	K<<A,B;

	for(int t=0;t<T;t++){
		Col_2 a_mean = As*state_distribution.back()->mean+b;
		M_2 a_cov = L+As*state_distribution.back()->covariance*
			As.transpose();
		Gaussian* Pat = new Gaussian(a_mean,a_cov);
		action_distribution.push_back(Pat);

		/* zt=[st,at]^T */
		Col_8 z_mean;
		z_mean<<state_distribution.back()->mean,a_mean;
		M_8 z_cov;
		M_6 s_cov = state_distribution.back()->covariance;
		z_cov<<s_cov,s_cov*As.transpose(),As*s_cov,a_cov;

		/* P(st+1|st,at) = P(st+1|zt) ~ N(st+1|K*zt,Sigma) */
		Col_6 nextst_mean = K*z_mean;
		M_6 nextst_cov = Sigma+K*z_cov*K.transpose();
		Gaussian* Pnextst = new Gaussian(nextst_mean,nextst_cov);
		state_distribution.push_back(Pnextst);	
	}
}

void LQControlInference::quadraticExpectation(M_6& E){
	
	E.setZero();
	if (state_distribution.empty()){
		cout<<"The state distribution is not generated!"<<endl;
		return;
	}
	int T = state_distribution.size();
	for(int t=0;t<T;t++){
		E.noalias() += state_distribution[t]->mean*
			state_distribution[t]->mean.transpose()+
			state_distribution[t]->covariance;
	}

}

double LQControlInference::cumulativeCosts(vector<VectorXd>&
			s_sequence,int index,M_6& quadraticSum){
	
	quadraticSum.setZero();
	for(int t=0;t<index;t++){
		quadraticSum.noalias() +=s_sequence[t]*s_sequence[t].transpose();
	}
	return (double) (M*quadraticSum).trace();
}

double LQControlInference::costTransformation(M_6& quadraticSum,
		Col_6& linearSum,M_6& R, Col_6& d, int length){
	return (double)(M*R*quadraticSum*R.transpose()+length*
		M*d*d.transpose()+2*M*R*linearSum*d.transpose()).trace();
}

double LQControlInference::getQ(VectorXd& a, VectorXd& s){
	Col_8 vec;
	vec<<a,s;
	double temp1 = vec.transpose()*C*vec;
	double temp2 = vec.transpose()*F;
	return (temp1+temp2);
}

double LQControlInference::getV(VectorXd& s){
	double temp1 = s.transpose()*(D-M)*s;
	double temp2 = s.transpose()*G;
	return (temp1+temp2);
}

void LQControlInference::resetDistribution(){
	for(vector<Gaussian* >::iterator it1 = state_distribution.begin();
				it1!=state_distribution.end();++it1){
		delete *it1;
	}
	state_distribution.clear();
	for(vector<Gaussian* >::iterator it2 = action_distribution.begin();
				it2!=action_distribution.end();++it2){
		delete *it2;
	}
	action_distribution.clear();
}
	
void LQControlInference::resetInference(){
		stateQuadrSum.setZero();
		stateLinSum.setZero();
		actionQuadrSum.setZero();
		actionLinSum.setZero();
		crossSum.setZero();
		/* Clear the history of state sequence*/
		firstState_seq.clear();
}

void LQControlInference::forwardInference(vector<pair<double,double> >& traj,
		int prev,int current, map<pair<double,double>,double>& likelihoods, int start){
    /* Extract new observation and push into newObs 
	 * cState converts the newObs data to states and
	 * add into firstState_seq
	 * */
	vector<pair<double,double> > newObs;

	if(prev >= start && prev <= current-1 && current <= traj.size()){
	  newObs.insert(newObs.end(),traj.begin()+prev,
				  traj.begin()+current);
	}else{
		cout<<"Invalid input pointers!"<<endl;
		return;
	}
	for(map<pair<double,double>,double>::iterator it = likelihoods.begin();
				it!=likelihoods.end();++it){
		if(it==likelihoods.begin()){
			cState.convertState(firstState_seq,newObs,it->first,Origin);
			stateMoments(prev-start);
		}
		it->second = 
			sequenceLikelihood(likelihoods.begin()->first,it->first,current-1-start);
	}
}

double LQControlInference::forwardInference(vector<pair<double,double> >& traj,
		int prev,int current,
		pair<double,double>& point){
	vector<pair<double,double> > newObs;
	if(prev>=0 && prev<=current-1&&current<=traj.size()){
	  newObs.insert(newObs.end(),traj.begin()+prev,
				  traj.begin()+current);
	}else{
		cout<<"Invalid input pointers!"<<endl;
		return -HUGE_VAL;
	}
    cState.convertState(firstState_seq,newObs,point,Origin);
    stateMoments(prev);
    return sequenceLikelihood(make_pair(-1.0,-1.0),make_pair(-1.0,-1.0),
				current-1);

}

double LQControlInference::forwardInference(vector<VectorXd>&
			externalStates){
	resetInference();
	stateMoments(externalStates);
	return sequenceLikelihood(make_pair(-1.0,-1.0),make_pair(-1.0,-1.0),
				externalStates.size()-1);
}

void LQControlInference::stateMoments(int prev){
	int prevPtr;
	if(prev==0){
		prevPtr = 0;
	}else{
		prevPtr = prev-1;
	}
	for(vector<VectorXd>::iterator it=firstState_seq.begin()+prevPtr;
				it!=firstState_seq.end()-1;++it){
		stateLinSum.noalias() += *it;
		stateQuadrSum.noalias() += *it*(it->transpose());
		Vector2d a = (it+1)->segment(2,2);
		actionLinSum.noalias() += a;
		actionQuadrSum.noalias() += a*a.transpose();
		crossSum.noalias() += a*(it->transpose());
	}
}

void LQControlInference::stateMoments(vector<VectorXd>& states){
	for(vector<VectorXd>::iterator it=states.begin();
				it!=states.end()-1;++it){
		stateLinSum.noalias() += *it;
		stateQuadrSum.noalias() += *it*(it->transpose());
		Vector2d a = (it+1)->segment(2,2);
		actionLinSum.noalias() += a;
		actionQuadrSum.noalias() += a*a.transpose();
		crossSum.noalias() += a*(it->transpose());

	}
}

double LQControlInference::sequenceLikelihood(vector<VectorXd>& states){
	double obj = 0.0;
	for(vector<VectorXd>::iterator it=states.begin();
				it!=states.end()-1;++it){
		VectorXd a = (it+1)->segment(2,2);
        //cout<<getV(*it)-getQ(a,*it)<<endl;
		obj+=getV(*it)-getQ(a,*it);
	}
	return obj;
}
	
double LQControlInference::determinantLikelihood(vector<VectorXd>& states){
	double obj = 0.0;
	for(vector<VectorXd>::iterator it=states.begin();
				it!=states.end();++it){
		obj-=it->transpose()*M*(*it);
	}
	obj+=getV(states.front())-getV(states.back());
	return obj;
}

double LQControlInference::sequenceLikelihood(const pair<double,double>& from,
			const pair<double,double>& to, int length){
	if(!valid_params){
		cout<<"The parameters are not updated. Please run value inference!"
			<<endl;
		return -1;
	}

	double L = 0;
	if(from==to){
		L = stateLinSum.transpose()*(G-F.tail(6)) + 
			((D-M-C.bottomRightCorner(6,6))*stateQuadrSum).trace() - 
			(C.topLeftCorner(2,2)*actionQuadrSum).trace() - 
			actionLinSum.transpose()*F.head(2) -
            2*(C.bottomLeftCorner(6,2)*crossSum).trace();
	}else{
		/* Compute transformation between two target coordinates */
		/* Transimission: S_to = R(to->from)*S_from + trans(to->from)*/
		double angle;
		angle = atan2(from.second-Origin.second,from.first
					-Origin.first) - atan2(to.second-Origin.second,
							to.first-Origin.first);
		Matrix2d rot2(2,2);
		rot2 = Rotation2D<double>(angle);
		M_6 Rot6;
		Rot6.setZero();
		Rot6.topLeftCorner(2,2) = rot2;
		Rot6.bottomRightCorner(2,2) = rot2;
		Rot6.block(2,2,2,2) = rot2;
		VectorXd trans2(2);
		trans2<<from.first-to.first,from.second-to.second;
		angle = -atan2(to.second-Origin.second,to.first-Origin.first);
		Rotation2D<double> rotTo(angle);
		trans2 = rotTo*trans2;
		Col_6 trans6;
		trans6.setZero();
		trans6.head(2) = trans2;

		/* Compute the likelihood */
		double quadr_term1 = 
		((D-M-C.bottomRightCorner(6,6))*(Rot6*stateQuadrSum*Rot6.transpose()
			+2*Rot6*stateLinSum*trans6.transpose()
	        +length*trans6*trans6.transpose())).trace();
		double quadr_term2 = -(C.topLeftCorner(2,2)*rot2*actionQuadrSum*rot2. 
					transpose()).trace();
		double quadr_term3 = -2*(C.bottomLeftCorner(6,2)*rot2*crossSum*
					Rot6.transpose()).trace();
		double lin_term1 = -2*(C.bottomLeftCorner(6,2)*rot2*actionLinSum*
					trans6.transpose()).trace();
		double lin_term2 = -(rot2*actionLinSum).transpose()*F.head(2);
		double lin_term3 = ((Rot6*stateLinSum).transpose()+
					length*trans6.transpose())*(G-F.tail(6));

		L = quadr_term1+quadr_term2+quadr_term3+lin_term1+
				lin_term2+lin_term3;
	}

	return L;
}

void LQControlInference::checkStateTransform(vector<VectorXd>& fromStates,
			vector<VectorXd>& toStates,const pair<double,double>& from,
			const pair<double,double>& to){
	toStates.clear();

	double angle;
	angle = atan2(from.second-Origin.second,from.first
				-Origin.first) - atan2(to.second-Origin.second,
						to.first-Origin.first);
	Matrix2d rot2(2,2);
	rot2 = Rotation2D<double>(angle);
	M_6 Rot6;
	Rot6.setZero();
	Rot6.topLeftCorner(2,2) = rot2;
	Rot6.bottomRightCorner(2,2) = rot2;
	Rot6.block(2,2,2,2) = rot2;
	VectorXd trans2(2);
	trans2<<from.first-to.first,from.second-to.second;
	angle = -atan2(to.second-Origin.second,to.first-Origin.first);
	Rotation2D<double> rotTo(angle);
	trans2 = rotTo*trans2;
	Col_6 trans6;
	trans6.setZero();
	trans6.head(2) = trans2;

	for(vector<VectorXd>::iterator it = fromStates.begin();
				it!=fromStates.end();it++){
		VectorXd s(6);
		s = Rot6*(*it)+trans6;
		toStates.push_back(s);
	}

}


void LQControlOptimizer::testSeqLikelihood(vector<VectorXd>& states){
	cout<<"Testing..."<<endl;
	double groundTruth = lq.sequenceLikelihood(states);
	double testee = lq.forwardInference(states);
	if((int)(groundTruth*10)!=(int)(testee*10)){
		cout<<"Ground truth: "<<groundTruth
			<<"Testee: "<<testee<<endl;
	}
	assert((int)(groundTruth*10)==(int)(testee*10));
}


double LQControlOptimizer::optimize_v(double step, int itrTimes, double thresh,
			int mode){
	int count = 0;
	cout<<"Start optimization: "<<endl<<lq.getM()<<endl;
	M_6 diff_M(lq.getM());
	double norm = 120.0;
	double prev_obj = -10000000;
	double obj = prev_obj+1+thresh;

	char buf[256];
    sprintf(buf,"../obj/stochastic.dat");
	ofstream file(buf);

	M_6 modelE;
	M_6 empiricalE;
    M_6 gradients;

	if(mode==BATCH_EXP){
		while(count<itrTimes && prev_obj<obj+thresh){
			count++;
		    gradients.setZero();
            lq.valueInference(50);
			prev_obj = obj;
			obj = 0.0;
		    for(int i=0;i<statesSet.size();i++){
				lq.computeDistribution(statesSet.at(i).front(),
						(int)statesSet.at(i).size());
			    lq.quadraticExpectation(modelE);
				cs.empiricalExpectation(statesSet.at(i),empiricalE);
				gradients-= empiricalE - modelE;
				obj+= lq.forwardInference(statesSet.at(i));
		    }
#if 0
	    cout<<"Gradients : "<<endl<<gradients<<endl;
#endif
			norm = gradients.norm();
			M_6& prev_M = lq.getM();
			logm(prev_M,prev_M);
			M_6 update = prev_M+step/(count+3)*gradients/statesSet.size();
			expm(update,update);
			diff_M = update-prev_M;
			lq.setM(update);

			cout<<"Iteration:  "<<count
					<<" Objective function: "<<obj<<endl;
#if 0
		cout<<"Previous: "<<endl<<prev_M<<endl
			<<"Update: "<<endl<<update<<endl;
		cout<<"M diff_Merence: "<<endl<<diff_M<<endl;
#endif
		}
	}else if(mode==STOCHASTIC_EXP){
		M_6 all_origin_M;

		while(count<itrTimes&&prev_obj<obj+thresh ){
			prev_obj = obj;
			obj = 0.0;
		    count++;
			all_origin_M << lq.getM();
		    for(int i=0;i<statesSet.size();i++){
		        gradients.setZero();
                lq.valueInference(50);
				lq.computeDistribution(statesSet.at(i).front(),
						(int)statesSet.at(i).size());
			    lq.quadraticExpectation(modelE);
				cs.empiricalExpectation(statesSet.at(i),empiricalE);
				gradients-= empiricalE - modelE;
				double temp = lq.forwardInference(statesSet.at(i));
				obj += temp;
				//cout<<"i: "<<i<<" "<<temp<<endl;

			//	testSeqLikelihood(statesSet.at(i));
				norm = gradients.norm();
				M_6& prev_M = lq.getM();
				logm(prev_M,prev_M);
				M_6 update = prev_M+step/(count+3)*gradients;
				expm(update,update);
				diff_M = update-prev_M;
				lq.setM(update);
#if 0
			cout<<"Previous: "<<endl<<prev_M<<endl
				<<"Update: "<<endl<<update<<endl;
			cout<<"M diff_Merence: "<<endl<<diff_M<<endl;
#endif
		    }
			cout<<"Iteration:  "<<count
						<<" Objective function: "<<obj<<endl;
			write_obj(obj, count, file);
		}
		lq.setM(all_origin_M);

	}else if(mode==STOCHASTIC_LINEAR){
		while(count<itrTimes && prev_obj<obj+thresh ){
		    for(int i=0;i<statesSet.size();i++){
				count++;
		        gradients.setZero();
                lq.valueInference(50);
				lq.computeDistribution(statesSet.at(i).front(),
					(int)statesSet.at(i).size());
			    lq.quadraticExpectation(modelE);
				cs.empiricalExpectation(statesSet.at(i),empiricalE);
				gradients-= empiricalE - modelE;
				prev_obj = obj;
				obj = lq.forwardInference(statesSet.at(i));
#if 0 
			  cout<<"Model: "<<endl<<modelE<<endl<<" Empirical: "<<endl
vector<double>* timestamp
				<<empiricalE<<endl;
	          cout<<"Gradients : "<<endl<<gradients<<endl;
#endif
				norm = gradients.norm();
				cout<<"Iteration:  "<<count
					    <<"  Gradients norm: "<<norm
						<<" Objective function: "<<obj<<endl;
						//<<" "<<lq.sequenceLikelihood(statesSet.at(i))<<endl;

				M_6 prev_M = lq.getM();
				M_6 update = prev_M + step/(count+3)*gradients;
				for (int ii=0;ii<update.rows();ii++){
					if (update(ii,ii)<=0){
					  update(ii,ii) = 0.1;
					  cout<<"Adjust diagonal elements."<<endl;
					}
			    }
				diff_M = update-prev_M;
				lq.setM(update);
#if 0 
			cout<<"Previous: "<<endl<<prev_M<<endl
				<<"Update: "<<endl<<update<<endl;
			cout<<"M diff_Merence: "<<endl<<diff_M<<endl;
#endif
		    }
		}
	}else if(mode==SMALL_BATCH_EXP){
		while(count<itrTimes && prev_obj<obj+thresh){
			count++;
		    gradients.setZero();
            lq.valueInference(50);
			prev_obj = obj;
			obj = 0.0;
		    for(int i=0;i<statesSet.size();i++){
				lq.computeDistribution(statesSet.at(i).front(),
						(int)statesSet.at(i).size());
			    lq.quadraticExpectation(modelE);
				cs.empiricalExpectation(statesSet.at(i),empiricalE);
				gradients-= empiricalE - modelE;
				obj += lq.forwardInference(statesSet.at(i));
#if 0
			  cout<<"Model: "<<endl<<modelE<<endl<<" Empirical: "<<endl
				<<empiricalE<<endl;
#endif
		    }
			norm = gradients.norm();
			cout<<"Iteration:  "<<count
			        <<"  Gradients norm: "<<norm
					<<" Objective function: "<<obj<<endl;

			M_6& prev_M = lq.getM();
			logm(prev_M,prev_M);
			M_6 update = prev_M+step/(count+3)*gradients;
			expm(update,update);
			diff_M = update-prev_M;
			lq.setM(update);
#if 0
		cout<<"Previous: "<<endl<<prev_M<<endl
			<<"Update: "<<endl<<update<<endl;
		cout<<"M difference: "<<endl<<diff_M<<endl;
#endif
		}
	}

	cout<<"Model: "<<endl<<modelE<<endl<<" Empirical: "<<endl
				<<empiricalE<<endl;
	cout<<"---------------------------"<<endl;

	cout<<"Total updation times: "<<count<<" Objective function: "
		<<obj<<" Prev obj: "<<prev_obj<<" Norm: "<<norm<<
		" Step: "<<step/log(count+3)<<endl;
	cout<<"M: "<<endl<<lq.getM()<<endl;
    string method;
	switch(mode) {
		case STOCHASTIC_EXP:
			method = "Stochastic exponentiated gradient";
			break;
		case BATCH_EXP:
			method = "Batch exponentiated gradient";
			break;
		case SMALL_BATCH_EXP:
			method = "Small batches exponentiated gradient";
			break;
		case STOCHASTIC_LINEAR:
			method = "Stochastic linear gradient";
			break;
		default:
			method = "Un-specified";
			break;
	}
	cout<<"optimization method: "<<method<<endl;
	file.close();
	return norm;
	
}




double LQControlOptimizer :: backtrack(int itrTimes, double thresh,
			double alpha, double beta, double mini_step, int mode){
	/**
	 * Reminder : lq is the linear-quadratic engine behind,
	 * which provides the obj funtion and the cost matrix M
	 */

	/**
	 * Record step sizes for each traj
	 */
	char buf[256];
    sprintf(buf,"../obj/backtrack.dat");
	ofstream file(buf);
	vector<double> step_sizes((int)statesSet.size(), 0.0001);
	int count = 0;
	cout<<"Start optimization: "<< endl <<
		lq.getM() << endl;
	M_6 diff_M(lq.getM());
	double prev_obj_sum = -10000000;
	double obj_sum = prev_obj_sum+1+thresh;

    M_6 gradients;
/**
 * At present, only the stachastic exponentiated
 * gradient with backtrack is implemented. 
 * TODO:The batch, linear version backtrack;
 */

	if(mode==BATCH_EXP){
	}else if(mode==STOCHASTIC_EXP){
		/**
		 * The most Outside loop that updates the parameter M
		 */
		M_6 all_origin_M;
		while(count < itrTimes && prev_obj_sum < obj_sum+thresh){
			prev_obj_sum = obj_sum;
			obj_sum = 0.0;
		    count++;
			all_origin_M << lq.getM();
		    for(int i = 0;i < statesSet.size(); i++){
                lq.valueInference(50); // compute legal parameters
				M_6 original_M;
				original_M << lq.getM();
				/**
				 * Initial objective function value
				 */
				double base =  lq.forwardInference(statesSet.at(i));
			    /**
				 * Compute gradient 
				 */
				get_gradient(gradients, statesSet.at(i));	

				double bound;
				double single_obj; 
				double approx;
               // cout <<"base: " << base << " bound: " << bound << endl;
				while(1){
					/*
					 * Start from original x
					 */
					lq.setM(original_M);
					/*
					 * get delta_x and new obj
					 */
					update_M(gradients, diff_M, step_sizes[i]);
					single_obj = lq.forwardInference(statesSet.at(i));
					/*
					 * Compute linear approximation bound
					 * Matrix differential:
					 * df = tr(A^T dX)
					 */
					bound =  (gradients.transpose() * diff_M).trace();
					approx = base + alpha * step_sizes[i] * bound;
					
					
			//		cout << i <<" obj: " << single_obj << "  base : "<< base
			//			 << " bound: " << bound 
			//			 << " approx	lq.setM(original_M);
						single_obj = lq.forwardInference(statesSet.at(i));
					if(bound < 0){
						break;
					}
					if(single_obj > approx){
						break;
					}
					step_sizes.at(i) *= beta;
					if(step_sizes[i] < mini_step){
						lq.setM(original_M);
						single_obj = lq.forwardInference(statesSet.at(i));

					//	cout<< "Less that mini step when training over traj: "
					//		<< i << " obj: " << single_obj <<" " << base <<endl;
						lq.setM(original_M);
						single_obj = lq.forwardInference(statesSet.at(i));
						break;
					}
			//		cout<< i << " current step size: " << step_sizes[i] << endl;
				}

				obj_sum += single_obj;
#if 0
			cout<<"Previous: "<<endl<<prev_M<<endl
				<<"Update: "<<endl<<update<<endl;
			cout<<"M difference: "<<endl<<diff_M<<endl;
#endif
		    }
			cout<<"Iteration:  "<<count
						<<" Objective function: "<<obj_sum<<endl;
			write_obj(obj_sum, count, file);
		}
		lq.setM(all_origin_M);
	}else if(mode==STOCHASTIC_LINEAR){
	}else if(mode==SMALL_BATCH_EXP){
	}

	cout<<"---------------------------"<<endl;

	cout<<"Total updation times: "<<count<<" Objective function: "
		<<obj_sum<<" Prev obj_sum: "<<prev_obj_sum
		<<endl;
	cout<<"M: "<<endl<<lq.getM()<<endl;
    string method;
	switch(mode) {
		case STOCHASTIC_EXP:
			method = "Stochastic exponentiated gradient";
			break;
		case BATCH_EXP:
			method = "Batch exponentiated gradient";
			break;
		case SMALL_BATCH_EXP:
			method = "Small batches exponentiated gradient";
			break;
		case STOCHASTIC_LINEAR:
			method = "Stochastic linear gradient";
			break;
		default:
			method = "Un-specified";
			break;
	}
	cout<<"optimization method: Stochastic linear gradient with backtrack"<<endl;
	file.close();
	return obj_sum;
	
}

void LQControlOptimizer :: get_gradient(M_6& gradient, 
			vector<VectorXd>& states){

	gradient.setZero();
	M_6 modelE;
	M_6 empiricalE;
	lq.computeDistribution(states.front(),
				(int)states.size());
	lq.quadraticExpectation(modelE);
	cs.empiricalExpectation(states,empiricalE);
	gradient -= empiricalE - modelE;		
#if 0
	cout<<"Model: "<<endl<<modelE<<endl<<" Empirical: "<<endl
				<<empiricalE<<endl;
#endif
}

void LQControlOptimizer :: update_M(M_6& gradients, M_6& diff_M, 
			double step){
	M_6& prev_M = lq.getM();
	logm(prev_M,prev_M);
	M_6 update = prev_M + step * gradients;
	expm(update,update);
	diff_M = update - prev_M; // delta M
	lq.setM(update);
    lq.valueInference(50); // compute legal parameters
}

void LQControlOptimizer :: get_next_M(M_6& gradients, M_6& original_M, 
		M_6& next_M, double step){
	M_6 prev_M;
	prev_M << original_M;
	logm(prev_M,prev_M);
    next_M = prev_M + step * gradients;
	expm(next_M, next_M);
}

void LQControlOptimizer :: write_obj(double obj, int itr, ofstream& file){
	file << itr << " " << obj << endl;
}
void LQContinuousPredictor::setPrior(vector<vector<double> >&
			_prior){
	prior = _prior;
}

void LQContinuousPredictor::predictAll(vector<pair<double,double> >& 
			observation,int prev, int current, int start){
	cout<<"Map prediction"<<endl;
	if(prev-start==0 && current-start==0){
		for(int x=0;x<dims.first;x++){
			for(int y=0;y<dims.second;y++){
				gridLikelihoods.at(x).at(y)=
					-log(dims.first*dims.second);
			}
		}
	}else{
		lqEngine.forwardInference(observation,prev,current,
				continuousLikelihoods, start);
		for(map<pair<int,int>,pair<double,double> >::iterator m=
					mapping.begin();m!=mapping.end();m++){
			gridLikelihoods.at(m->first.first).at(m->first.second) = 
				continuousLikelihoods[m->second];
		}
	}
	cout<<"Generate posterior"<<endl;
	double sum = -HUGE_VAL;
	for(int x=0;x<dims.first;x++){
	  for(int y=0;y<dims.second;y++){
		  posterior.at(x).at(y) = prior[x][y] + gridLikelihoods[x][y];
		  sum = LogAdd(sum,posterior[x][y]);
	  }
	}
	for(int x=0;x<dims.first;x++)
	  for(int y=0;y<dims.second;y++)
		posterior.at(x).at(y) -= sum;
}

double LQContinuousPredictor::predictPoint(vector<pair<double,double> >&
			observation,pair<double,double>& target,
			int prev, int current){
	cout<<"Point prediction"<<endl;
	if (prev==0&&current==0){
		return -log(dims.first*dims.second);
	}
	return lqEngine.forwardInference(
				observation,prev,current,target);
}

#if 0
void LQContinuousPredictor::test(vector<pair<double,double> >& obs){
	int prev = 0;
	int current = obs.size();
	lqEngine.forwardInference(obs,prev,current,
				continuousLikelihoods);
	vector<VectorXd> firstStates;
	vector<VectorXd> grdtthStates;
	vector<VectorXd> checkStates;
	lqEngine.cState.convertState(firstStates,obs,
				continuousLikelihoods.begin()->first);
	
	for(map<pair<double,double>,double>::iterator m=
		continuousLikelihoods.begin();m!=continuousLikelihoods.end();m++){
		checkStates.clear();
		grdtthStates.clear();
		lqEngine.cState.convertState(grdtthStates,obs,m->first);
		cout<<"x: "<<m->first.first<<" y: "<<m->first.second<<endl;
		lqEngine.cState.convertState(grdtthStates,obs,m->first);
		lqEngine.checkStateTransform(firstStates,checkStates,
					continuousLikelihoods.begin()->first,
					m->first);
		for(int i=0;i<grdtthStates.size();i++){
//			cout<<"Ground state: "<<endl<<grdtthStates[i]<<endl
//				<<"Transformed state: "<<endl<<checkStates[i]<<endl;
			int diff = 0;
			for(int j=0;j<6;j++){
				diff+=(int)((grdtthStates.at(i)(j)-checkStates.at(i)(j))
					*100000);
			}
			assert(diff==0);
		}

		double groundTruth = lqEngine.sequenceLikelihood(grdtthStates);
		double function = m->second;

//		cout<<"groundtruth: "<<groundTruth<<" forward function: "
//			<<function<<endl;
        assert((int)(groundTruth*10)==(int)(function*10));
	}
	
}
#endif

void LQContinuousPredictor::test(vector<pair<double,double> >& obs){
	int prev = 0;
	int current = 5; 
	for(;current<=obs.size();current+=5){
		cout<<"Current: "<<current<<endl;
		vector<VectorXd> grdtthStates;
		vector<pair<double,double> > partialObs;
		partialObs.insert(partialObs.begin(),obs.begin(),
					obs.begin()+current);

		lqEngine.forwardInference(obs,prev,current,
				continuousLikelihoods);
		prev=current;
	
		for(map<pair<double,double>,double>::iterator m=
			continuousLikelihoods.begin();m!=continuousLikelihoods.end();m++){
			grdtthStates.clear();
			lqEngine.cState.convertState(grdtthStates,partialObs,m->first);
			double groundTruth = lqEngine.sequenceLikelihood(grdtthStates);
			double function = m->second;

			if((int)(groundTruth*10)!=(int)(function*10)){
				cout<<"x: "<<m->first.first<<" y: "<<m->first.second<<endl;
				cout<<"groundtruth: "<<groundTruth<<" forward function: "
					<<function<<endl;
			}

			assert((int)(groundTruth*10)==(int)(function*10));
		}
	}
}


void LQContinuousPredictor::testAfterPredict(vector<pair<double,double> >& obs,
		int current){
	if(current==0)
	  return;

	cout<<"Testing... "<<endl;
	vector<VectorXd> grdtthStates;
	vector<pair<double,double> > partialObs;
    partialObs.insert(partialObs.begin(),obs.begin(),
					obs.begin()+current);

	for(int x=0;x<dims.first;x++){
		for(int y=0;y<dims.second;y++){
			pair<double,double> target = 
				grid.grid2Real(x,y);
			grdtthStates.clear();
			lqEngine.cState.convertState(grdtthStates,partialObs,target);
			
			double groundTruth = lqEngine.sequenceLikelihood(grdtthStates);
			double function = gridLikelihoods.at(x).at(y);
	
			if((int)(groundTruth*10)!=(int)(function*10)){
				cout<<"x: "<<x<<" y: "<<y<<endl;
			    cout<<"groundtruth: "<<groundTruth<<" forward function: "
				    <<function<<endl;
			}
			assert((int)(groundTruth*10)==(int)(function*10));
		}
	}
	
}
