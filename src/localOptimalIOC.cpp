#include "localOptimalIOC.h"
Tensor_v* Likelihood :: g_tilde(){
    int T = actions.size();// # of states
    
    VectorXd tmp_v(2);
    Tensor_v* g_til = new Tensor_v(T,1,2); // bottom 2X1 vector
    MatrixXd tmp_B_TP(2,6);

    for(int i = 0; i < T-1; i++){
        tmp_B_TP = B.transpose();
        tmp_v = 2*tmp_B_TP*M*B*actions.at(i) + 2*tmp_B_TP*M*A*states.at(i);
        if (!g_til->set_v(i,0,tmp_v)){
            cout << "error in add bottom vectors into g_tilde" << endl;
        }
    } 
    tmp_v(0) = 0;
    tmp_v(1) = 0;
    if (!g_til->set_v(T-1,0,tmp_v)){
        cout << "error in add bottom vectors into g_tilde" << endl;
    }
    return g_til;
}  

Tensor_v* Likelihood :: g_hat(){
    int T = actions.size();// # of states
    
    VectorXd tmp_v(6);
    Tensor_v* ghat = new Tensor_v(T,1,6);//bottom 6X1 vector
    
    MatrixXd tmp_F(f_cost->get_Num(),6);
    MatrixXd tmp_F_TP(6,f_cost->get_Num());
    for(int i = 0; i < T; i++){
        tmp_F = f_cost->F_first_order(states.at(i));
        tmp_F_TP = tmp_F.transpose();
        tmp_v = 2*M*states.at(i) + tmp_F_TP*Theta;
        if(!ghat->set_v(i,0,tmp_v)){
            cout << "error in adding bottom vectors into g_hat" << endl;
        }
    }
    return ghat; 
}

void Likelihood :: get_g(){
    int T = actions.size();
    Tensor_v* g_h = g_hat();

    Tensor_v* g_til = g_tilde();
    
    Tensor_v* tmp_Jg = new Tensor_v(T,1,2);
    if(!J->multiply(g_h,tmp_Jg)){
        cout << "error in multiplying J with g_hat" << endl;
    }
    if(!tmp_Jg->plus(g_til,g)){// pass the value to g
        cout << "error in adding g_tilde with Jg" << endl;
    }
    delete g_h;
    delete g_til;
    delete tmp_Jg;
}

Tensor_m* Likelihood :: H_tilde(){
    int T = actions.size();
    Tensor_m* H_til = new Tensor_m(T,T,2,2);
    MatrixXd tmp_m(2,2);
    MatrixXd B_TP(2,6);

    B_TP = B.transpose();
    tmp_m = 2*B_TP*M*B;
    for(int i = 0; i < T-1; i++){
        if(!H_til->set_m(i,i,tmp_m)){
            cout << "error in setting 2BTMB in H_tilde"<<endl;
        }
    }
    tmp_m.setZero();
    if(!H_til->set_m(T-1,T-1,tmp_m)){
            cout << "error in setting 2BTMB in H_tilde"<<endl;
        }
    return H_til;
}

Tensor_m* Likelihood :: H_hat(){
    int T = actions.size();
    
    Tensor_m* H_h = new Tensor_m(T,T,6,6);
    MatrixXd tmp_m(6,6);
    Tensor_v* F_two;
    
    
    for(int i = 0; i < T; i++){
       F_two = f_cost->F_second_order(states.at(i)); 
       
       if(!F_two->collapse(Theta,tmp_m)){
           cout << "errors in collapse F/si2 with Theta" << endl;
       }
       tmp_m = tmp_m + 2*M;
       if(!H_h->set_m(i,i,tmp_m)){
           cout << "errors in setting bottom in h_hat" << endl;
       }
    }
    delete F_two;
    return H_h;
}

void Likelihood :: get_H(){
    int T = actions.size();
    Tensor_m* Hhat = H_hat();
    Tensor_m* Htilde = H_tilde();
    Tensor_m* tmp_JH = new Tensor_m(T,T,2,6);
    Tensor_m* tmp_JHJ = new Tensor_m(T,T,2,2);
    Tensor_m* J_T = J->transpose();

    if(!J->multiply(Hhat,tmp_JH)){
        cout << "error in multiplying JH" << endl;
    }

    if(!tmp_JH->multiply(J_T,tmp_JHJ)){
        cout << "error in computing JHJ" << endl;
    }
    if(!tmp_JHJ->plus(Htilde,H)){
        cout << "error in adding H " << endl;
    }
    delete tmp_JH;
    delete tmp_JHJ;
    delete Hhat;
    delete Htilde;
    delete J_T;
}

void Likelihood :: setStates(vector<pair<double, double> >& rawData,
			pair<double, double>& target){
	/** Reset states and actions **/
	states.clear();
	actions.clear();

	/** Transform the states **/
	cs->convertState(states, rawData, target);
    extract_actions();
    
	/** Transform the obstacles **/
	f_cost->transform(rawData.front(), target);
}

void Likelihood :: extract_actions(){
	if(states.size() == 0){
		cerr << "States is not initialized! " << endl;
		return;
	}
	for(vector<VectorXd>::iterator it = states.begin();
				it!=states.end()-1; ++it){
		VectorXd a(2);
		a = (it+1)->segment(2,2);
		actions.push_back(a);
	}
}

double Likelihood :: get_likelihood(vector<pair<double, double> >& rawData,
			pair<double, double>& target){
    
	/** Reset states and actions **/
	states.clear();
	actions.clear();

	/** Transform the states **/
	cs->convertState(states, rawData, target);
    extract_actions();
    
	/** Transform the obstacles **/
	f_cost->transform(rawData.front(), target);
    
	/** Initialize g, H, J */
	int T = actions.size();
	compute_J();
	g = new Tensor_v(T,1,2);
    
	H = new Tensor_m(T, T, 2, 2);
    
    get_g();
    
    get_H();
    
    MatrixXd g_exp(2*T,1);
    MatrixXd g_exp_T(1,2*T);
    MatrixXd H_exp(2*T,2*T);
    MatrixXd H_exp_tmp(2*T,2*T);
    double H_deter;
    g_exp = g->expand();
    H_exp = H->expand();
    H_exp = -1*H_exp;
    H_exp_tmp = -1*H_exp;
    H_deter = H_exp_tmp.determinant();
    //H_deter = fabs(H_deter);
    //cout<<"H_deter"<<H_deter<<endl;
    H_exp_tmp = H_exp.inverse();
    MatrixXd L(1,1);
    double l;
    g_exp_T = g_exp.transpose();
    
    //cout<<"H_exp"<<H_exp<<endl;
    //cout<<"g_exp_T*H_exp_tmp"<<g_exp_T*H_exp_tmp<<endl;
    cout<<"H_deter"<<H_deter<<endl;
    
    L = g_exp_T*H_exp_tmp*g_exp;
    cout<<"L"<<L<<endl;
    l = 0.5*L(0,0) + 0.5*log(H_deter)-log(2*M_PI);
    return l;
}

Tensor_m* Likelihood :: get_g_wrt_M(){
    // Compute g_tilde wrt M 2TX1(6X6)
    int T = actions.size();
    Tensor_m* g_tilde_M = new Tensor_m(2*T,1,6,6);
    MatrixXd tmp_M(6,6);
    VectorXd St(6,1);
    VectorXd At(2,1);
    VectorXd AS(6,1);
    for(int t = 0; t < T; t++){
        St = states.at(t);
        At = actions.at(t);//get the current s and a
        for(int k = 0; k < 2; k++){
            AS = A*St;
            if(k==0){
                for(int i = 0; i < 6;i++){
                    for(int j = 0; j < 6; j++){
                        tmp_M(i,j) = 2*B(i,0)*B(j,0)*At(0) + 2*B(i,0)*B(j,1)*At(1) + 2*B(i,0)*AS(j);
                    }    
                }
                if(!g_tilde_M->set_m(2*t,0,tmp_M)){
                    cout << "error in setting g_tilde_M" << endl;
                }
            }else{
                for(int i = 0; i < 6;i++){
                    for(int j = 0; j < 6;j++){
                        tmp_M(i,j) = 2*B(i,1)*B(j,0)*At(0) + 2*B(i,1)*(j,1)*At(1) + 2*B(i,1)*AS(j);
                    }
                }
                if(!g_tilde_M->set_m(2*t+1,0,tmp_M)){
                    cout << "error in setting g_tilde_M" << endl;
                }           
            }
        }
    }
    Tensor_m* g_hat_M = new Tensor_m(6*T,1,6,6);
    
    for(int t=0; t < T; t++){
        St = states.at(t);
        for(int k = 0; k < 6; k++){
            tmp_M.setZero();
            tmp_M.row(k) << St.transpose();
            tmp_M = tmp_M*2;
            if(!g_hat_M->set_m(6*t+k,0,tmp_M)){
                cout << "error in setting g_hat_M" << endl;
            }
        }
    }
    //put all together
    Tensor_m* Jg_hat_M = new Tensor_m(2*T,1,6,6);
    if(!J->collapse_multiply(g_hat_M,Jg_hat_M)){
        cout << "error in deyang's J inner product" << endl;
    }
    //deyang's method

    Tensor_m* g_wrt_M = new Tensor_m(2*T,1,6,6);
    if(!Jg_hat_M->plus(g_tilde_M,g_wrt_M)){
        cout << "error in adding g_tilde with Jghat" << endl;
    }
    delete Jg_hat_M;
    delete g_tilde_M;
    delete g_hat_M;
    return g_wrt_M;
}

Tensor_m* Likelihood :: get_H_wrt_M(){
    int T = actions.size();
    //compute H_tilde wrt M TXT(12X12)
    Tensor_m* H_tilde_M = new Tensor_m(T,T,12,12);
    MatrixXd tmp_M(12,12);
    VectorXd B1(6);
    VectorXd B2(6);
    
    B1 = B.col(0);
    B2 = B.col(1);
    tmp_M.topLeftCorner(6,6) = B1*B1.transpose();
    tmp_M.topRightCorner(6,6) = B2*B1.transpose();
    tmp_M.bottomLeftCorner(6,6) = B1*B2.transpose();
    tmp_M.bottomRightCorner(6,6) = B2*B2.transpose();
    tmp_M = 2*tmp_M;
    for(int i = 0; i < T-1; i++){
        if(!H_tilde_M->set_m(i,i,tmp_M)){
            cout<<"error in setting H_tilde_M"<<endl;
        }
    }
    tmp_M.setZero();
    if(!H_tilde_M->set_m(T-1,T-1,tmp_M)){
        cout<<"error in setting H_tilde_M"<<endl;
    }
    //compute JHhatJ wrt M TxT(12X12)
    Tensor_m* J_Hhat_J_M = new Tensor_m(T,T,12,12);
    if(!J->vector_inner_product(J_Hhat_J_M)){
        cout << "Deyangzhao vector inner product"<<endl;
    }
    Tensor_m* H_wrt_M = new Tensor_m(T,T,12,12);
    if(!J_Hhat_J_M->plus(H_tilde_M,H_wrt_M)){
        cout << "error in plus h_tilde with JHhatJ"<<endl;
    }
    delete H_tilde_M;
    delete J_Hhat_J_M;
    return H_wrt_M;
}

MatrixXd Likelihood :: get_g_wrt_theta(){
    int T = actions.size();
    int m = f_cost->get_Num();
    Tensor_m* g_hat_theta = new Tensor_m(T,1,6,m);
    MatrixXd tmp_F(m,6);
    for(int i = 0; i < T; i++){
        tmp_F = f_cost->F_first_order(states.at(i));
        
        if(!g_hat_theta->set_m(i,0,tmp_F.transpose())){
            cout<<"error in setting g_hat_theta"<<endl;
        }
    }
    MatrixXd J_expand(2*T,6*T);
    MatrixXd g_hat_expand(6*T,m);
    J_expand = J->expand();
    g_hat_expand = g_hat_theta->expand();
    MatrixXd g_wrt_theta(2*T,m);
    g_wrt_theta = J_expand*g_hat_expand;
    delete g_hat_theta;
    return g_wrt_theta;
}

Tensor_m* Likelihood :: get_H_wrt_theta(){
    int T = actions.size();
    int m = f_cost->get_Num();
    //compute H_hat wrt theta 6Tx6T(mx1)
    Tensor_v* H_hat_theta = new Tensor_v(6*T,6*T,m);
    Tensor_v* H_ii = new Tensor_v(6,6,m);
    VectorXd tmp_v(m);
    for(int t = 0; t < T; t++){
        H_ii = f_cost->F_second_order(states.at(t));
        for(int i = 6*t; i < 6*t+6;i++){
            for(int j = 6*t; j < 6*t+6;j++){
               if(!H_ii->get_v(i-6*t,j-6*t,tmp_v)){
                   cout << "error in getting tmp_v"<< endl;
               }
               if(!H_hat_theta->set_v(i,j,tmp_v)){
                   cout << "error in setting tmp_v"<< endl;
               } 
            }
        }
    }
    Tensor_m* tmp_Jj = new Tensor_m(1,T,2,6);
    Tensor_m* tmp_Ji = new Tensor_m(1,T,2,6);
    Tensor_m* tmp_Ji_T;
    Tensor_v* tmp_JH = new Tensor_v(2,6*T,m);
    MatrixXd tmp_JH_expand(2*m,6*T);
    MatrixXd tmp_JiT_expand(6*T,2);
    Tensor_m* tmp_H_wrt_theta = new Tensor_m(T,T,2*m,2);
    MatrixXd tmp_theta(2*m,2);
    for(int i = 0; i < T; i++){
        for(int j = 0; j < T; j++){
           if(!J->get_row(j,tmp_Jj)){
               cout << "error in getting jth row of J"<<endl;
           }
           if(!J->get_row(i,tmp_Ji)){
               cout << "error in getting ith row of J"<<endl; 
           }
           tmp_Ji_T = tmp_Ji->transpose();
           if(!tmp_Jj->collapse_multiply(H_hat_theta,tmp_JH)){
               cout << "error in Jj* H_hat/theta"<<endl;
           }
           tmp_JH_expand = tmp_JH->expand();//2mx6T
           tmp_JiT_expand = tmp_Ji_T->expand();//6Tx2
           if(!tmp_H_wrt_theta->set_m(i,j,tmp_theta)){
               cout << "error in setting H_wrt_theta"<<endl;
           }
        }
    }
    Tensor_m* H_wrt_theta = new Tensor_m(2*T,2*T,m,1);
    if(!tmp_H_wrt_theta->reshape(H_wrt_theta)){
        cout << "error in reshape H_wrt_theta" << endl;
    }
    delete H_hat_theta;
    delete H_ii;
    delete tmp_Jj;
    delete tmp_Ji;
    delete tmp_Ji_T;
    delete tmp_JH;
    delete tmp_H_wrt_theta;
    return H_wrt_theta;//2Tx2T(mx1)
}

MatrixXd Likelihood :: get_L_wrt_M(){
    MatrixXd L_wrt_M(6,6);
    int T = actions.size();
    MatrixXd g_expand(2*T,1);
    MatrixXd H_expand(2*T,2*T);
    MatrixXd h(2*T,1);
    MatrixXd h_T(1,2*T);
    g_expand = g->expand();
    H_expand = H->expand();
    H_expand = H_expand.inverse();//now it's inverse
    h = H_expand*g_expand;//2Tx1
    
    h_T = h.transpose();//1X2T
    Tensor_m* g_wrt_M = get_g_wrt_M();//2Tx1(6x6)
    Tensor_m* H_wrt_M = get_H_wrt_M();//TxT(12x12)
    Tensor_m* H_wrt_M_reshape = new Tensor_m(2*T,2*T,6,6);
    if(!H_wrt_M->reshape(H_wrt_M_reshape)){
        cout<<"error in reshaping H_wrt_M"<<endl;
    }
    Tensor_m* hT_gM = new Tensor_m(1,1,6,6);
    if(!g_wrt_M->left_multiply(h_T,hT_gM)){
        cout<<"error in left multiplying hT with g_M"<<endl;
    }
    L_wrt_M = hT_gM->get_m(0,0);
    
    Tensor_m* hT_HM = new Tensor_m(1,2*T,6,6);
    if(!H_wrt_M_reshape->left_multiply(h_T,hT_HM)){
        cout<<"error in left multiplying hT with H_M"<<endl;
    }
    Tensor_m* hT_HM_h = new Tensor_m(1,1,6,6);
    if(!hT_HM->right_multiply(h,hT_HM_h)){
        cout<<"error in right multiplying hT_HM with h"<<endl;
    }
    L_wrt_M = L_wrt_M - 0.5*hT_HM_h->get_m(0,0);

    Tensor_m* Hi_HM = new Tensor_m(2*T,2*T,6,6);
    if(!H_wrt_M_reshape->left_multiply(H_expand,Hi_HM)){
        cout<<"error in left multiplying H_inverse with H_wrt_M"<<endl;
    }
    L_wrt_M = L_wrt_M + 0.5*Hi_HM->trace();
    delete hT_HM_h;
    delete hT_HM;
    delete hT_gM;
    delete H_wrt_M_reshape;
    delete g_wrt_M;
    delete H_wrt_M;
    delete Hi_HM;
    return L_wrt_M;
}

VectorXd Likelihood :: get_L_wrt_theta(){
    int m = f_cost->get_Num();
    VectorXd L_wrt_theta(m);
    int T = actions.size();
    MatrixXd g_expand(2*T,1);
    MatrixXd H_expand(2*T,2*T);
    MatrixXd h(2*T,1);
    MatrixXd h_T(1,2*T);
    g_expand = g->expand();
    H_expand = H->expand();
    H_expand = H_expand.inverse();//now it's inverse
    h = H_expand*g_expand;//2Tx1
    h_T = h.transpose();//1X2T

    MatrixXd g_wrt_theta = get_g_wrt_theta();//2Txm

    Tensor_m* H_wrt_theta = get_H_wrt_theta();//2Tx2T(mx1)

    
    L_wrt_theta = (h_T*g_wrt_theta).transpose();//mx1
    
    Tensor_m* hT_Htheta = new Tensor_m(1,2*T,m,1);
    if(!H_wrt_theta->left_multiply(h_T,hT_Htheta)){
        cout<<"error in left multiplying hT with H_theta"<<endl;
    }
    Tensor_m* hT_Htheta_h = new Tensor_m(1,1,m,1);
    if(!hT_Htheta->right_multiply(h,hT_Htheta_h)){
        cout<<"error in right multiplying hT_Htheta with h"<<endl;
    }

    MatrixXd tmp = hT_Htheta_h->get_m(0,0);
    L_wrt_theta = L_wrt_theta - 0.5*tmp.col(0);

    
    Tensor_m* Hi_Htheta = new Tensor_m(2*T,2*T,m,1);
    if(!H_wrt_theta->left_multiply(H_expand,Hi_Htheta)){
        cout<<"error in left multiplying H_inverse with H_wrt_theta"<<endl;
    }
    tmp = Hi_Htheta->trace();
    L_wrt_theta = L_wrt_theta + 0.5*tmp.col(0);
    delete Hi_Htheta;
    delete hT_Htheta;
    delete hT_Htheta_h;
    delete H_wrt_theta;
    return L_wrt_theta;
}

void Likelihood :: compute_J(){
    int T = actions.size();
    J = new Tensor_m(T,T,2,6);
    MatrixXd AT = A.transpose();
    MatrixXd BT = B.transpose();
    for(int i = 0; i < T; i++){
        for(int j = 0; j < T; j++){
            if(i == j+1){
                if(!J->set_m(i,j,BT)){
                    cout<<"error in set J when i == j+1"<<endl;
                }
            }else if(i > j+1){
                if(!J->set_m(i,j,J->get_m(i-1,j)*AT)){
                    cout<<"error in set J when i > j+1"<<endl;
                }
            }else{
                
            }
        }
    }
    
}


void LocalEOptimizer :: gradientDescent(double stepsize, double eps, int max_iter){
    int N = evid.size();//# paths
    double sum_gradient = 100;

    MatrixXd tmp_M;// current M
    MatrixXd tmp_M_g;//current gradient of M
    VectorXd tmp_theta;// current theta
    VectorXd tmp_theta_g;//current gradient of theta

    int iter = 1;

    double objective_value=1;
    double old_objective_value=0;
    //while(sum_gradient > eps && iter < max_iter && old_objective_value < objective_value){
    while(sum_gradient > eps && iter < max_iter){
        old_objective_value = objective_value;
        objective_value = 0;
        for(int i = 0; i < N; i++){
        //optimize M first
        double oneL = L->get_likelihood(evid.at_raw(i), evid.at_raw(i).back());
		cout << " One path likelihood: " << i << " " << oneL << endl;
        objective_value += oneL;
        tmp_M = L->getM();
        tmp_M_g = L->get_L_wrt_M();
        
        tmp_M = tmp_M + stepsize*tmp_M_g;
        L->setM(tmp_M);
        //optimize theta

        tmp_theta = L->getTheta();
        tmp_theta_g = L->get_L_wrt_theta();
        cout<<"get L wrt M"<<endl;
        tmp_theta = tmp_theta + stepsize*tmp_theta_g;
        L->setTheta(tmp_theta);
        sum_gradient = tmp_M.norm() + tmp_theta.norm();
        cout << "Iteration:" << iter << " using state:" << i 
            << "  Gradient sum:"<< sum_gradient <<endl;
        cout << "M: "<<endl << L->getM()<<endl;
        cout << "Theta: "<<endl << L->getTheta().transpose()<<endl;
        }
        cout<< "Objective Value: " << objective_value << endl;
        iter ++;
        stepsize = stepsize/(1+log(iter));
    }
	cout << "M: " << L->getM() << endl;
	cout << "Theta:  " << L->getTheta() << endl;
}

void F_COST :: transform(pair<double, double>& origin, 
					pair<double, double>& target){
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

			transformed_u.clear();
			for(int i = 0; i < raw_obstacles.size(); i++){
				VectorXd xy(2);
				xy << raw_obstacles.at(i).first, raw_obstacles.at(i).second;
				xy = rot*xy+trans;
				transformed_u.push_back(xy);
			}
			//cout << "Transform the obstacles in the coordinates define by Origin: "
               //  << origin.first << " " << origin.second 
				// << " Target : " << target.first << " " << target.second << endl;
} 

void LocalIOCPredictor :: predictAll(vector<pair<double,double> >& 
			observation, int index){
	cout<<"Map prediction"<<endl;
	if(index == 0){
		cout << "fast " << endl;
		for(int x=0;x<dims.first;x++){
			for(int y=0;y<dims.second;y++){
				gridLikelihoods.at(x).at(y)=
					-log(dims.first*dims.second);
			}
		}
	}else{
		vector<pair<double,double> > current;
		for(int i = 0; i <= index; i++){
			current.push_back(observation.at(i));
		}
		cout << current.size() << endl;

		for(map<pair<int,int>,pair<double,double> >::iterator m=
					mapping.begin();m!=mapping.end();m++){
//			cout << m->first.first << " " << m->first.second << " " 
//				 << m->second.first << " " << m->second.second 
//				 << endl;
			gridLikelihoods.at(m->first.first).at(m->first.second) = 
				L->get_likelihood(current, m->second);
		cout << gridLikelihoods.at(m->first.first).at(m->first.second) << endl;
			
		}
	}

	cout<<"Generate posterior"<<endl;
	double sum = -HUGE_VAL;
	for(int x = 0; x < dims.first; x++){
	  for(int y = 0; y < dims.second; y++){
		  posterior.at(x).at(y) = prior[x][y] + gridLikelihoods[x][y];
		  sum = LogAdd(sum,posterior[x][y]);
	  }
	}

	for(int x = 0; x < dims.first; x++)
	  for(int y = 0; y < dims.second; y++)
		posterior.at(x).at(y) -= sum;
}
