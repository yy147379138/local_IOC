#include "options.h"
#include "visualize.h"
#include "localOptimalIOC.h"
#include <math.h>


int main(int argc, char **argv) {
   OptionParser opts;

   string mapFile, evidFile;
   double step, eps;
   int max_iter, write;
   opts.addOption(new StringOption("map", 
            "--map <filename>                 : map file",
            "../input/grid.bmp", mapFile, false));

   opts.addOption(new StringOption("evidence", 
            "--evidence <filename>            : evidence file",
            "", evidFile, true));
   
   opts.addOption(new DoubleOption("step", 
            "--step <double>                  : learning step size",
            1.0, step, true));

   opts.addOption(new IntOption("itrs", 
            "--itrs <int>                   : iteration times",
            1000, max_iter, true));
	   
   opts.addOption(new DoubleOption("eps", 
            "--eps <double>                  : stop threshold",
            0.0001, eps, true));

   opts.addOption(new IntOption("write", 
            "--write <int>                   : write signal",
            0, write, true));

   double factor = 1.0;

   opts.parse(argc,argv);

   cout << "Loading Map File"<<endl;
   BMPFile bmpFile(mapFile); 
   Grid grid(bmpFile, black);
   cout << "Loading Evidence"<<endl;
   Evidence trainSet(evidFile, grid, factor);
  
   /****** INITIALIZE PARAMETERS *******************/
   M_6 A;
   A.setZero();
   A(0,0) = 1;
   A(1,1) = 1;
   A(4,2) = -1;
   A(5,3) = -1;
   M_6_2 B;
   B<<1,0,
	  0,1,
	  1,0,
	  0,1,
	  1,0,
	  0,1;
   MatrixXd M(6,6);
   VectorXd theta(2);
   theta<< -1, -1;

   ifstream infile("../params/local.dat");
   for(int row=0;row<M.rows();row++){
	   for(int col=0;col<M.cols();col++){
		   double temp;
		   infile>>temp;
		   M(row,col) = temp;
	   }
   }
   infile.close();
   M_6 sigma;
   sigma<<0.001,0,0,0,0,0,
	      0,0.001,0,0,0,0,
		  0,0,0.005,0,0,0,
		  0,0,0,0.005,0,0,
		  0,0,0,0,0.005,0,
		  0,0,0,0,0,0.005;

   vector<MatrixXd> covs;
   MatrixXd v1(2,2);
   v1.setIdentity();
   MatrixXd v2(2,2);
   v2.setIdentity();
   v2 = 10.0 * v2;
   covs.push_back(v1);
   covs.push_back(v2);
   /********************************************************************/
   ContinuousState* stateConvertor = new ContinuousState();
   F_COST* fcost = new F_COST(&grid, covs);
   Likelihood* L = new Likelihood(fcost, stateConvertor, A, B, sigma);
   //set all parameters
   L->setM(M);
   L->setSigma(sigma);
   L->setTheta(theta);
   LocalEOptimizer* LEO = new LocalEOptimizer(L, trainSet);
   LEO->gradientDescent(step, eps, max_iter);//optimize it
   /*
   cout<<"Convertor "<<endl;
   ContinuousState stateConvertor;
   cout<<"Inference Engine "<<endl;
   LQControlInference inferEngine(A,B,sigma,M,stateConvertor);
   cout<<"Optimizer  "<<endl;
   LQControlOptimizer optimizer(inferEngine,stateConvertor,trainSet);
#if 1 
   cout<<"Optimization norm: "<<optimizer.optimize_v(eta,times,thr)<<endl;
  // cout<<"Optimization norm: "<<optimizer.optimize_v(eta,times,thr,LQControlOptimizer::BATCH_EXP)<<endl;
#else
   double alpha = 0.001;
   double beta = 0.8;
   double mini_step = 1e-8;
   optimizer.backtrack(times, thr, alpha, beta, mini_step);
#endif
*/

if(write){
   ofstream outfile("../params/learn.dat");
   outfile << L->getM() <<endl;
   outfile.close();
}


   return 0;

}
