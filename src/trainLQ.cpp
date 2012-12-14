#include "options.h"
#include "visualize.h"
#include "linearquadratic.h"
#include <math.h>


int main(int argc, char **argv) {
   OptionParser opts;

   string mapFile, evidFile;
   double eta,thr;
   int times,write;
   opts.addOption(new StringOption("map", 
            "--map <filename>                 : map file",
            "../input/grid.bmp", mapFile, false));

   opts.addOption(new StringOption("evidence", 
            "--evidence <filename>            : evidence file",
            "", evidFile, true));
   
   opts.addOption(new DoubleOption("rate", 
            "--rate <double>                  : learning rate",
            0.0001, eta, true));

   opts.addOption(new IntOption("times", 
            "--times <int>                   : iteration times",
            10000, times, true));
	   
   opts.addOption(new DoubleOption("thresh", 
            "--thresh <double>                  : stop threshold",
            0.0001, thr, true));

   opts.addOption(new IntOption("write", 
            "--write <int>                   : write signal",
            0, write, true));

   double factor = 1.0;

   opts.parse(argc,argv);

   cout << "Loading Map File"<<endl;
   BMPFile bmpFile(mapFile); 
   Grid grid(bmpFile, black);
   cout << "Loading Evidence"<<endl;
   Evidence evid(evidFile,grid,factor);
   Evidence trainSet(grid),testSet(grid);
   evid.split(trainSet,testSet,1.0);
   
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
   M_6 M;

#if 0
   M<< 0.0113537,-0.00980196, 0.0212685, -0.0857003, -0.0214117,-0.0117607,
	  -0.0980196, 0.0286002,    0.13912,  0.0492078,  0.0872365, 0.0852322,
	   0.0212685,   0.13912,    1.67384, -0.293427,   0.512174,  0.0121294,
	  -0.0857003, 0.0492078,  -0.293427,  0.874917, -0.0835453,   0.23488,
	  -0.0214117, 0.0872365,   0.512174, -0nonrob2000.0835453,    1.88756,-0.0312228,
	  -0.0117607, 0.0852322,  0.0121294,  0.23488,  -0.0312228,   1.60772;
#endif
#if 0
   M<< 0.0113537,-0.00980196, 0.0212685, -0.0857003, -0.0214117,-0.0117607,
	  -0.0980196, 0.0286002,    0.13912,  0.0492078,  0.0872365, 0.0852322,
	   0.0212685,   0.13912,    1.67384, -0.293427,   0.512174,  0.0121294,
	  -0.0857003, 0.0492078,  -0.293427,  0.874917, -0.0835453,   0.23488,
	  -0.0214117, 0.0872365,   0.512174, -0.0835453,    1.88756,-0.0312228,
	 -0.00284587, -0.013616,-0.00252935,  0.131111,-0.00326836,    1.27364;   

#endif
#if 1
   M.setZero();
   for(int ii=0;ii<6;ii++){
	   M(ii,ii) +=1.0; 
   }
   for(int row=0;row<M.rows();row++){
	   for(int col=0;col<M.cols();col++){
		   M(row,col) += 1.5;
	   }
   }
#endif
#if 0
   ifstream infile("../params/nonrob930.dat");
   for(int row=0;row<M.rows();row++){
	   for(int col=0;col<M.cols();col++){
		   double temp;
		   infile>>temp;
		   M(row,col) = temp;
	   }
   }
   infile.close();
#endif
   M_6 sigma;
   sigma<<0.001,0,0,0,0,0,
	      0,0.001,0,0,0,0,
		  0,0,0.005,0,0,0,
		  0,0,0,0.005,0,0,
		  0,0,0,0,0.005,0,
		  0,0,0,0,0,0.005;

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

if(write){
   ofstream outfile("../params/stochastic.dat");
   outfile << inferEngine.getM() <<endl;
   outfile.close();
}



   return 0;

}
