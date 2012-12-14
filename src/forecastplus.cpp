#include "main.h"
#include "options.h"
#include "evidence.h"
#include "features.h"
#include "intent.h"

#ifndef NUMROBEFAT
#define NUMROBFEAT 0
#endif

#ifndef VEL_DIM
#define VEL_DIM 2
#endif 

#ifndef NUMPOSFEAT
#define NUMPOSFEAT 6
#endif

int main(int argc, char **argv) {
   OptionParser opts;

   string mapFile,trainFile,testFile;

   int factor = 1;
   double step;

   opts.addOption(new StringOption("map", 
            "--map <filename>                 : map file",
            "../input/grid.bmp", mapFile, false));
   opts.addOption(new StringOption("evidence", 
            "--test evidence <filename>            : evidence file",
            "", testFile, true));

   opts.addOption(new DoubleOption("step",
            "--step <double>                   : inference interval",
            1.0, step, true));

   opts.parse(argc,argv);

   JetColorMap jet;
   RGBTRIPLE black = {0,0,0};
   RGBTRIPLE white = {255,255,255};
   RGBTRIPLE red;
   red.R = 255;
   red.G = 0;
   red.B = 0;
   RGBTRIPLE blue;
   blue.R = 0;
   blue.G = 0;
   blue.B = 255;
   RGBTRIPLE green;
   green.R = 0;
   green.G = 255;
   green.B = 0; 
   RGBTRIPLE initialColor;
   initialColor.R = 111; 
   initialColor.G = 49;
   initialColor.B = 152;
   RGBTRIPLE currentColor;
   currentColor.R = 181;
   currentColor.G = 165;
   currentColor.B = 213;
   RGBTRIPLE magenta;
   magenta.R = 255;
   magenta.G = 0;
   magenta.B = 255;
   RGBTRIPLE cyan;
   cyan.R = 0;
   cyan.G = 255;
   cyan.B = 255;
   RGBTRIPLE yellow;
   yellow.R = 255;
   yellow.G = 255;
   yellow.B = 0;

   BMPFile bmpFile(mapFile);
   Grid grid(bmpFile, black);

   
   Evidence testSet(testFile, grid, factor);
 //  Evidence trainSet(trainFile, grid, factor);

   pair<int, int> dims = grid.dims();
   
   cout << " Speed Feature"<<endl;
   vector<double> speedTable(VEL_DIM,0.0);
   speedTable.at(1) = 0.75;
   DisVecSeqFeature speedfeat(speedTable);

   vector<int> dimensions;
   dimensions.push_back(dims.first);
   dimensions.push_back(dims.second);
   dimensions.push_back(VEL_DIM);
   
   /* ****************************************
	*      INITIALIZE MARKOV DECESION PROCESS 
	*      BASED MODEL PARAMETERS
	* ****************************************/
   vector<double> p_weights(NUMPOSFEAT,-0.0);
   p_weights.at(0) = -2.23; //-2.23 for PPP forecast
   p_weights.at(1) = -6.2;
   p_weights.at(2) = -0.35;
   p_weights.at(3) = -2.73;
   p_weights.at(4) = -0.92;
   p_weights.at(5) = -0.26;
   vector<double> r_PosWeights(NUMPOSFEAT+NUMROBFEAT, -0.0);
   r_PosWeights.at(0) = -3.83;
   r_PosWeights.at(1) = -8.36;
   r_PosWeights.at(2) = -2.65;
   r_PosWeights.at(3) = -5.43;
   r_PosWeights.at(4) = -3.15;
   r_PosWeights.at(5) = -3.30;
   //r_PosWeights.at(6) =  0.60;
   //r_PosWeights.at(7) =  0.45;
   vector<double> nr_PosWeights(NUMPOSFEAT+NUMROBFEAT, -0.0);
   nr_PosWeights.at(0) = -4.51;
   nr_PosWeights.at(1) = -6.2;
   nr_PosWeights.at(2) = -0.35;
   nr_PosWeights.at(3) = -2.73;
   nr_PosWeights.at(4) = -0.93;
   nr_PosWeights.at(5) = -0.28;
   //nr_PosWeights.at(6) = -0.50;
   //nr_PosWeights.at(7) = -0.286;
   vector<double> r_SeqWeights(VEL_DIM, -0.0);
   r_SeqWeights.at(0) = 0.59;
   r_SeqWeights.at(1) = -0.83;
   vector<double> nr_SeqWeights(VEL_DIM, -0.0);
   nr_SeqWeights.at(0) = -1.21;
   nr_SeqWeights.at(1) = 0.49;

   Parameters p(p_weights);
   Parameters r_Pos(r_PosWeights);
   Parameters nr_Pos(nr_PosWeights);
   Parameters r_Seq(r_SeqWeights);
   Parameters nr_Seq(nr_SeqWeights);

   /* ****************************************
	*      INITIALIZE LINEAR QUADRATIC CONTROL 
	*      BASED MODEL PARAMETERS
	* ****************************************/
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
   M_6 costM;
   ifstream infile("../params/nonrob2000.dat");
   for(int row=0;row<costM.rows();row++){
	   for(int col=0;col<costM.cols();col++){
		   double temp;
		   infile>>temp;
		   costM(row,col) = temp;
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


   /* ****************************************
	*      DECLARATION OF INFERENCE ENGINES    
	* ****************************************/
   OrderedWaveInferenceEngine pp(InferenceEngine::GRID8);
   DisSeqOrderInferEngine mdpr(InferenceEngine::GRID8);
   DisSeqOrderInferEngine mdpnr(InferenceEngine::GRID8);
   ContinuousState cState;
   LQControlInference lq(A,B,sigma,costM,cState);
   lq.valueInference();


   IntentRecognizer IR(grid,p,r_Pos,r_Seq,nr_Pos,nr_Seq,
			   speedfeat,pp,mdpr,mdpnr,lq);

   cout << testSet.size() <<" Examples"<<endl;

   for (int i=0; i < testSet.size(); i++) {

      vector<pair<int, int> > & traj = testSet.at(i);
	  vector<double> & vels = testSet.at_v(i);
      vector<double> times = testSet.getTimes(i); 
	  pair<int,int> & botinGrid = testSet.at_bot(i);
	  vector<pair<double,double> > & obs = 
		  testSet.at_raw(i);
      vector<double> & rawTimes = testSet.at_rawTime(i);

      IR.combineForecast(traj,vels,obs,times,rawTimes,
				  botinGrid,i,step);
      
   }
}


