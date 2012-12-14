#include "options.h"
#include "visualize.h"
#include "linearquadratic.h"
#include <math.h>
#include <queue>

double entropy(vector<vector<double> >& P){
    double H = 0.0;
	for (int x =0;x<P.size();x++)
        for (int y=0;y<P.at(0).size();y++)
		   H-=exp(P.at(x).at(y))*P.at(x).at(y)/log(2);
	return H;
}

double genPosterior(vector<vector<double> >& prior,
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
	cout<<"SUM:"<<sum<<endl;
	return sum;
}





int main(int argc, char **argv) {
   OptionParser opts;

   string mapFile,testFile;

   double interval;

   opts.addOption(new StringOption("map", 
            "--map <filename>                 : map file",
            "../input/grid.bmp", mapFile, false));

   opts.addOption(new StringOption("evidence", 
            "--evidence <filename>            : evidence file",
            "", testFile, true));

   opts.addOption(new DoubleOption("interval",
            "--interval <double>               : prediction internal",
            1.0, interval, true));

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
//   initialColor.G = 152;
//   initialColor.B = 49;


   RGBTRIPLE currentColor;
   currentColor.R = 181;
   currentColor.G = 165;
   currentColor.B = 213;
//   currentColor.G = 213;
//   currentColor.B = 165;

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
   pair<int,int> dims = grid.dims();

   
   Evidence testSet(testFile, grid);

   cout<<"Initializing parameters A,B,Sigma and M"<<endl;
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
   M_6 M_r,M_nr;
   ifstream infile("../params/lbfgs.dat");
   for(int row=0;row<M_r.rows();row++){
	   for(int col=0;col<M_r.cols();col++){
		   double temp;
		   infile>>temp;
		   M_r(row,col) = temp;
		   M_nr(row,col) = temp;
	   }
   }
   infile.close();
#if 0
   M_r<< 0.0113537,-0.00980196, 0.0212685, -0.0857003, -0.0214117,-0.0117607,
	  -0.0980196, 0.0286002,    0.13912,  0.0492078,  0.0872365, 0.0852322,
	   0.0212685,   0.13912,    1.67384, -0.293427,   0.512174,  0.0121294,
	  -0.0857003, 0.0492078,  -0.293427,  0.874917, -0.0835453,   0.23488,
	  -0.0214117, 0.0872365,   0.512174, -0.0835453,    1.88756,-0.0312228,
	 -0.00284587, -0.013616,-0.00252935,  0.131111,-0.00326836,    1.27364;   
   M_nr<< 0.0113537,-0.00980196, 0.0212685, -0.0857003, -0.0214117,-0.0117607,
	  -0.0980196, 0.0286002,    0.13912,  0.0492078,  0.0872365, 0.0852322,
	   0.0212685,   0.13912,    1.67384, -0.293427,   0.512174,  0.0121294,
	  -0.0857003, 0.0492078,  -0.293427,  0.874917, -0.0835453,   0.23488,
	  -0.0214117, 0.0872365,   0.512174, -0.0835453,    1.88756,-0.0312228,
	 -0.00284587, -0.013616,-0.00252935,  0.131111,-0.00326836,    1.27364;
#endif
   M_6 sigma;
   sigma<<0.001,0,0,0,0,0,
	      0,0.001,0,0,0,0,
		  0,0,0.005,0,0,0,
		  0,0,0,0.005,0,0,
		  0,0,0,0,0.005,0,
		  0,0,0,0,0,0.005;

   ContinuousState stateConvertor;
   LQControlInference infer_r(A,B,sigma,M_r,stateConvertor);
   infer_r.valueInference();
   LQContinuousPredictor predictor_r(infer_r,grid);
   LQControlInference infer_nr(A,B,sigma,M_nr,stateConvertor);
   infer_nr.valueInference();
   LQContinuousPredictor predictor_nr(infer_nr,grid);

   double prior_weight = -log(dims.first*dims.second);
   vector<vector<double> > prior(dims.first,
			   vector<double> (dims.second,prior_weight));
   //prior = 1/|G| number of goals
   predictor_r.setPrior(prior);
   predictor_nr.setPrior(prior);
   vector<vector<double> > posterior,likelihoods;

   BMPFile gridView(dims.first, dims.second);
   cout << testSet.size() <<" Examples"<<endl;

   for (int i=0; i < testSet.size(); i++) {
	   

      vector<pair<int, int> > & traj = testSet.at(i);
      vector<pair<double, double> > & rawObs = testSet.at_raw(i);
	  if(rawObs.size()==0){
		  cout<<"Empty raw obs"<<endl;
		  return 0;
	  }
      vector<double> times = testSet.getTimes(i); 
	  vector<double>* rawTimes = &testSet.at_rawTime(i);
	  pair<int,int> & botinGrid = testSet.at_bot(i); 
	  pair<double,double> & botinPoint = testSet.at_rbot(i);
	  pair<int,int> start = traj.front();
	  pair<double,double>& end = rawObs.back();

	  predictor_r.setOriginWrapper(rawObs.front());
	  predictor_nr.setOriginWrapper(rawObs.front());

	  int rawIndex = 0;
	  int prevIndex = 0;
	  int gridIndex = 0;
      char buf[512];
      sprintf(buf, "../output/lqpredict%03d.dat", i);
      ofstream outfile(buf);
#if 0
	  predictor_nr.test(rawObs);
	  predictor_nr.setOriginWrapper(rawObs.front());
	  cout<<"********CHECK FINISH****************"<<endl;
#endif
	  double startTime = rawTimes->front();
	  for (double tick=startTime;rawIndex<rawObs.size();tick+=interval) {

		for (;gridIndex < traj.size()&&times.at(gridIndex) < tick;
					       gridIndex++);
			if (gridIndex == traj.size() ) break;
 
		for (;rawIndex < rawObs.size()&&rawTimes->at(rawIndex) < tick;
					       rawIndex++); 
			if (rawIndex == rawObs.size() ) break;

	        cout <<"Evidence: "<<i<<" timestep: "
				<<tick-startTime<<" index: "<<rawIndex<<" previous: "<<prevIndex<<endl;
			predictor_nr.predictAll(rawObs,prevIndex,rawIndex);

			//Throughout testing
//			predictor_nr.testAfterPredict(rawObs,rawIndex);

//			double botLikelihood = 
//				predictor_r.predictPoint(rawObs,rawTimes,botinPoint,prevIndex,rawIndex);
			//cout<<"update index"<<endl;
			prevIndex = rawIndex;
			likelihoods = predictor_nr.getLikelihoods(); 
            cout<<"end: "<<likelihoods.at(traj.back().first).at(traj.back().second);
#if 0
			int pointResult = botLikelihood*10;
			int allResult = likelihoods.at(botinGrid.first).at(botinGrid.second)*10;
			if(pointResult!=allResult){
				cout<<"**********************P: "<<botLikelihood<<" A: "<<
					likelihoods.at(botinGrid.first).at(botinGrid.second)<<endl;
			}
			assert(pointResult==allResult);
#endif
	//		likelihoods.at(botinGrid.first).aNonRobCostMatrix690t(botinGrid.second) = botLikelihood;
			genPosterior(prior,posterior,likelihoods);

			//cout<<"Write output"<<endl;
			vector<pair<int, int> > subTraj;
            subTraj.insert(subTraj.end(), traj.begin(), traj.begin()+gridIndex);
      
		      
			double maxV = -HUGE_VAL;
            double minV = HUGE_VAL;
			pair<int,int> predestGrid;
			pair<double,double> predestPoint;
    
			  for(int ii=0; ii< dims.first; ii++) { 
                  for (int jj=0; jj < dims.second; jj++) {
						
					  if(posterior[ii][jj]>maxV){
							predestGrid.first = ii;
							predestGrid.second = jj;
						}
					  maxV = max(maxV, posterior.at(ii).at(jj));
					  minV = min(minV, posterior.at(ii).at(jj));
                  }
			  }
			  
			  predestPoint = grid.grid2Real(predestGrid.first,predestGrid.second);
		      double dist = sqrt((end.first-predestPoint.first)*
						  (end.first-predestPoint.first)+
						  (end.second-predestPoint.second)*
						  (end.second-predestPoint.second));

			  double logloss = entropy(posterior);

#if 0
		    gridView.addBelief(posterior, -50, -0.0, white,red);
#else
		    gridView.addBelief(posterior, -30, -0.0, jet);
#endif
            grid.addObstacles(gridView, black);
            gridView.addVector(subTraj, red, 1);
		    gridView.addLabel(botinGrid,green);
            sprintf(buf, "../compare/lq%03d-%f.bmp", i, tick-startTime); 
            gridView.write(buf);


            cout <<"BELIEFS: "<<rawIndex<<" "
				<<posterior.at(botinGrid.first).at(botinGrid.second)
				<<" Max: "<<maxV<<" Min: "<<minV<<" Start: "
				<<posterior.at(start.first).at(start.second)
				<<" log loss: "<<logloss<<endl;
			outfile <<tick-startTime
				<<" "<<logloss
				<<" "<<posterior[botinGrid.first][botinGrid.second]
				<<" "<<posterior[traj.back().first][traj.back().second]
				<<" "<<maxV<<" "<<dist<<endl;
			


	  }
      outfile.close(); 
   }

   return 0;

}


