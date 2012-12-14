#include "options.h"
#include "visualize.h"
#include "localOptimalIOC.h"
#include <math.h>

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
   MatrixXd M(6,6);

   ifstream infile("../params/learn.dat");
   for(int row=0;row<M.rows();row++){
	   for(int col=0;col<M.cols();col++){
		   double temp;
		   infile>>temp;
		   M(row,col) = temp;
	   }
   }
   infile.close();
   cout << M << endl;
   M_6 sigma;
   sigma<<0.001,0,0,0,0,0,
	      0,0.001,0,0,0,0,
		  0,0,0.005,0,0,0,
		  0,0,0,0.005,0,0,
		  0,0,0,0,0.005,0,
		  0,0,0,0,0,0.005;

   VectorXd theta(2);
   theta<< -1, -1;
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
   L->setTheta(theta);

   LocalIOCPredictor predictor(grid, L);


   double prior_weight = -log(dims.first*dims.second);
   vector<vector<double> > prior(dims.first,
			   vector<double> (dims.second,prior_weight));
   predictor.setPrior(prior);
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


	  int rawIndex = 0;
	  int gridIndex = 0;
      char buf[512];
      sprintf(buf, "../output/local%03d.dat", i);
      ofstream outfile(buf);
	  double startTime = rawTimes->front();
	  for (double tick=startTime;rawIndex<rawObs.size();tick+=interval) {

		for (;gridIndex < traj.size()&&times.at(gridIndex) < tick;
					       gridIndex++);
			if (gridIndex == traj.size() ) break;
 
		for (;rawIndex < rawObs.size()&&rawTimes->at(rawIndex) < tick;
					       rawIndex++); 
			if (rawIndex == rawObs.size() ) break;

	        cout <<"Evidence: "<<i<<" timestep: "
				<<tick-startTime<<" index: "<<rawIndex<<endl;
			predictor.predictAll(rawObs, rawIndex);

			posterior = predictor.getPosterior(); 
			likelihoods = predictor.getLikelihoods(); 
		    sprintf(buf, "../output/matrix.dat");
			ofstream matrix(buf);
	        for(int ii=0; ii< dims.first; ii++) { 
                  for (int jj=0; jj < dims.second; jj++) {
					  matrix << likelihoods[ii][jj] << " ";
                  }
				  matrix << endl;
			}
			matrix.close();





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
            sprintf(buf, "../compare/local%03d-%f.bmp", i, tick-startTime); 
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


