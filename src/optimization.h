#ifndef OPTIMIZATION_H__
#define OPTIMIZATION_H__

#include "main.h"
#include "grid.h"
#include "evidence.h"
#include "features.h"
#include "inference.h"
#include <math.h>


enum {OPT_LINEAR, OPT_EXP};

class Optimizer { 
   public:
      Optimizer():itrcount(0){}
      virtual double eval(vector<double> &params, vector<double> &gradient) = 0; 
      void optimize(vector<double> &params, double step,int itrTimes, 
            double accuracy, int method=OPT_LINEAR);
   protected:
      int itrcount;

};

/* Optimization class for position feature model only*/
class trajectoryOptimizer : public Optimizer {
   public:
      trajectoryOptimizer(Grid &_grid, Evidence &_evidence,
           vector<PosFeature> &_features, InferenceEngine &_engine):
           Optimizer(),grid(_grid),evidence(_evidence),features(_features),
           engine(_engine){}
      double eval(vector<double> &params,vector<double> &gradient);
   protected:
      Grid &grid;
      Evidence &evidence;
      vector<PosFeature> &features;
      InferenceEngine &engine;
};

/* Optimization class for model with discrete velocity feature*/
class trajOptimizerplus : public Optimizer {

   public:
	   trajOptimizerplus(Grid &_grid, Evidence &_evidence,
          vector<PosFeature> &_posFeatures, DisVecSeqFeature &_seqFeat, 
		  DisSeqOrderInferEngine &_engine) : Optimizer(),
          grid(_grid),evidence(_evidence),
	      posFeatures(_posFeatures),seqFeat(_seqFeat),engine(_engine){}
       double eval(vector<double> &params, vector<double> &gradient);

   protected:
      Grid &grid;
      Evidence &evidence;
      vector<PosFeature> &posFeatures;
	  DisVecSeqFeature &seqFeat;
      DisSeqOrderInferEngine &engine;
};
	


#endif
