#ifndef FEATURES_H__
#define FEATURES_H__

#include "main.h"
#include "grid.h"

#include <vector>

using namespace std;

class SeqFeature {
   public:
	   virtual int getFeat(double realv) = 0;
	   virtual int num_V() =  0;
	   virtual double featureCounts(vector<vector<vector<double> > >
				   &occupancy) = 0;
	   virtual double featureCounts(vector<double> &seq)  = 0;
   protected:
  
};

class DiscreteSeqFeature : public SeqFeature {
   public:
	   DiscreteSeqFeature(vector<double> &_disTable);
	   int num_V(){
		   return disTable.size();
	   }
	   int getFeat(double realv);
	   double featureCounts(vector<vector<vector<double> > > &occupancy);
	   double featureCounts(vector<double> &seq);
   protected:
	   vector<double> disTable;
};


class DisVecSeqFeature : public DiscreteSeqFeature{
	public:
	   DisVecSeqFeature(vector<double> &_disTable);
	   vector<int> getFeat_vec(double realv);
	   void featureCounts_vec(vector<vector<vector<double> > >
				   &occupancy, vector<double> &featCount);
	   void featureCounts_vec(vector<double> &seq, vector<double>& 
				   featCount);
   protected:
};

/*
class DiffBoolSeqFeature : public SeqFeature {
   public:
	   DiffBoolSeqFeature(vector<double> &realseq, double dt);
       static const int num_V = 2;
	   int getFeat(double prevrealv, double currtrealv);
   protected:
	   double diffthresh;
};
*/
class ThreshBoolSeqFeature : public SeqFeature {
   public:
	   ThreshBoolSeqFeature( double t);
       int num_V() { int v = 2; return v;}
	   int getFeat(double realv);
   protected:
	   double thresh;
};

class PosFeature {
   public:
      double at(int i, int j) { 
         return values.at(i).at(j); 
      }
      pair<int, int> dims() {
         pair<int, int> dimensions;
         dimensions.first = values.size();
         if (values.size() > 0) 
            dimensions.second = values.at(0).size();
         else
            dimensions.second = 0;
         return dimensions;
      } 
	  vector<vector<double> >& getMap(){return values;}

   protected:
      vector<vector<double> > values;
  
};

class ConstantFeature : public PosFeature {
   public:
      ConstantFeature(Grid &grid);
   protected:
};

class ObstacleFeature : public PosFeature {
   public:
      ObstacleFeature(Grid &grid); 
   protected: 
};

class ObstacleBlurFeature : public PosFeature { 
   public:
      ObstacleBlurFeature(Grid &grid, int radius);
   protected:
};

class ObstacleMaxFeature : public PosFeature { 
   public:
      ObstacleMaxFeature(Grid &grid, int radius);
   protected:

};

class RobotLocalBlurFeature : public PosFeature {
   public:
      RobotLocalBlurFeature(Grid &grid, pair<int,int> &robpos, 
				  int radius);
   protected:

};

class RobotGlobalFeature : public PosFeature {
   public:
      RobotGlobalFeature(Grid &grid, pair<int,int> &robpos);
   protected:
};

class FeatureArray {
   public:
      FeatureArray(vector<PosFeature> &_features);
      FeatureArray(FeatureArray &featArray, int factor);
      vector<double> & at(int i, int j) { return values.at(i).at(j); }
      int size() { return N; }
      void featureCounts(vector<vector<double> > &occupancy, 
            vector<double> &featCount);
      void featureCounts(vector<vector<vector<double> > > &occupancy, 
            vector<double> &featCount);
      void featureCounts(vector<pair<int,int> > &path,
            vector<double> &featCount); 
      pair<int, int> dims() { return dimens; }
   protected:
      //Grid grid;
      vector<vector<vector<double> > > values;
      int N;
      pair<int, int> dimens;
};
/*
class SeqFeatureArray {
   public:
      SeqFeatureArray(vector<SeqFeature*> &_features);
      vector<int> & at(int i) { return values.at(i);}
      int size() { return N; }
	  vector<int > getOnePointFeatArray(double realv);
	  void featureCounts(vector<vector<double> > &occupancy, 
            vector<double> &featCount); 
	  void featureCounts(vector<pair<int,int> > &path,
            vector<double> &featCount);
	  int lens(){ return L;}
	  int dims(){ return D;}
   protected:
	  vector<SeqFeature*> sFeats;
	  vector<vector<int> > values;
      int N;//# of features
	  int L;
      int D;//dimension of features
};*/


#endif
