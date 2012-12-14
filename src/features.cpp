#include "features.h"


DiscreteSeqFeature::DiscreteSeqFeature(vector<double> &_disTable)
		:disTable(_disTable) {
}

int DiscreteSeqFeature::getFeat(double realv){
	if (realv < disTable.at(0)){
		cout << "The real number exceed the low bound!" <<endl;	
		return -1;
	}
	int n = num_V();
	int newfeature;

	for (int ind = 0; ind < n-1; ind++){
		if (disTable.at(ind)<= realv && realv < disTable.at(ind+1) ){
			newfeature = ind;
			break;
		}		  
	}
	if (realv >= disTable.at(n-1))
	  newfeature = n-1;
	
	return newfeature;
}

double DiscreteSeqFeature::featureCounts(vector<vector<vector<double> > > &occupancy){
   double counts = 0.0;
   int dim_V = occupancy.at(0).at(0).size();
   
   for (int k=0;k < dim_V; k++){
	   double weight = 0.0;
	   for (int x=0; x < occupancy.size(); x++) {
		   for (int y=0; y < occupancy.at(0).size(); y++) {
			   if (isnan(exp(occupancy.at(x).at(y).at(k)))) 
				 continue;
               weight += exp(occupancy.at(x).at(y).at(k));
           }
       }
       counts += weight*k;     
	 //  cout<<"model vel "<<k<<" weight: "<<weight<<" count: " <<counts<<endl;
   }
   return counts;
}

double DiscreteSeqFeature::featureCounts(vector<double> &seq){
   double counts = 0.0;
   for (int i=0; i < seq.size(); i++) {
	   counts+= getFeat(seq.at(i));
   }
   //cout<<"path vel feature counts:  "<<counts<<endl;
   return counts;
}


DisVecSeqFeature::DisVecSeqFeature(vector<double> &_disTable)
		:DiscreteSeqFeature(_disTable) {
}

vector<int> DisVecSeqFeature::getFeat_vec(double realv){
	int n = disTable.size();
    vector<int> velocity(n,0);
	velocity.at(getFeat(realv)) = 1;
	return velocity;
}

void DisVecSeqFeature::featureCounts_vec(
			vector<vector<vector<double> > > &occupancy,
			vector<double> &featCount){
   int dis_V = disTable.size();  
   double sum = 0.0;
   for (int k=0;k < dis_V; k++){
	   double weight = 0.0;
	   for (int x=0; x < occupancy.size(); x++) {
		   for (int y=0; y < occupancy.at(0).size(); y++) {
			   if (isnan(exp(occupancy.at(x).at(y).at(k)))) 
				 continue;
               weight += exp(occupancy.at(x).at(y).at(k));
           }
       }
	   sum += weight;
       featCount.push_back(weight);     
	   //cout<<"model vel "<<k<<" counts: "<<weight<<endl;
   }
/*
   for (int k=0;k < dis_V; k++){
	   featCount.at(featCount.size()-1-k) = 
		   featCount.at(featCount.size()-1-k)/sum; 
   }
*/
}

void DisVecSeqFeature::
     featureCounts_vec(vector<double> &seq, vector<double> &featCount){
   int dis_V = disTable.size();  
   double sum = 0.0;
   vector<double> counts (dis_V,0.0);
   for (int i=0; i < seq.size(); i++) {
	   counts.at(getFeat(seq.at(i))) += 1;
   }
   for (int v=0;v<dis_V;v++){
	   sum +=counts[v];
   }
   for (int v=0;v<dis_V;v++){
	 featCount.push_back(counts[v]);///sum
     //cout<<"path vel "<<v<<" counts: "<<counts[v]<<endl;
   }
   counts.clear();
}

ThreshBoolSeqFeature::ThreshBoolSeqFeature( double t) :  thresh(t) {
}

int ThreshBoolSeqFeature::getFeat(double realv){
	return  (realv  <=  thresh ? 1:0);
}

RobotLocalBlurFeature::RobotLocalBlurFeature(Grid &grid,
			pair<int,int> &robpos, int radius){
	
	pair<int,int> dims = grid.dims();
    values.resize(dims.first,vector<double>(dims.second,0.0));

    int N = radius*radius + (radius-1)*(radius-1);

    int count = 0;
    for (int x=max(0,robpos.first-radius+1); 
				x < min(dims.first, robpos.first+radius); x++) {
         int r = radius - abs(robpos.first - x); 
         for (int y=max(0,robpos.second-r+1); 
					 y < min(dims.second, robpos.second+r); y++)  {
			 if (grid.at(x,y)==1){
			    continue;
			 }
              count++;
              values.at(x).at(y) = 1.0; 
         }
    }
    //if (count > N) 
      //cout << "COUNT: "<<count<<endl;
}

RobotGlobalFeature::RobotGlobalFeature(Grid &grid, pair<int,int> &robpos){
	
	pair<int,int> dims = grid.dims();
	values.resize(dims.first,vector<double>(dims.second,0.0));
    
	for (int x=0;x < dims.first;x++){
		for(int y=0;y<dims.second;y++){
			if (grid.at(x,y))
			  continue;
			double dist = sqrt(pow(x-robpos.first,2)+pow(y-robpos.second,2));
	        values.at(x).at(y) = (1e-4)*exp(0.05*dist);
			//cout<< (1e-6)*exp(0.05*dist)<<endl;
		}
	}

}


ConstantFeature::ConstantFeature(Grid &grid) {
   pair<int, int> dims = grid.dims();

   values.resize(dims.first, vector<double>(dims.second, 0.0));

   for (int i=0; i < dims.first; i++) 
      for (int j=0; j < dims.second; j++) 
         values.at(i).at(j) = 1.0; 
}


ObstacleFeature::ObstacleFeature(Grid &grid) {
   pair<int, int> dims = grid.dims();

   values.resize(dims.first, vector<double>(dims.second, 0.0)); 

   for (int i=0; i < dims.first; i++) {
      for (int j=0; j < dims.second; j++) {
         int value = grid.at(i, j);
         values.at(i).at(j) = (value == 1);//value==1 meaning obstacle
      }
   }
}

ObstacleBlurFeature::ObstacleBlurFeature(Grid &grid, int radius) {
   pair<int, int> dims = grid.dims();

   values.resize(dims.first, vector<double>(dims.second, 0.0)); 

   int N = radius*radius + (radius-1)*(radius-1);

 
   double totalSum = 0.0;

   for (int i=0; i < dims.first; i++) {
      for (int j=0; j < dims.second; j++) {
         int value = grid.at(i, j);
         if (value != 1)//free space
            continue;  

         int count = 0;
         for (int x=max(0,i-radius+1); x < min(dims.first, i+radius); x++) {
            int r = radius - abs(i - x); 
            for (int y=max(0,j-r+1); y < min(dims.second, j+r); y++)  {
                count++;
                values.at(x).at(y) += 1.0/N; 
                totalSum += 1.0/N;
            }
         }
         //if (count > N) 
         //if (i-j == 100)
         //   cout << "COUNT: "<<count<<"  "<<N<<endl;
      }
   } 
   //cout << "TOTAL: "<<totalSum<<endl;
}

ObstacleMaxFeature::ObstacleMaxFeature(Grid &grid, int radius) {
   pair<int, int> dims = grid.dims();

   values.resize(dims.first, vector<double>(dims.second, 0.0)); 

   double totalSum = 0.0;
   
   int count=0;
   
   for (int i=0; i < dims.first; i++) {
      for (int j=0; j < dims.second; j++) {
         int value = grid.at(i, j);
         if (value != 1)//free space
            continue;
         for (int x=max(0,i-radius); x < min(dims.first, i+radius); x++) {
            int r = radius - abs(i - x);
            for (int y=max(0,j-r); y < min(dims.second, j+r); y++)  {
               count+=(int)(1-values.at(x).at(y));
               values.at(x).at(y) = 1.0; 
            }
         }
      }
   } 
   cout << "FRAC: "<<(double)count/(dims.first*dims.second)<<endl;
}

FeatureArray::FeatureArray(vector<PosFeature> &_features) 
   : N(_features.size()) {

   dimens = _features.at(0).dims();
   values.resize(dimens.first, vector<vector<double> >(dimens.second,
            vector<double>(_features.size(), 0.0)));

   for (int i=0; i < dimens.first; i++) 
      for (int j=0; j < dimens.second; j++) 
         for (int k=0; k < _features.size(); k++) 
            values.at(i).at(j).at(k) = _features.at(k).at(i,j);

}

void FeatureArray::featureCounts(vector<vector<double> > &occupancy,
      vector<double> &featCount) {

   featCount.clear();
   featCount.resize(values.at(0).at(0).size(), 0.0);

   for (int x=0; x < occupancy.size(); x++) {
      for (int y=0; y < occupancy.at(x).size(); y++) {

         //cout << "FC: "<<x<<"  "<<y<<"  "<<occupancy.at(x).at(y)<<endl;

         double weight = exp(occupancy.at(x).at(y));

         if (isnan(weight)) continue;

         for (int k=0; k < values.at(x).at(y).size(); k++) { 
            featCount.at(k) += values.at(x).at(y).at(k) * weight;
         }
      }

   }
}

void FeatureArray::featureCounts(vector<vector<vector<double> > > &occupancy,
      vector<double> &featCount) {
	

   featCount.clear();
   featCount.resize(N, 0.0);
   
   int dim_V = occupancy.at(0).at(0).size();

   for (int x=0; x < dimens.first; x++) {
      for (int y=0; y < dimens.second; y++) {
         double weight = 0.0;
		 for (int k = 0;k<dim_V;k++){
			 if (isnan(exp(occupancy.at(x).at(y).at(k)))) 
			   continue;
		   weight += exp(occupancy.at(x).at(y).at(k));
		 }
         for (int i=0; i < N; i++) { 
            featCount.at(i) += values.at(x).at(y).at(i) * weight;
			/*if (i==6||i==7){
				if((weight>=0.00001)&&values.at(x).at(y).at(i)!=0){
				   cout<<i<<" weight: "<<weight<<" v: "<<values.at(x).at(y).at(i)
					 <<" cout: "<<featCount.at(i)<<endl;
				}
			 }*/
         }
      }
   }
}


void FeatureArray::featureCounts(vector<pair<int, int> > &path,
      vector<double> &featCount) {
   featCount.clear();
   featCount.resize(values.at(0).at(0).size(), 0.0);

   for (int i=0; i < path.size(); i++) {
      int x = path.at(i).first;
      int y = path.at(i).second;

      double weight = 1.0;

      for (int k=0; k < values.at(x).at(y).size(); k++) {
         featCount.at(k) += values.at(x).at(y).at(k) * weight;
      } 
   } 
}

FeatureArray::FeatureArray(FeatureArray &featArrayIn, int factor) {
   dimens = featArrayIn.dims();
   N = featArrayIn.size();
   int newMaxX = (int)ceil((float)dimens.first/factor);
   int newMaxY = (int)ceil((float)dimens.second/factor);

   dimens.first = newMaxX;
   dimens.second = newMaxY;

   values.resize(newMaxX, vector<vector<double> >(newMaxY,
         vector<double>(featArrayIn.size(), 0.0))); 

   vector<vector<int> > counts(newMaxX,
         vector<int>(newMaxY, 0));
	

   /* Take compressed feature sums */
   for (int i=0; i < dimens.first; i++) { 
      for (int j=0; j < dimens.second; j++) {
         counts.at(i/factor).at(j/factor)++;
         vector<double> feats = featArrayIn.at(i, j);

         for (int k=0; k < feats.size(); k++) 
            values.at(i/factor).at(j/factor).at(k) += feats.at(k); 
      }
   }

   /* Normalize all values */
   for (int i=0; i < dimens.first; i++) 
      for (int j=0; j < dimens.second; j++) 
         for (int k=0; k < values.at(i).at(j).size(); k++) 
            values.at(i).at(j).at(k)/=counts.at(i).at(j); 
}


/*
//TO DO: modify the SeqFeatureArray class implementation
SeqFeatureArray::SeqFeatureArray(vector<SeqFeature*> &_features) :
 N(_features.size()), sFeats(_features) {

	// first dimension: sequence length
	 second dimension: # of feature types
	
	L  = _features.at(0)->lens();
    values.resize(L, vector<int> (N,0));
    D = _features.at(0)->num_V();
   for (int i=0; i < L; i++) 
      for (int j=0; j < N; j++)
		values.at(i).at(j)  = _features.at(j)->at(i); 
}

vector<int> SeqFeatureArray::getOnePointFeatArray(double realv){
	vector<int> featarray;
	for (int i=0;i<N;i++){
		featarray.push_back(sFeats.at(i)->getFeat(realv));
	}
	return featarray;
}

void SeqFeatureArray::featureCounts(vector<vector<double> > &occupancy,
      vector<double> &featCount) {
}

void SeqFeatureArray::featureCounts(vector<pair<int, int> > &path,
      vector<double> &featCount) {
}*/


