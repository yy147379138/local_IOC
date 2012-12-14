/*****************************************************
 * Header for the Eigen linear algebra library
 * Deyang Zhao
 *
 *  ***************************************************/

#ifndef MYMATH_H__
#define MYMATH_H__

#include "Eigen/Dense"

#define PI 3.14159265
using namespace Eigen;

/* Matrix class in Eigen is implemented in arrays.
 * Elements in a matrix are stored in contiguous array.
 *
 *  */

typedef Matrix<double,8,8> M_8;
typedef Matrix<double,6,6> M_6;
typedef Matrix<double,6,2> M_6_2;
typedef Matrix<double,2,6> M_2_6;
typedef Matrix<double,6,8> M_6_8;
typedef Matrix2d M_2;
typedef Matrix4d M_4;
typedef Matrix<double,6,1> Col_6;
typedef Matrix<double,8,1> Col_8;
typedef Vector2d Col_2;



class Gaussian{
	public:
		Gaussian(VectorXd _mean, MatrixXd _covariance, 
			double _factor=0):mean(_mean), 
		    covariance(_covariance),
		    factor(_factor){
				precision = covariance.transpose();
		}

		static double gaussian(VectorXd& x, VectorXd& mean, 
					MatrixXd& precision){
			if(x.size() != mean.size() ||
						x.size() != precision.rows()||
						x.size() != precision.cols()){
				cerr << "Error : not matched dimension in Gaussion!!!!!!!!!!!!!!!!" << endl;
				return -1;
			}else{
				VectorXd diff(x.size());
				VectorXd prod(x.size());

				diff = x - mean;
				prod = precision * diff;
                                
                                double d = diff.transpose() * prod;
                                d = -0.5*d;
                                d = exp(d);
                                d = d/2/3.1415926;
                                d = d*precision(0,0)*precision(1,1);
				return d;  
			}
		}

		VectorXd mean;
		MatrixXd covariance;
		MatrixXd precision;
		double factor;
	private:
};
/*
void matrixExp(MatrixXd& indice, MatrixXd& Exp){
	EigenSolver<MatrixXd> eigensolver(indice);
	MatrixXd D = eigensolver.pseudoEigenvalueMatrix();
	MatrixXd V = eigensolver.pseudoEigenvectors();
	for(int ii=0;ii<D.rows();ii++){
		D(ii,ii) = exp(D(ii,ii));
	}
	Exp = V*D*V.inverse();
}

void matrixLog(MatrixXd& indice, MatrixXd& Log){
	EigenSolver<MatrixXd> eigensolver(indice);
	MatrixXd D = eigensolver.pseudoEigenvalueMatrix();
	MatrixXd V = eigensolver.pseudoEigenvectors();
	for(int ii=0;ii<D.rows();ii++){
		D(ii,ii) = log(D(ii,ii));
	}
	Log = V*D*V.inverse();
}
*/
#endif
