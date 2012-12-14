#ifndef LOCALOPTIMALIOC
#define LOCALOPTIMALIOC

#include "main.h"
#include "mymath.h"
#include "grid.h"
#include "linearquadratic.h"

class Tensor{
	public:
        enum{T, M, V};
	    Tensor(int h_dim_1, int h_dim_2, int i_dim_1, int i_dim_2){
			out_dims.first = h_dim_1;
			out_dims.second = h_dim_2;
			in_dims.first = i_dim_1;
			in_dims.second = i_dim_2;
		    
		};	

	//	bool multiply(Tensor* multiplier, Tensor* product);
	
		// virtual bool plus(Tensor* adder, Tensor* sum);
		pair<int,int>& outer_dims(){
			return out_dims;
		}
		pair<int,int>& inner_dims(){
			return in_dims;
		}

        inline int rows(){
			return out_dims.first;
		}
        inline int cols(){
			return out_dims.second;
		}
		int get_type(){
			return type;
		}

	protected:
		int type;
		/** outer(tensor) dimension **/ 
		pair<int,int> out_dims;
		/** inner(data) dimension **/ 
		pair<int,int> in_dims;


};

/*
 * Tensor of vecros
 */
class Tensor_v : public Tensor{
	public:
	    Tensor_v(int h_dim_1, int h_dim_2, int in_length) : Tensor(h_dim_1,
					h_dim_2, in_length, 1){
			/** Initialize data **/
			VectorXd v(in_length);
			v.setZero();
			data.resize(h_dim_1, vector<VectorXd>(h_dim_2,v));
			type = Tensor :: V;
		}	
/*
		bool plus(Tensor* adder, Tensor* sum){
			Tensor_v* adder_v = dynamic_cast<Tensor_v*>(adder);
			Tensor_v* sum_v = dynamic_cast<Tensor_v*>(sum);

			pair<int,int> o_dim1 = adder->outer_dims();
			pair<int,int> o_dim2 = sum->outer_dims();
			pair<int,int> i_dim1 = adder->inner_dims();
			pair<int,int> i_dim2 = sum->inner_dims();

			if(o_dim1.first != out_dims.first || o_dim1.second != out_dims.second
			|| o_dim2.first != out_dims.first || o_dim2.second != out_dims.second
			|| i_dim1.first != in_dims.first || i_dim1.second != in_dims.second
			|| i_dim2.first != in_dims.first || i_dim2.second != in_dims.second){
				cerr << "Error: Dimensions not matched!" << endl;
				return false;
			}else{
				VectorXd temp;
				for(int x = 0; x < out_dims.first; x++){
					for(int y = 0; y < out_dims.second; y++){
                        adder_v->get_v(x, y, temp);
						sum_v->set_v(x, y, data.at(x).at(y) + temp);
					}
				}

				adder = adder_v;
				sum = sum_v;
				return true;
			}
		}
		*/
		bool plus(Tensor_v* adder, Tensor_v* sum){

			pair<int,int> o_dim1 = adder->outer_dims();
			pair<int,int> o_dim2 = sum->outer_dims();
			pair<int,int> i_dim1 = adder->inner_dims();
			pair<int,int> i_dim2 = sum->inner_dims();

			if(o_dim1.first != out_dims.first || o_dim1.second != out_dims.second
			|| o_dim2.first != out_dims.first || o_dim2.second != out_dims.second
			|| i_dim1.first != in_dims.first || i_dim1.second != in_dims.second
			|| i_dim2.first != in_dims.first || i_dim2.second != in_dims.second){
				cerr << "Error: Dimensions not matched!" << endl;
				return false;
			}else{
			    VectorXd temp(in_dims.first);
				for(int x = 0; x < out_dims.first; x++){
					for(int y = 0; y < out_dims.second; y++){
                        adder->get_v(x, y, temp);
						sum->set_v(x, y, data.at(x).at(y) + temp);
					}
				}
				return true;
			}
		}

		MatrixXd expand(){
			MatrixXd result(out_dims.first*in_dims.first, out_dims.second);
			int x_m_step = in_dims.first;
			for(int x_t = 0; x_t < out_dims.first; x_t++){
				for(int y_t = 0; y_t < out_dims.second; y_t++){
					for(int x_m = 0; x_m < x_m_step; x_m++){
						result(x_t*x_m_step + x_m, y_t) = 
							data.at(x_t).at(y_t)(x_m);
					}
				}
			}
			return result;
		}

		bool collapse(VectorXd v, MatrixXd& result){
			if(v.size() != in_dims.first || result.rows() != 
						out_dims.first || result.cols() != 
						out_dims.second){
				cerr << "Error: Dimensions not matched!" << endl;
				return false;
			}else{
				for(int x = 0; x < out_dims.first; x++){
					for(int y = 0; y < out_dims.second; y++){
						result(x,y) = data.at(x).at(y).transpose() * v;
					}
				}
				return true;
			}	

		}

		bool set_v(int i, int j, VectorXd v){
			if(v.size() != in_dims.first){ return false; }
			else{
				data.at(i).at(j) = v;
				return true;
			}
		}

		bool get_v(int i, int j, VectorXd& re){
			if(i >= out_dims.first || i < 0 ||
						j >= out_dims.second || j < 0){
				return false;
			}else{
				re = data.at(i).at(j);
				return true;
			}
		}

		VectorXd get_v(int i, int j){
			return data.at(i).at(j);
		}

	private:
		vector<vector<VectorXd> > data;
};


/*
 * Tensor of matrices
 */
class Tensor_m : public Tensor{
    public:
		Tensor_m(int h_dim_1, int h_dim_2, int i_dim_1, int i_dim_2) : Tensor(h_dim_1,
					h_dim_2, i_dim_1, i_dim_2){
			/** Initialize data **/
			MatrixXd m(i_dim_1,i_dim_2);
			m.setZero();
			data.resize(h_dim_1, vector<MatrixXd>(h_dim_2,m));
			type = Tensor :: M;
		}
		
		/* Method called on a tensor_m, multiplied with a tensoe_v, resulting in
		 * tensor_v */
		bool multiply(Tensor_v* multiplier, Tensor_v* product){
			if(type != M || cols() != multiplier->rows()||
						in_dims.second != multiplier->inner_dims().first||
						rows() != product->rows()||
						in_dims.first != product->inner_dims().first ||
						multiplier->cols() != product->cols()){
				cerr << "Error: Multiplication unmatched dimensions!" << endl;
				return false;
			}else{
				for(int row1 = 0; row1 < rows(); row1++){
				  // row of the matrix tensor
				  for(int col2 = 0; col2 < multiplier->cols(); col2++){
					  // col of the vector tensor
					  VectorXd temp(in_dims.first);
					  temp.setZero();
					  for(int col1 = 0; col1 < cols(); col1++){
						  temp += data.at(row1).at(col1) * multiplier->get_v(col1,col2); 
					  }
					  product->set_v(row1, col2, temp);
				  }
				}
				return true;
			}
		}


		/* Method called on a tensor_m, multiplied with a tensoe_m, resulting in
		 * tensor_m */
		bool multiply(Tensor_m* multiplier, Tensor_m* product){
			if(type != M || cols() != multiplier->rows()||
						rows() != product->rows()||
						multiplier->cols() != product->cols()||
						in_dims.first != product->inner_dims().first ||
						in_dims.second != multiplier->inner_dims().first||
						multiplier->inner_dims().second != product->inner_dims().second){
				cerr << "Error: Multiplication unmatched dimensions!" << endl;
				return false;
			}else{
				for(int row1 = 0; row1 < rows(); row1++){
				  // row of the matrix tensor
				  for(int col2 = 0; col2 < multiplier->cols(); col2++){
					  // col of the vector tensor
					  MatrixXd temp(in_dims.first, multiplier->inner_dims().second);
					  temp.setZero();
					  for(int col1 = 0; col1 < cols(); col1++){
						  temp += data.at(row1).at(col1) * multiplier->get_m(col1,col2); 
					  }
					  product->set_m(row1, col2, temp);
				  }
				}
				return true;
			}
		}

		/* 
		 * Method called for Jacobian matrix 
		 * Implements:
		 * Each block is vectorized and then the outer product 
		 * is computed for each localtion
		 * Return a tensor of large matrix
							      data.at(j).at(t).row(1).transpose();
		 * */
		bool vector_inner_product(Tensor_m* result){
			int N = in_dims.first * in_dims.second;
			if(rows() != result->rows() || cols() != result->cols()||
					result->inner_dims().first != N ||
					result->inner_dims().second != N){
				cerr << "Error: Multiplication unmatched dimensions!" << endl;
				return false;
			}else{
				for(int row = 0; row < rows(); row++){
					for(int col = 0; col < cols(); col++){
						MatrixXd oneM(N, N);
						oneM.setZero();
						for(int t = 0; t < cols(); t++){
							VectorXd v1(N);
							VectorXd v2(N);
							v1 << data.at(row).at(t).row(0).transpose(), 
							      data.at(row).at(t).row(1).transpose();
							v2 << data.at(col).at(t).row(0).transpose(), 
							      data.at(col).at(t).row(1).transpose();
							oneM += v1 * v2.transpose();
						}
						result->set_m(row, col, oneM);
					}
				}
				return true;
			}
		}

		/* 
		 * Self_tensor_expanded * multiplier = product
		 * Method called for collapsing multiplication betweeen tensors of
		 * matrices
		 * Implements:
		 * The multiplier has higer dimension 
		 * The self tensor is expanded as a large matrix
		 * This matrix is multiplied by the tensor_m along the outest dimension
		 * Return a tensor of tensor_m
		 * */
		bool collapse_multiply(Tensor_m* multiplier, Tensor_m* product){

			if(cols()*in_dims.second != multiplier->rows() ||
						rows()*in_dims.first != product->rows() ||
						multiplier->cols() != product->cols() ||
						multiplier->inner_dims().first != 
						  product->inner_dims().first ||
						multiplier->inner_dims().second !=
						  product->inner_dims().second){
				cerr << "Error: Multiplication unmatched dimensions!" << endl;
				return false;
			}else{
				MatrixXd expanded = expand();

				for(int row = 0; row < expanded.rows(); row++){
					for(int col = 0; col < multiplier->cols(); col++){
						MatrixXd oneM(multiplier->inner_dims().first, 
									    multiplier->inner_dims().second);
						oneM.setZero();
						for(int k = 0; k < expanded.cols(); k++){
							oneM += expanded(row, k) * multiplier->get_m(k, col);
						}
						product->set_m(row, col, oneM);
					}
				}
				return true;
			}
		}


	    /* 
		 * Self_tensor_expanded * multiplier = product
		 * Method called for collapsing multiplication betweeen the self tensor of
		 * matrix and a tensor of vector (multiplier)
		 * Implements:
		 * The multiplier has higer dimension 
		 * The self tensor is expanded as a large matrix
		 * This matrix is multiplied by the tensor_m along the outest dimension
		 * Return a tensor of tensor_v
		 * */
		bool collapse_multiply(Tensor_v* multiplier, Tensor_v* product){

			if(cols()*in_dims.second != multiplier->rows() ||
						rows()*in_dims.first != product->rows() ||
						multiplier->cols() != product->cols() ||
						multiplier->inner_dims().first != 
						  product->inner_dims().first){
				cerr << "Error: Multiplication unmatched dimensions!" << endl;
				return false;
			}else{
				MatrixXd expanded = expand();

				for(int row = 0; row < expanded.rows(); row++){
					for(int col = 0; col < multiplier->cols(); col++){
						VectorXd oneV(multiplier->inner_dims().first);
						oneV.setZero();
						for(int k = 0; k < expanded.cols(); k++){
							oneV += expanded(row, k) * multiplier->get_v(k, col);
						}
						product->set_v(row, col, oneV);
					}
				}
				return true;
			}
		}




		/**
		 * left_m * self_tensor = product
		 * Left multiply a matrix. 
		 * Same multiplication as collapse_multiply except for the 
		 * method is called on the right tensor_m
		 */
		bool valid; 
		bool left_multiply(MatrixXd& left_m, Tensor_m* product){
			if(left_m.cols() != rows() || left_m.rows() != product->rows()||
						cols() != product->cols()||
						in_dims.first != product->inner_dims().first || 
						in_dims.second != product->inner_dims().second){
				cerr << "Error: Multiplication unmatched dimensions!" << endl;
				return false;
			}else{
				for(int row = 0; row < left_m.rows(); row++){
					for(int col = 0; col < cols(); col++){
						MatrixXd oneM(in_dims.first, in_dims.second);
						oneM.setZero();
						for(int k = 0; k < left_m.cols(); k++){
							oneM += left_m(row, k) * data.at(k).at(col);
						}
						product->set_m(row, col, oneM);
					}
				}
				return true;
			}
		}



		/**
		 * self_tensor * right_m = product
		 * Right multiply a matrix. 
		 * Same multiplication as collapse_multiply except for the 
		 * method is called on the left tensor_m taking a MatrixXd as 
		 * the parameter
		 */
		bool right_multiply(MatrixXd& right_m, Tensor_m* product){
			if(cols() != right_m.rows() || right_m.cols() != product->cols()||
						rows() != product->rows()||
						in_dims.first != product->inner_dims().first || 
						in_dims.second != product->inner_dims().second){
				cerr << "Error: Multiplication unmatched dimensions!" << endl;
				return false;
			}else{
				for(int row = 0; row < rows(); row++){
					for(int col = 0; col < right_m.cols(); col++){
						MatrixXd oneM(in_dims.first, in_dims.second);
						oneM.setZero();
						for(int k = 0; k < cols(); k++){
							oneM += right_m(k, col) * data.at(row).at(k);
						}
						product->set_m(row, col, oneM);
					}
				}
				return true;
			}
		}


/*
		bool plus(Tensor* adder, Tensor* sum){
			Tensor_m* adder_m = dynamic_cast<Tensor_m*>(adder);
			Tensor_m* sum_m = dynamic_cast<Tensor_m*>(sum);

			pair<int,int> o_dim1 = adder->outer_dims();
			pair<int,int> o_dim2 = sum->outer_dims();
			pair<int,int> i_dim1 = adder->inner_dims();
			pair<int,int> i_dim2 = sum->inner_dims();

			if(o_dim1.first != out_dims.first || o_dim1.second != out_dims.second
			|| o_dim2.first != out_dims.first || o_dim2.second != out_dims.second
			|| i_dim1.first != in_dims.first || i_dim1.second != in_dims.second
			|| i_dim2.first != in_dims.first || i_dim2.second != in_dims.second){
				cerr << "Error: Dimensions not matched!" << endl;
				return false;
			}else{
				MatrixXd temp;
				for(int x = 0; x < out_dims.first; x++){
					for(int y = 0; y < out_dims.second; y++){
                        adder_m->get_m(x, y, temp);
						sum_m->set_m(x, y, data.at(x).at(y) + temp);
					}
				}
				adder = adder_m;
				sum = sum_m;
				return true;
			}
		}
		*/

		bool plus(Tensor_m* adder, Tensor_m* sum){
			pair<int,int> o_dim1 = adder->outer_dims();
			pair<int,int> o_dim2 = sum->outer_dims();
			pair<int,int> i_dim1 = adder->inner_dims();
			pair<int,int> i_dim2 = sum->inner_dims();

			if(o_dim1.first != out_dims.first || o_dim1.second != out_dims.second
			|| o_dim2.first != out_dims.first || o_dim2.second != out_dims.second
			|| i_dim1.first != in_dims.first || i_dim1.second != in_dims.second
			|| i_dim2.first != in_dims.first || i_dim2.second != in_dims.second){
				cerr << "Error: Dimensions not matched!" << endl;
				return false;
			}else{
				MatrixXd temp;
				for(int x = 0; x < out_dims.first; x++){
					for(int y = 0; y < out_dims.second; y++){
                        adder->get_m(x, y, temp);
						sum->set_m(x, y, data.at(x).at(y) + temp);
					}
				}
				return true;
			}
		}

        /**
		 * Reshape a tensor_m into different outer and inner dimension
		 * The desired dimension is defined in reshaped
		 */
		bool reshape(Tensor_m* reshaped){
			int out_dim1 = reshaped->rows();
			int out_dim2 = reshaped->cols();
			int in_dim1 = reshaped->inner_dims().first;
			int in_dim2 = reshaped->inner_dims().second;

			if(out_dim1*in_dim1 != rows()*in_dims.first ||
						out_dim2*in_dim2 != cols()*in_dims.second){
				cerr << "Error: Reshape dimension is not correct!" << endl;
				return false;
			}else{
				MatrixXd expanded = expand();
				for(int o_row = 0; o_row < out_dim1; o_row++){ 
					for(int o_col = 0; o_col < out_dim2; o_col++){
						MatrixXd m(in_dim1, in_dim2);
						for(int i_row = 0; i_row < in_dim1; i_row++){
							for(int i_col = 0; i_col < in_dim2; i_col++){
								m(i_row, i_col) = expanded(o_row * in_dim1 + i_row,
											o_col*in_dim2 + i_col);
							}
						}
						reshaped->set_m(o_row, o_col, m);
					}
				}
				return true;
			}
		}

        /*
		 * Block-wise trace method.
		 * Sum along the block diagonal of the outer tensor.
		 * Return a matrix with same size as the bottom matrix
		 * */
		MatrixXd trace(){
			MatrixXd tt(in_dims.first, in_dims.second);
			tt.setZero();
			if(rows() != cols()){
				cerr << "Error: the tensor is not a square tensor, cannot compute trace!" << endl;
				return tt;
			}
			for(int ii = 0; ii < rows(); ii ++){
				tt += data.at(ii).at(ii);
			}
			return tt;
		}

        /**
		 * Retrieve one row of a tensor_m.
		 * Return as a tensor_m
		 */
		bool get_row(int i, Tensor_m* tensor_row){
			if(tensor_row->rows() != 1 || tensor_row->cols() != cols() ||
						tensor_row->inner_dims().first != in_dims.first||
						tensor_row->inner_dims().second != in_dims.second){
				cerr << "Error: the dimension of your row tensor is not correct!" << endl;
				return false;
			}else{
				for(int col = 0; col < cols(); col++){
					tensor_row->set_m(0, col, data.at(i).at(col));
				}
				return true;
			}
		}


        /**
		 * Expand the tensor_m into a large two-dimension matrix
		 */
		MatrixXd expand(){
			MatrixXd result(out_dims.first * in_dims.first, 
						out_dims.second * in_dims.second);
            int x_m_step = in_dims.first;
			int y_m_step = in_dims.second;

			for(int x_t = 0; x_t < out_dims.first; x_t ++){
				for(int y_t = 0; y_t < out_dims.second; y_t ++){
					for(int x_m = 0; x_m < x_m_step; x_m++){
						for(int y_m = 0; y_m < y_m_step; y_m++){
							result(x_t*x_m_step + x_m, y_t*y_m_step + y_m) =
								data.at(x_t).at(y_t)(x_m,y_m);
						}
					}
				}
			}
			return result;
		}

        /**
		 * Transpose the outer dimension of the tensor and the inner dimension 
		 * of the bottom matrix
		 */
		Tensor_m* transpose(){
			Tensor_m* result = new Tensor_m(out_dims.second, out_dims.first, 
						in_dims.second, in_dims.first);
			for(int x = 0; x < out_dims.first; x++){
				for(int y = 0; y < out_dims.second; y++){
					result->set_m(y, x, data.at(x).at(y).transpose());
				}
			}
			return result;
		}

		bool set_m(int i, int j, MatrixXd m){
			if(m.rows() != in_dims.first || m.cols() != in_dims.second){
				return false;
			}else{
				data.at(i).at(j) = m;
				return true;
			}
		}

		bool get_m(int i, int j, MatrixXd& re){
			if(i >= out_dims.first || i < 0 ||
						j >= out_dims.second || j < 0){
				return false;
			}else{
				re = data.at(i).at(j);
				return true;
			}
		}
		
		MatrixXd get_m(int i, int j){
			return data.at(i).at(j);
		}

	private:
		vector<vector<MatrixXd> > data;
	
};


/*
 * Tensor of tensors
 */
class Tensor_t : public Tensor{
    public:
	    Tensor_t(int h_dim_1, int h_dim_2, int i_dim_1, int i_dim_2):
	             Tensor(h_dim_1, h_dim_2, i_dim_1, i_dim_2){
			/** Innitialize data **/
		    Tensor* t = new Tensor_m(i_dim_1,i_dim_2, 0, 0);
			data.resize(h_dim_1, vector<Tensor*>(h_dim_2,t));
			type = Tensor :: V;
		}

		bool plus(Tensor* adder, Tensor* sum){
			return false;
		}

	    bool set_t(int i, int j, Tensor* t){
			if(t->inner_dims().first != in_dims.first ||
						t->inner_dims().second != in_dims.second){
				return false;
			}else{
				data.at(i).at(j) = t;
				return true;
			}
		}

		bool get_t(int i, int j, Tensor* re){
			if(i >= out_dims.first || i < 0 ||
						j >= out_dims.second || j < 0){
				return false;
			}else{
				re = data.at(i).at(j);
				return true;
			}
		}
	private:
		vector<vector<Tensor*> > data;
};

class F_COST{
	public:
		F_COST(Grid* _grid, vector<MatrixXd>& covariances):num_prec(covariances.size()){
			grid = _grid;
			loadObstacles();
			for(int i = 0; i < covariances.size(); i++){
				prec_matrices.push_back(covariances.at(i).transpose());
			}
		}

		int get_Num(){
			return num_prec;
		}

		void transform(pair<double, double>& origin, 
					pair<double, double>& target);

		MatrixXd F_first_order(VectorXd& state){
			MatrixXd m(num_prec,6);
			m.setZero();
			VectorXd xy(2);
			xy << state(0), state(1);

			for(int l = 0; l < num_prec; l++){
				VectorXd oneRow(2);
			    oneRow.setZero();
				for(int k = 0; k < num_obstacles; k++){
					double factor = - Gaussian::gaussian(xy, 
								transformed_u.at(k), prec_matrices.at(l));
					oneRow += factor * (xy-transformed_u.at(k));
				}
				oneRow = prec_matrices.at(l) * oneRow;
				m(l, 0) = oneRow(0);
				m(l, 1) = oneRow(1);
			}
			return m;
		}

		Tensor_v* F_second_order(VectorXd& state){
			Tensor_v* derivative = new Tensor_v(6,6,num_prec);
			VectorXd xy(2);
			xy << state(0), state(1);

			MatrixXd I(2,2);
			I.setIdentity();

			VectorXd V00(num_prec);
			VectorXd V01(num_prec);
			VectorXd V10(num_prec);
			VectorXd V11(num_prec);

			/** Compute the Jacobian matrices for along the third dimension **/
			for(int l = 0; l < num_prec; l++){
				MatrixXd jacobian(2,2);
				jacobian.setZero();
				MatrixXd temp(2,2);
				for(int k = 0; k < num_obstacles; k++){
					double factor = Gaussian::gaussian(xy, 
								transformed_u.at(k), prec_matrices.at(l));
					temp = (xy-transformed_u.at(k)) * (xy - transformed_u.at(k)).transpose();
					temp = I - temp * prec_matrices.at(l);
					jacobian += factor * temp;
				}
				jacobian = - prec_matrices.at(l) * jacobian;
                
				V00(l) = jacobian(0,0);
				V01(l) = jacobian(0,1);
				V10(l) = jacobian(1,0);
				V11(l) = jacobian(1,1);
			}

			derivative->set_v(0,0,V00);
			derivative->set_v(0,1,V01);
			derivative->set_v(1,0,V10);
			derivative->set_v(1,1,V11);

			return derivative;
		}


	private:
		Grid* grid;
		vector<MatrixXd> prec_matrices;
		int num_prec;
		int num_obstacles; 
		vector<pair<double, double> > raw_obstacles;
		vector<VectorXd> transformed_u; /** transformed obtacle vector in goal aligned coordinates */
		 
		void loadObstacles(){
			pair<int, int> dims = grid->dims();
			for(int x = 0; x < dims.first; x++){
				for(int y = 0; y < dims.second; y++){
					if(grid->at(x,y) == 1){ // obstacle
						pair<double, double> point = 
							grid->grid2Real(x,y);
						raw_obstacles.push_back(point);
					}
				}
			}
			num_obstacles = raw_obstacles.size();
			cout << "F_COST loads " << num_obstacles 
				 << " obstacle points in total." << endl;
		}


};


class Likelihood{
	public:
		/** Constructor **/
		Likelihood(F_COST* _f_cost, ContinuousState* _cs, M_6& _A, M_6_2& _B, M_6& _Sigma):
	    f_cost(_f_cost), cs(_cs), A(_A), B(_B), Sigma(_Sigma){
			/** Initialize other tensor_m */
		}

		~Likelihood(){
			if(J){
				delete J;
			}	
			if(g){
				delete g;
			}	
			if(H){
				delete H;
			}
		}
		
		/**
		 * Get g for locally optimal likelihoods
		 * g =\frac{ \pratial r }{\partial A}
		 */
		void get_g();
		/**
		 * Get H for locally optimal likelihoods
		 * H =\frac{ \pratial^2 r }{\partial A^2}
		 */
		void get_H();
        /**
                 * 2Txm
		 * Get the derivative of g wrt parameter theta
		 * get \frac{\partial g}{\partial \theta}
		 * */
		MatrixXd get_g_wrt_theta();
		/**
                 * 2Tx2T(mx1)
		 * Get the derivative of H wrt parameter theta
		 * get \frac{\partial H}{\partial \theta}
		 * */
		Tensor_m* get_H_wrt_theta();
		/**
                 * 2Tx1(6X6)
		 * Get the derivative of g wrt parameter M
		 * get \frac{\partial g}{\partial M}
		 * */
		Tensor_m* get_g_wrt_M();
		/**
                 * TxT(12X12)
		 * Get the derivative of H wrt parameter theta
		 * get \frac{\partial H}{\partial M}
		 * */
		Tensor_m* get_H_wrt_M();
      
		/*
		 * Get the derivative of the likelihood wrt M
		 */
		MatrixXd get_L_wrt_M();

		/*
		 * Get the derivative of the likelihood wrt theta
		 */
		VectorXd get_L_wrt_theta();

        /*
		 * Get the likelihoods give a sequnce of states
		 */
		double get_likelihood(vector<pair<double, double> >& rawData, 
					pair<double, double>& target);


		void setM(MatrixXd& _M){ 
			M =_M;
		}
		void setA(M_6& _A){
			A =_A;
		}
		void setB(M_6_2& _B){
			B =_B;
		}
		void setSigma(M_6& _S){ 
			Sigma =_S;
		}
		void setTheta(VectorXd& _Theta){
			Theta =_Theta;
		}
		MatrixXd& getM(){ return M;}
		M_6& getA(){ return A;}
		M_6_2& getB(){ return B;}
		M_6& getSigma(){ return Sigma;}
        VectorXd& getTheta(){ return Theta;}
		
		void setStates(vector<pair<double, double> >& rawData, 
					pair<double, double>& target);

	private:
		M_6& A;
		M_6_2& B;
        M_6& Sigma;
		MatrixXd M;
		VectorXd Theta;
	    /* 
		 * Compute parts of g and H
		 * */
		Tensor_v* g_hat();
		Tensor_v* g_tilde();
		Tensor_m* H_hat();
		Tensor_m* H_tilde();

		vector<VectorXd> states;
		vector<VectorXd> actions;

		Tensor_v* g;
		Tensor_m* H;
		Tensor_m* J;
		F_COST* f_cost;
		ContinuousState* cs;

		void extract_actions();
		void compute_J();

};

class LocalEOptimizer{
        public:
            LocalEOptimizer(Likelihood* _L, Evidence& _evid):
				L(_L), evid(_evid){
				cout << "Optimize over: " << evid.size() 
					 << " examples." <<endl;
			}
			//gd method, it will stop when gradient is almost 0
            void gradientDescent(double stepsize=0.1, double eps=10, int max_iter=1000);
        private:
			Likelihood* L;
			Evidence& evid;
}; 

class LocalIOCPredictor{
	public:
		LocalIOCPredictor(Grid& _grid, Likelihood* _L):grid(_grid),
	    L(_L){
			dims = grid.dims();
			for(int x=0;x<dims.first;x++){
				for(int y = 0;y<dims.second;y++){
					pair<double,double> realVals = 
						grid.grid2Real(x,y);
					mapping[make_pair(x,y)] = realVals;
				}
			}
			assert(mapping.size()==dims.first*dims.second);
			prior.resize(dims.first,vector<double>(dims.second,
						-HUGE_VAL));
			posterior.resize(dims.first,vector<double>(dims.second,
							-HUGE_VAL));
			gridLikelihoods.resize(dims.first,
						vector<double>(dims.second,-HUGE_VAL));

		}
		
		void predictAll(vector<pair<double,double> >& observation, int index);
		vector<vector<double> >& getPosterior(){ return posterior; }
		vector<vector<double> >& getLikelihoods(){ 
			return gridLikelihoods;}
        void setPrior(vector<vector<double> > &_prior){
			prior = _prior;
		};

	private:
		Grid& grid;
		Likelihood* L;
		vector<vector<double> > prior, posterior, gridLikelihoods;
		map<pair<int,int>,pair<double,double> > mapping;
		pair<int,int> dims;
;
};
#endif
