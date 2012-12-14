#include<vector>
#include<iostream>
#include<cstdlib>

using namespace std;
/*
class Matrix{
	public:
		~Matrix(){
			A->clear();
			delete A;
	        cout<<"release ptr"<<endl;
			sleep(10);
		}
		void assign(){
			A = new vector<vector<vector<vector<double> > > > (10,                 
			vector<vector<vector<double> > >(
					 500,vector<vector<double> >(
					 500,vector<double>(5,0.0))) );
	        cout<<"assign"<<endl;
			sleep(10);
		}
	private:
		vector<vector<vector<vector<double> > > >* A;
};
*/

class Data{
	public:
		Data(double _a, double _b, double _c):a(_a),b(_b),c(_c)
	    {}
	private:
		double a;
		double b;
		double c;
};



class Matrix{
	public:
		~Matrix(){
			A.clear();
			B.clear();
			C.clear();
			D.clear();
	        cout<<"release ref"<<endl;
			sleep(5);
		}

        void process(){
			vector<double> temp(100,0.0);
			assignB();

        	D.resize(500,vector<vector<double> > 
						(300,vector<double> (3,1.0)));
			cout<<"assign D"<<endl;
			
			assignA();
            
			C.resize(10,vector<vector<vector<double> > > (
					 500,vector<vector<double> >(
					 300,vector<double>(3,0.0))));
			cout<<"assign C"<<endl;

			cout<<"Data vector"<<endl;
			sleep(5);
			vector<Data> order;
			for (int count=0;count<10;count++){
			for (int x=0;x<D.size();x++){
				for (int y=0;y<D.at(0).size();y++){
					for (int v=0;v<D.at(0).at(0).size();v++){
						Data s(x,y,v);
						order.push_back(s);
					}
				}
			}
			}
		    sleep(5);
			cout<<"push finished"<<endl;
			order.clear();


		}

		void assignD(){
			D.resize(500,vector<vector<double> > 
						(300,vector<double> (3,1.0)));
			cout<<"assign D"<<endl;
		}

		void assignA(){
			A.resize(10,vector<vector<vector<double> > > (
					 500,vector<vector<double> >(
					 300,vector<double>(3,0.0)))); 
			A.at(0) = D;
			cout<<"assign A"<<endl;

		}

		void assignB(){
			B.resize(10,vector<vector<vector<double> > > (
					 500,vector<vector<double> >(
					 300,vector<double>(3,0.0))));
			cout<<"assign B"<<endl;
 
		}


	private:
		vector<vector<vector<vector<double> > > > A;
		vector<vector<vector<vector<double> > > > B;
		vector<vector<vector<vector<double> > > > C;
		vector<vector<vector<double> > >  D;
};

void eval(){
	vector<double> temp(100,0.0);
	for (int i = 0;i<10;i++)
	{
		Matrix* m = new Matrix();
		m->process();
		delete m;
	}

}

void op(Data& _d){
}

int main(int argc,char **argv){
    Data* d = new Data(0,0,0);
    op(*d);
	for (int j=0;j<10;j++)
	  eval();

	exit(0);
}
