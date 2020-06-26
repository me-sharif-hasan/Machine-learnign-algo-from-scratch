#include <bits/stdc++.h>
#include "csv.h"
using namespace rapidcsv;
using namespace std;
class MultiClassLogisticRegression{
public:
    double alpha=0.01;
    vector <vector<double>> features;
    vector <double> cls;
    vector <double> originalClass;
    vector <double> theta;
    vector <pair<double,vector<double>> > rec;
    MultiClassLogisticRegression(vector <vector<double >> x,vector <double> y,double lr=0.01){
        for(int i=0;i<x.size();i++){
            x[i].insert(x[i].begin(),1);
        }
        features=x;
        originalClass=y;
        alpha=lr;
    }
    double cost(){
            double sum=0;
            for(int i=0;i<features.size();i++){
                double sig=hypothesis(features[i]);
                sum+=(-cls[i]*log(sig)-(1-cls[i])*log(1-sig));
            }
            return sum/features.size();
    }
    double hypothesis(vector <double> featureVector){
        double sum=0;
        for(int i=0;i<featureVector.size();i++){
            sum+=theta[i]*featureVector[i];
        }
        return 1.0/(1+exp(-sum));
    }
    void gradientDescentStep(){
        vector <double> tmp=theta;
        for(int j=0;j<theta.size();j++){
            double grd=0;
            for(int i=0;i<features.size();i++){
                grd+=(hypothesis(features[i])-cls[i])*features[i][j];
            }
            tmp[j]=theta[j]-(alpha/features.size())*grd;
        }
        theta=tmp;
    }
    void run(){
        for(int i=0;i<100000;i++){
            gradientDescentStep();
            if(i%1000==0) cout<<cost()<<endl;
        }
    }
    void train(vector <double> classList){
        if(features.size()==0){
            cout<<"no features\n";
            return;
        }
        vector<double> z(features[0].size(),0);
        for(int i=0;i<classList.size();i++){
            int c=classList[i];
            cls.clear();
            for(int j=0;j<originalClass.size();j++){
                if(abs(originalClass[j]-c)<0.000001){
                    cls.push_back(1);
                }else{
                    cls.push_back(0);
                }
            }
            theta=z;
            run();
            rec.push_back({c,theta});
            cout<<"tht in "<<c<<"= ";
            for(auto a:theta){
                cout<<a<<" ";
            }
            cout<<endl;
        }
    }
    pair <double,double> predict(vector <double> vp){
        vp.insert(vp.begin(),1);
        double mx=0,c=0;
        for(int i=0;i<rec.size();i++){
            theta=rec[i].second;
            double prediction=hypothesis(vp);
            theta.clear();
            if(prediction>mx){
                c=rec[i].first;
                mx=prediction;
            }
        }
        return {c,mx};
    }

};
int main() {
    vector<vector<double>> in;
    vector <double> cls;
    Document dataset("../dataset.csv");
    vector<double> x1=dataset.GetColumn<double>("x1");
    vector<double> x2=dataset.GetColumn<double>("x2");
    vector<double> y=dataset.GetColumn<double>("y");
    for(int i=0;i<x1.size();i++){
        vector <double> k={x1[i],x2[i]};
        in.push_back(k);
        cls.push_back(y[i]);
    }
    MultiClassLogisticRegression mc(in,cls,.001);
    mc.train({0,1,2});

    Document test("../test.csv");
    vector<double> $x1=test.GetColumn<double>("x1");
    vector<double> $x2=test.GetColumn<double>("x2");
    vector<double> $y=test.GetColumn<double>("y");

    in.clear();
    y.clear();

    for(int i=0;i<$x1.size();i++){
        in.push_back({$x1[i],$x2[i]});
        y.push_back($y[i]);
    }

    int i=0,k=0;
    for(auto a:in){
        pair <double,double> dk=mc.predict(a);
        if(dk.first==y[i])k++;
        cout<<"probability= "<<dk.second<<" to be class "<<dk.first<<":"<<y[i]<<(dk.first==y[i]?" ***, "+to_string(k):"")<<endl;
        i++;
    }
    cout<<"total "<<k<<" correct of "<<in.size()<<" total";
    return 0;
}
