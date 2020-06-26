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
    vector <double> x1,x2,x3,x4;vector <string>y;
    Document doc("../iris.csv");
    x1=doc.GetColumn<double>("sepal length");
    x2=doc.GetColumn<double>("sepal width");
    x3=doc.GetColumn<double>("petal length");
    x4=doc.GetColumn<double>("petal length");
    y=doc.GetColumn<string>("class");

    vector <vector<double>> features;
    vector <double> cls;
    for(int i=0;i<x1.size();i++){
        features.push_back({x1[i],x2[i],x3[i],x4[i]});
        if(y[i]=="Iris-setosa"){
            cls.push_back(0);
            cout<<"--added class 0\n";
        }else if(y[i]=="Iris-versicolor"){
            cls.push_back(1);
            cout<<"--added class 1\n";
        }else{
            cls.push_back(2);
            cout<<"--added class 2\n";
        }
        cout<<y[i]<<endl;
    }
    cout<<"---dataset size= "<<cls.size()<<endl;
    MultiClassLogisticRegression mlc(features,cls,0.01);
    mlc.train({0,1,2});
    int k=0;
    for(int i=0;i<x1.size();i++){
        pair<double,double> dk=mlc.predict({x1[i],x2[i],x3[i],x4[i]});
        cout<<"Algo predicting class: "<<dk.first<<"/ originally: "<<cls[i]<<"; "<<(dk.first==cls[i]?"****":"")<<endl;
        if(dk.first==cls[i]) k++;
    }
    cout<<k;
    return 0;
}
