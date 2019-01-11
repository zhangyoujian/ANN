#pragma once
#include"stdafx.h"
#include"qobject.h"
#include"qdialog.h"
#include"QTime" 

class Sigmoid
{
public:
	Sigmoid();
	~Sigmoid();
	const mat FeedForward(const mat featureMap,double &wsum);
	const mat backpropagation(const mat delta);
private :
	mat z;
	mat cache;
	mat delat_X;
};

class Tanh
{
public:
	Tanh();
	~Tanh();
	const mat FeedForward(const mat featureMap, double &wsum);
	const mat backpropagation(const mat delta);
private:
	mat z;
	mat cache;
	mat delat_X;
};

class Relu
{
public:
	Relu();
	~Relu();
	const mat FeedForward(const mat featureMap, double &wsum);
	const mat backpropagation(const mat delta);
private:
	mat z;
	mat cache;
	mat delat_X;
};

class LinearLayer
{
public:
	LinearLayer();
	~LinearLayer();
	const mat FeedForward(const mat featureMap, double &wsum);
	const mat backpropagation(const mat output);

private:
	mat cache;
	mat z;
	mat delta_X;

};

typedef struct H_PARA {

	QString method="";
	double alpha = 0.001;
	double mu = 0.9;
	double zeta = 0.01;
	int batch_size=1;
	double beta1=0.9;
	double beta2=0.999;

}PARAMS;

class Fullconnectlayer 
{
public:
	Fullconnectlayer(vector<int> kernel_size);
	~Fullconnectlayer();
	const mat FeedForward(const mat X, double &wsum);
	const mat backpropagation(const mat delta,double &delta_sum);
	void update_kernel(PARAMS parames);


	//===================================”√”⁄»´æ÷—µ¡∑=========================================
	void update_weight(vec w_int);
	vec backpropagation(PARAMS params);
	vec output_weight();
	//================================================================================

	vector<int>layerSize;
private :
	mat cache;
	mat activations;
	mat delta_X;
	mat delta_w;
	mat delta_b;
private:
	mat weight;
	mat bias;
	mat gradient_history;
	mat bias_history;
	mat m_kernel;
	mat m_bias;
	mat v_kernel;
	mat v_bias;
	int  timestamp;
};

typedef struct H_PS {
	QString name;
	int xcol;
	rowvec xmax;
	rowvec xmin;
	rowvec xrange;
	int ycol;
	double ymax;
	double ymin;
	double yrange;
	bool n_change;
	rowvec gain;
	rowvec xoffset;
}PS;




class NeuralNetWork : public QObject
{
	Q_OBJECT
public:
	NeuralNetWork(const cube& X, const mat& Y, int item,QObject *parent = 0);
	NeuralNetWork(int item, QObject *parent = NULL);
	NeuralNetWork(const mat X,const mat Y, int item, QObject *parent = NULL);
	bool createMode(int inputSize,QString active_fun = "Tanh", int HideLayer = 2, int epoch = 1500, double lambda = 0, double learningRate = 0.001);
	bool save();
	bool load();


	~NeuralNetWork();
	void predict();
	mat predict(const mat Xtest);


public slots:
	void train();
	void trainGlobal();  //wolf—µ¡∑

signals:
	void SendCurrentProgress(int, int);
	void SendTimeElapse(double times);
	void SendTrainingFinished();
private:
	mat mapminmax(PS &output,const mat X, double minx=0, double maxx=1,QString name = "mapminmax");
	bool convertTomat(const cube xdata, const vec target);
	void shuffle_index(int len, int *a);
	static double max(double a, double b);
	static double min(double a, double b);
	
	 mat FeedForward(const mat X, double &wsum);
	 double loss_function(mat pred, mat y, const double wsum);
	 void backpropagation(const mat Y,double &delta_sum);
	 void update_parameters();



	 vec fmincg(vec w_init);//wolf—µ¡∑
	 double costFun(vec weight, vec &grad_out); //wolf—µ¡∑
	 void update_weight(vec w_init);//wolf—µ¡∑
	 vec output_weight();//wolf—µ¡∑
	 vec backpropagation();//wolf—µ¡∑


private:
	int currentItem;
	QFile *xmlfile;
private:
	mat Null;

	int n_Sample;
	int varsX;
	int varsY;
	//—µ¡∑ºØ ˝æ›
	mat traingX;
	mat traingY;


	//! Observations.
	const cube &X;
	//! Predictions.
	const mat &Y;

private:
	PARAMS params;
	int maxiters;
	QTime timer;

	QString active_fun;  //º§ªÓ∫Ø ˝¿‡–Õ
	vector<QString> type;  //
	vector<void*> nodes;
	int InputSize;
	int hideLayerSize;

	PS ps_in;
	PS ps_out;
};



