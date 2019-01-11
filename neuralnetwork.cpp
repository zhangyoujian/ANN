#include "neuralnetwork.h"


Sigmoid::Sigmoid()
{
	z.clear();
	cache.clear();
	delat_X.clear();
}
Sigmoid::~Sigmoid()
{
	z.clear();
	cache.clear();
}

const mat Sigmoid::FeedForward(const mat X, double &wsum)
{
	z.clear();
	cache.clear();
	cache = X;
	z = 1.0 / (1.0 + arma::exp(-X));

	if((&wsum)!=NULL)
		wsum = 0;
	return z;
}
const mat Sigmoid::backpropagation(const mat delta)
{
	delat_X.clear();
	delat_X = delta % z % (1.0 - z);
	return delat_X;
}

//===============================================================================
Tanh::Tanh()
{
	z.clear();
	cache.clear();
	delat_X.clear();
}
Tanh::~Tanh()
{
	z.clear();
	cache.clear();
	delat_X.clear();
}

const mat Tanh::FeedForward(const mat X, double &wsum)
{
	z.clear();
	cache.clear();
	cache = X;
	z = (exp(X) - exp(-X)) / (exp(X) + exp(-X));
	if ((&wsum) != NULL)
		wsum = 0;
	return z;
}
const mat Tanh::backpropagation(const mat delta)
{
	delat_X.clear();
	delat_X = delta % (1.0 - z % z);
	return delat_X;
}

Relu::Relu()
{
	z.clear();
	cache.clear();
	delat_X.clear();
}
Relu::~Relu()
{
	z.clear();
	cache.clear();
	delat_X.clear();
}

const mat Relu::FeedForward(const mat X, double &wsum)
{
	z.clear();
	cache.clear();
	cache = X;

	z = zeros(X.n_rows, X.n_cols);
	for (int i = 0; i<(int)X.n_rows; i++)
		for (int j = 0; j < (int)X.n_cols; j++)
		{
			if (X(i, j) >=0)
				z(i, j) = X(i, j);
		}

	if ((&wsum) != NULL)
		wsum = 0;

	return z;
}


const mat Relu::backpropagation(const mat delta)
{
	delat_X.clear();
	delat_X = zeros(delta.n_rows, delta.n_cols);
	for (int i = 0; i<(int)delta.n_rows; i++)
		for (int j = 0; j < (int)delta.n_cols; j++)
		{
			if (cache(i, j) >=0)
				delat_X(i, j) = 1;
		}

	delat_X = delat_X%delta;
	return delat_X;
}

LinearLayer::LinearLayer()
{
	cache.clear();
	delta_X.clear();
	z.clear();
}
LinearLayer::~LinearLayer()
{
	cache.clear();
	delta_X.clear();
	z.clear();
}
	
const mat LinearLayer::FeedForward(const mat featureMap, double &wsum)
{
	cache.clear();
	cache = featureMap;
	z = featureMap;
	if ((&wsum) != NULL)
		wsum = 0;
	return z;
}
const mat LinearLayer::backpropagation(const mat output)
{
	delta_X.clear();
	delta_X = (z - output) / output.n_rows;
	return delta_X;
}




Fullconnectlayer::Fullconnectlayer(vector<int> kernel_size)
{
	layerSize.clear();
	layerSize = kernel_size;
	qsrand(QTime(0, 0, 0).secsTo(QTime::currentTime()));
	double f= sqrt(6) / (sqrt(kernel_size.at(0) + kernel_size.at(1)));
	double epsilon = 1e-6;
	weight = randu(kernel_size.at(0), kernel_size.at(1)) * 2 * f - f+ epsilon;
	bias = randu(1,kernel_size.at(1)) * 2.0 * f - f + epsilon;
	gradient_history = zeros(kernel_size.at(0), kernel_size.at(1));
	bias_history = zeros(1,kernel_size.at(1));

	m_kernel = zeros(kernel_size.at(0), kernel_size.at(1));
	m_bias = zeros(1, kernel_size.at(1));
	v_kernel = zeros(kernel_size.at(0), kernel_size.at(1));
	v_bias = zeros(1, kernel_size.at(1));
	timestamp = 0;

	cache.clear();
	activations.clear();
	delta_X.clear();
	delta_w.clear();
	delta_b.clear();
	
}

Fullconnectlayer::~Fullconnectlayer()
{
	

}

const mat Fullconnectlayer::FeedForward(const mat X, double &ws)
{
	cache.clear();
	activations.clear();
	cache = X;
	activations = X*weight + repmat(bias, X.n_rows, 1);
	if ((&ws) != NULL)
	{
		mat wsum;
		wsum.clear();
		wsum = square(weight);
		ws = sum(sum(wsum));
	}
	return activations;
}
const mat Fullconnectlayer::backpropagation(const mat delta, double &delta_sum)
{
	delta_X.clear();
	delta_X = delta * weight.t();
	delta_w = cache.t()*delta;
	delta_b = sum(delta, 0);

	if ((&delta_sum)!=NULL)
	{
		delta_sum = sum(sum(square(delta_w))) + sum(sum(square(delta_b)));
	}
		
	return delta_X;
}

vec Fullconnectlayer::backpropagation(PARAMS params)
{
	mat w = delta_w + params.zeta*weight / params.batch_size;
	vec out = reshape(w, w.n_elem, 1);
	out = join_cols(out, reshape(delta_b, delta_b.n_elem, 1));
	return out;
}

vec Fullconnectlayer::output_weight()
{
	mat w = weight;
	vec out = reshape(w, w.n_elem, 1);
	out = join_cols(out, reshape(bias, bias.n_elem, 1));
	return out;
}

void Fullconnectlayer::update_weight(vec w_int)
{
	vec w = w_int.rows(0,layerSize.at(0)*layerSize.at(1)-1);
	vec b = w_int.rows(layerSize.at(0)*layerSize.at(1), layerSize.at(0)*layerSize.at(1)+ layerSize.at(1)-1);
	this->weight = reshape(w, layerSize.at(0), layerSize.at(1));
	this->bias = reshape(b, 1, layerSize.at(1));
}

void Fullconnectlayer::update_kernel(PARAMS parames)
{
	QString method = parames.method;
	double alpha = parames.alpha;
	double mu = parames.mu;
	double zeta = parames.zeta;
	int batch_size = parames.batch_size;
	double beta1 = parames.beta1;
	double beta2 = parames.beta2;
	double   fudge_factor = 1e-8;

	if (method == "adagrad")
	{
		gradient_history += square(delta_w + (zeta*weight / batch_size));
		bias_history += square(delta_b);

		weight -= (
					alpha*(delta_w + (zeta*weight / batch_size)) %
					(1.0 / (sqrt(gradient_history)+ fudge_factor))
				);

		bias-= (
				alpha*(delta_b) %
				(1.0 / (sqrt(bias_history) + fudge_factor))
			   );
	}
	else if (method == "gd_momentum")
	{
		mat new_delta_K = alpha*(delta_w + (zeta*weight / batch_size)) + mu *gradient_history;
		mat new_delta_b = alpha * delta_b + mu *bias_history;
		weight -= new_delta_K;
		bias -= new_delta_b;
		gradient_history = delta_w + (zeta*weight / batch_size);
		bias_history = delta_b;
	}
	else if (method == "adam")
	{
		timestamp += 1;
		alpha = alpha*sqrt(1 - pow(beta2, timestamp)) / (1 - pow(beta1, timestamp));
		m_kernel = beta1*m_kernel + (1 - beta1)*(delta_w + (zeta * weight / batch_size));
		m_bias = beta1 * m_bias + (1 - beta1) * delta_b;
		v_kernel = beta2 * v_kernel + (1 - beta2) * square(
			(delta_w + (zeta *weight / batch_size)));
		v_bias = beta2 * v_bias + (1 - beta2) *square(delta_b);

		weight -= (
				(alpha * m_kernel) %
				(1.0 / (sqrt(v_kernel) + fudge_factor))
			);
		bias-= (
			(alpha * m_bias) %
			(1.0 / (sqrt(v_bias) + fudge_factor))
			);
	}
	else
	{
		weight -= alpha * (delta_w + zeta * weight / batch_size);
		bias -= alpha * delta_b;
	}
}


bool NeuralNetWork::createMode(int inputSize,QString active_fun, int HideLayer, int epoch , double lambda , double learningRate)
{
	if (HideLayer < 1)
		return false;
	this->active_fun = active_fun;
	params.zeta = lambda;
	params.alpha = learningRate;
	params.method = "adam";
	maxiters = epoch;
	this->InputSize = inputSize;
	this->hideLayerSize = HideLayer;
	int NodeSize = 2 * inputSize + 1;  //隐藏层神经元个数

	for (int i = 0; i < nodes.size(); i++)
	{
		void* node = nodes.at(i);
		if (node)
		{
			delete node;
		}
	}
	nodes.clear();
	type.clear();

	vector<int> layer;
	layer.clear();

	layer.push_back(inputSize);
	layer.push_back(NodeSize);
	

	Fullconnectlayer *fullyLayerFirst = new Fullconnectlayer(layer);

	nodes.push_back(fullyLayerFirst);
	type.push_back("Fullconnectlayer");

	if (active_fun == "Relu")
	{
		Relu *layer2 = new Relu();
		nodes.push_back(layer2);
		type.push_back("Relu");
	}
	else if (active_fun == "Tanh")
	{
		Tanh *layer2 = new Tanh();
		nodes.push_back(layer2);
		type.push_back("Tanh");
	}
	else
	{
		Sigmoid *layer2 = new Sigmoid();
		nodes.push_back(layer2);
		type.push_back("Sigmoid");
	}
	//=========================================中间隐含层=========================================================	

	for (int i = 1; i < HideLayer; i++)
	{
		layer.clear();
		layer.push_back(NodeSize);
		layer.push_back(NodeSize);
		Fullconnectlayer *HideFullLayer = new Fullconnectlayer(layer);
		nodes.push_back(HideFullLayer);
		type.push_back("Fullconnectlayer");

		if (active_fun == "Relu")
		{
			Relu *HideLayer = new Relu();
			nodes.push_back(HideLayer);
			type.push_back("Relu");
		}
		else if (active_fun == "Tanh")
		{
			Tanh *HideLayer = new Tanh();
			nodes.push_back(HideLayer);
			type.push_back("Tanh");
		}
		else
		{
			Sigmoid *HideLayer = new Sigmoid();
			nodes.push_back(HideLayer);
			type.push_back("Sigmoid");
		}
	}

	//==========================================================================================================
	
	layer.clear();
	layer.push_back(NodeSize);
	layer.push_back(1);

	Fullconnectlayer *fullyLayerLast = new Fullconnectlayer(layer);
	nodes.push_back(fullyLayerLast);
	type.push_back("Fullconnectlayer");

	LinearLayer *outLayer = new LinearLayer();
	nodes.push_back(outLayer);
	type.push_back("LinearLayer");
	return true;
}


NeuralNetWork::NeuralNetWork(const cube& X, const mat& Y, int item, QObject *parent)
	:X(X),Y(Y),QObject(parent)
{
	if (X.n_slices != Y.n_rows)
	{
		QMessageBox::warning(NULL, "Error", "spectrum X not match concentration Y");
		return;
	}
	currentItem = item;
	n_Sample = X.n_slices;
	varsX = X.slice(0).n_cols;
	varsY = Y.n_cols;
	type.clear();
	nodes.clear();
	convertTomat(X,Y.col(currentItem));
	xmlfile = new QFile();
}

NeuralNetWork::NeuralNetWork(const mat X, const mat Y, int item, QObject *parent)
:X(cube(0, 0, 0)), Y(Null),QObject(parent)
{
	currentItem = item;
	n_Sample = X.n_rows;
	params.batch_size = n_Sample;
	varsX = X.n_cols;
	varsY = Y.n_cols;
	type.clear();
	nodes.clear();
	traingX = X;
	traingY = Y;
	traingX = mapminmax(ps_in, traingX, -1, 1);
	traingY = mapminmax(ps_out, traingY, -1, 1);
	xmlfile = new QFile();
}

NeuralNetWork::NeuralNetWork(int item, QObject *parent)
:X(cube(0, 0, 0)), Y(Null), QObject(parent)
{
	currentItem = item;
	type.clear();
	nodes.clear();
	xmlfile = new QFile();
}

NeuralNetWork::~NeuralNetWork()
{
	for (int i = 0; i < nodes.size(); i++)
	{
		void* node = nodes.at(i);
		if (node)
		{
			delete node;
		}
	}
	type.clear();
	if (xmlfile)
	{
		delete xmlfile;
		xmlfile = NULL;
	}
}

bool NeuralNetWork::save()
{
	if (xmlfile == NULL)
		return false;
	QDomDocument doc;
	QDomProcessingInstruction instruction = doc.createProcessingInstruction("xml", "version=\"1.0\" encoding=\"UTF-8\"");
	doc.appendChild(instruction);  //添加标题
    //=======================创建根结点======================
	QDomElement root = doc.createElement("BPnet_storage");
	doc.appendChild(root);

	QDomElement ParaNode = doc.createElement("parameter");


	QDomElement InputNode = doc.createElement("InputSize");
	QDomText InputSizeData = doc.createTextNode(QString::number(this->InputSize));
	InputNode.appendChild(InputSizeData);
	ParaNode.appendChild(InputNode);


	QDomElement ActiveFunNode = doc.createElement("ActiveFun");
	QDomText ActiveFunStr = doc.createTextNode(this->active_fun);
	ActiveFunNode.appendChild(ActiveFunStr);
	ParaNode.appendChild(ActiveFunNode);

	QDomElement HideNode = doc.createElement("HideLayer");
	QDomText HideLayerData = doc.createTextNode(QString::number(this->hideLayerSize));
	HideNode.appendChild(HideLayerData);
	ParaNode.appendChild(HideNode);

	QDomElement EpochNode = doc.createElement("Epoch");
	QDomText EpochData = doc.createTextNode(QString::number(this->maxiters));
	EpochNode.appendChild(EpochData);
	ParaNode.appendChild(EpochNode);

	QDomElement LambdaNode = doc.createElement("Lambda");
	QDomText LambdaData = doc.createTextNode(QString::number(this->params.zeta));
	LambdaNode.appendChild(LambdaData);
	ParaNode.appendChild(LambdaNode);

	QDomElement AlphaNode = doc.createElement("alpha");
	QDomText AlphaData = doc.createTextNode(QString::number(this->params.alpha));
	AlphaNode.appendChild(AlphaData);
	ParaNode.appendChild(AlphaNode);


	QDomElement ItemNode = doc.createElement("Item");
	QDomText ItemData = doc.createTextNode(QString::number(this->currentItem));
	ItemNode.appendChild(ItemData);
	ParaNode.appendChild(ItemNode);
	root.appendChild(ParaNode);

	//=================================下一个结点=============================================
	QString datasequence;
	datasequence.clear();

	QDomElement PSNode = doc.createElement("PS");
	QDomElement PSInNode = doc.createElement("PS_IN");


	QDomElement PsInNameNode = doc.createElement("name");
	QDomText PSInNameData = doc.createTextNode(this->ps_in.name);
	PsInNameNode.appendChild(PSInNameData);
	PSInNode.appendChild(PsInNameNode);


	QDomElement PsInxcolNode = doc.createElement("xcol");
	QDomText PsInxcolData = doc.createTextNode(QString::number(this->ps_in.xcol));
	PsInxcolNode.appendChild(PsInxcolData);
	PSInNode.appendChild(PsInxcolNode);


	QDomElement PsInxmaxNode = doc.createElement("xmax");
	datasequence.clear();
	for (int i = 0; i < this->ps_in.xmax.n_elem; i++)
	{
		if (i == (int)this->ps_in.xmax.n_elem - 1)
		{
			datasequence = datasequence + QString::number(ps_in.xmax(i)) + "\0";
		}
		else
		{
			datasequence = datasequence + QString::number(ps_in.xmax(i)) + " ";
		}
	}
	QDomText PsInxmaxData = doc.createTextNode(datasequence);
	PsInxmaxNode.appendChild(PsInxmaxData);
	PSInNode.appendChild(PsInxmaxNode);


	QDomElement PsInxminNode = doc.createElement("xmin");
	datasequence.clear();
	for (int i = 0; i < this->ps_in.xmin.n_elem; i++)
	{
		if (i == (int)this->ps_in.xmin.n_elem - 1)
		{
			datasequence = datasequence + QString::number(ps_in.xmin(i)) + "\0";
		}
		else
		{
			datasequence = datasequence + QString::number(ps_in.xmin(i)) + " ";
		}
	}
	QDomText PsInxminData = doc.createTextNode(datasequence);
	PsInxminNode.appendChild(PsInxminData);
	PSInNode.appendChild(PsInxminNode);


	QDomElement PsInxrangeNode = doc.createElement("xrange");
	datasequence.clear();
	for (int i = 0; i < this->ps_in.xrange.n_elem; i++)
	{
		if (i == (int)this->ps_in.xrange.n_elem - 1)
		{
			datasequence = datasequence + QString::number(ps_in.xrange(i)) + "\0";
		}
		else
		{
			datasequence = datasequence + QString::number(ps_in.xrange(i)) + " ";
		}
	}
	QDomText PsInxrangeData = doc.createTextNode(datasequence);
	PsInxrangeNode.appendChild(PsInxrangeData);
	PSInNode.appendChild(PsInxrangeNode);


	QDomElement PsInycolNode = doc.createElement("ycol");
	QDomText PsInycolData = doc.createTextNode(QString::number(this->ps_in.ycol));
	PsInycolNode.appendChild(PsInycolData);
	PSInNode.appendChild(PsInycolNode);

	QDomElement PsInymaxNode = doc.createElement("ymax");
	QDomText PsInymaxData = doc.createTextNode(QString::number(this->ps_in.ymax));
	PsInymaxNode.appendChild(PsInymaxData);
	PSInNode.appendChild(PsInymaxNode);

	QDomElement PsInyminNode = doc.createElement("ymin");
	QDomText PsInyminData = doc.createTextNode(QString::number(this->ps_in.ymin));
	PsInyminNode.appendChild(PsInyminData);
	PSInNode.appendChild(PsInyminNode);

	QDomElement PsInyrangeNode = doc.createElement("yrange");
	QDomText PsInyrangeData = doc.createTextNode(QString::number(this->ps_in.yrange));
	PsInyrangeNode.appendChild(PsInyrangeData);
	PSInNode.appendChild(PsInyrangeNode);


	QDomElement PsIngainNode = doc.createElement("gain");
	datasequence.clear();
	for (int i = 0; i < this->ps_in.gain.n_elem; i++)
	{
		if (i == (int)this->ps_in.gain.n_elem - 1)
		{
			datasequence = datasequence + QString::number(ps_in.gain(i)) + "\0";
		}
		else
		{
			datasequence = datasequence + QString::number(ps_in.gain(i)) + " ";
		}
	}
	QDomText PsIngainData = doc.createTextNode(datasequence);
	PsIngainNode.appendChild(PsIngainData);
	PSInNode.appendChild(PsIngainNode);


	QDomElement PsInxoffsetNode = doc.createElement("xoffset");
	datasequence.clear();
	for (int i = 0; i < this->ps_in.xoffset.n_elem; i++)
	{
		if (i == (int)this->ps_in.xoffset.n_elem - 1)
		{
			datasequence = datasequence + QString::number(ps_in.xoffset(i)) + "\0";
		}
		else
		{
			datasequence = datasequence + QString::number(ps_in.xoffset(i)) + " ";
		}
	}
	QDomText PsInxoffsetData = doc.createTextNode(datasequence);
	PsInxoffsetNode.appendChild(PsInxoffsetData);
	PSInNode.appendChild(PsInxoffsetNode);
	PSNode.appendChild(PSInNode);


	//==============================因变量归一化因子=======================================
	QDomElement PSOutNode = doc.createElement("PS_out");
	QDomElement PSOutNameNode = doc.createElement("name");
	QDomText PSOutNameData = doc.createTextNode(this->ps_out.name);
	PSOutNameNode.appendChild(PSOutNameData);
	PSOutNode.appendChild(PSOutNameNode);


	QDomElement PSOutxcolNode = doc.createElement("xcol");
	QDomText PSOutxcolData = doc.createTextNode(QString::number(this->ps_out.xcol));
	PSOutxcolNode.appendChild(PSOutxcolData);
	PSOutNode.appendChild(PSOutxcolNode);


	QDomElement PSOutxmaxNode = doc.createElement("xmax");
	datasequence.clear();
	for (int i = 0; i < this->ps_out.xmax.n_elem; i++)
	{
		if (i == (int)this->ps_out.xmax.n_elem - 1)
		{
			datasequence = datasequence + QString::number(ps_out.xmax(i)) + "\0";
		}
		else
		{
			datasequence = datasequence + QString::number(ps_out.xmax(i)) + " ";
		}
	}
	QDomText PSOutxmaxData = doc.createTextNode(datasequence);
	PSOutxmaxNode.appendChild(PSOutxmaxData);
	PSOutNode.appendChild(PSOutxmaxNode);


	QDomElement PSOutxminNode = doc.createElement("xmin");
	datasequence.clear();
	for (int i = 0; i < this->ps_out.xmin.n_elem; i++)
	{
		if (i == (int)this->ps_out.xmin.n_elem - 1)
		{
			datasequence = datasequence + QString::number(ps_out.xmin(i)) + "\0";
		}
		else
		{
			datasequence = datasequence + QString::number(ps_out.xmin(i)) + " ";
		}
	}
	QDomText PSOutxminData = doc.createTextNode(datasequence);
	PSOutxminNode.appendChild(PSOutxminData);
	PSOutNode.appendChild(PSOutxminNode);


	QDomElement PSOutxrangeNode = doc.createElement("xrange");
	datasequence.clear();
	for (int i = 0; i < this->ps_out.xrange.n_elem; i++)
	{
		if (i == (int)this->ps_out.xrange.n_elem - 1)
		{
			datasequence = datasequence + QString::number(ps_out.xrange(i)) + "\0";
		}
		else
		{
			datasequence = datasequence + QString::number(ps_out.xrange(i)) + " ";
		}
	}
	QDomText PSOutxrangeData = doc.createTextNode(datasequence);
	PSOutxrangeNode.appendChild(PSOutxrangeData);
	PSOutNode.appendChild(PSOutxrangeNode);


	QDomElement PSOutycolNode = doc.createElement("ycol");
	QDomText PSOutycolData = doc.createTextNode(QString::number(this->ps_out.ycol));
	PSOutycolNode.appendChild(PSOutycolData);
	PSOutNode.appendChild(PSOutycolNode);

	QDomElement PSOutymaxNode = doc.createElement("ymax");
	QDomText PSOutymaxData = doc.createTextNode(QString::number(this->ps_out.ymax));
	PSOutymaxNode.appendChild(PSOutymaxData);
	PSOutNode.appendChild(PSOutymaxNode);

	QDomElement PSOutyminNode = doc.createElement("ymin");
	QDomText PSOutyminData = doc.createTextNode(QString::number(this->ps_out.ymin));
	PSOutyminNode.appendChild(PSOutyminData);
	PSOutNode.appendChild(PSOutyminNode);

	QDomElement PSOutyrangeNode = doc.createElement("yrange");
	QDomText PSOutyrangeData = doc.createTextNode(QString::number(this->ps_out.yrange));
	PSOutyrangeNode.appendChild(PSOutyrangeData);
	PSOutNode.appendChild(PSOutyrangeNode);


	QDomElement PSOutgainNode = doc.createElement("gain");
	datasequence.clear();
	for (int i = 0; i < this->ps_out.gain.n_elem; i++)
	{
		if (i == (int)this->ps_out.gain.n_elem - 1)
		{
			datasequence = datasequence + QString::number(ps_out.gain(i)) + "\0";
		}
		else
		{
			datasequence = datasequence + QString::number(ps_out.gain(i)) + " ";
		}
	}
	QDomText PSOutgainData = doc.createTextNode(datasequence);
	PSOutgainNode.appendChild(PSOutgainData);
	PSOutNode.appendChild(PSOutgainNode);


	QDomElement PSOutxoffsetNode = doc.createElement("xoffset");
	datasequence.clear();
	for (int i = 0; i < this->ps_out.xoffset.n_elem; i++)
	{
		if (i == (int)this->ps_out.xoffset.n_elem - 1)
		{
			datasequence = datasequence + QString::number(ps_out.xoffset(i)) + "\0";
		}
		else
		{
			datasequence = datasequence + QString::number(ps_out.xoffset(i)) + " ";
		}
	} 
	QDomText PSOutxoffsetData = doc.createTextNode(datasequence);
	PSOutxoffsetNode.appendChild(PSOutxoffsetData);
	PSOutNode.appendChild(PSOutxoffsetNode);
	PSNode.appendChild(PSOutNode);
	root.appendChild(PSNode);

	//===========================================权重保存========================================================
	vec weight = this->output_weight();
	QDomElement weightNode = doc.createElement("weight");
	datasequence.clear();
	for (int i = 0; i < weight.n_elem; i++)
	{
		if (i == (int)weight.n_elem - 1)
		{
			datasequence = datasequence + QString::number(weight(i)) + "\0";
		}
		else
		{
			datasequence = datasequence + QString::number(weight(i)) + " ";
		}
	}

	QDomText weightData = doc.createTextNode(datasequence);
	weightNode.appendChild(weightData);
	root.appendChild(weightNode);
	
	extern QString BpModePath;
	QString Path = BpModePath + QString("BPNET_DATA_ITEM%1.xml").arg(currentItem);
	xmlfile->setFileName(Path);
	bool ret = xmlfile->open(QIODevice::WriteOnly | QIODevice::Text);
	if (!ret)
	{
		QMessageBox::warning(NULL, "warning", "保存模型文件失败");
		xmlfile->close();
		return false;
	}

	QTextStream stream(xmlfile);
	stream.setCodec("UTF-8");
	doc.save(stream, 4);  //文件每行缩进4个空格
	xmlfile->close();
	
	return true;

}
bool NeuralNetWork::load()
{
	if (xmlfile == NULL)
		return false;

	extern QString BpModePath;
	QString Path = BpModePath + QString("BPNET_DATA_ITEM%1.xml").arg(currentItem);
	xmlfile->setFileName(Path);
	bool ret = xmlfile->open(QIODevice::ReadOnly | QIODevice::Text);
	if (!ret)
	{
		xmlfile->close();
		QMessageBox::warning(NULL, "warning", "mode load failured!");
		return false;
	}

	QDomDocument doc;
	ret = doc.setContent(xmlfile);
	if (!ret)
	{
		xmlfile->close();
		QMessageBox::warning(NULL, "warning", "关联xml文件失败");
		return false;
	}

	xmlfile->close();
	QDomElement docElm = doc.documentElement();
	QString datasequence;
	QDomNode n = docElm.firstChild();
	if (n.nodeName() == "parameter")
	{
		QDomNodeList childList = n.childNodes();
		for (int i = 0; i < childList.size(); i++)
		{
			QDomNode childNode = childList.at(i);
			QDomElement chileElement = childNode.toElement();
			datasequence.clear();
			if (chileElement.nodeName() == "InputSize")
			{
				datasequence = chileElement.text();
				this->InputSize = datasequence.toInt();
			}
			else if (chileElement.nodeName() == "ActiveFun")
			{
				datasequence = chileElement.text();
				this->active_fun = datasequence;
			}
			else if (chileElement.nodeName() == "HideLayer")
			{
				datasequence = chileElement.text();
				this->hideLayerSize = datasequence.toInt();
			}
			else if (chileElement.nodeName() == "Epoch")
			{
				datasequence = chileElement.text();
				this->maxiters = datasequence.toInt();
			}
			else if (chileElement.nodeName() == "Lambda")
			{
				datasequence = chileElement.text();
				this->params.zeta = datasequence.toDouble();
			}
			else if (chileElement.nodeName() == "alpha")
			{
				datasequence = chileElement.text();
				this->params.alpha = datasequence.toDouble();
			}
			else if (chileElement.nodeName() == "Item")
			{
				datasequence = chileElement.text();
				if (datasequence.toInt() != this->currentItem)
					return false;
			}
			else{}
		}
	}
	else
	{
	   return false;
	}

	//===================================开始解析第PS节点==================================================
	n = n.nextSibling();
	if (n.nodeName() == "PS")
	{
		QDomNode PS = n.firstChild();
		if (PS.nodeName() != "PS_IN")
		{
			return false;
		}

		QDomNodeList childList = PS.childNodes();
		for (int i = 0; i < childList.size(); i++)
		{
			QDomNode childNode = childList.at(i);
			QDomElement chileElement = childNode.toElement();
			datasequence.clear();
			if (chileElement.nodeName() == "name")
			{
				datasequence = chileElement.text();
				this->ps_in.name = datasequence;
			}
			else if (chileElement.nodeName() == "xcol")
			{
				datasequence = chileElement.text();
				this->ps_in.xcol = datasequence.toInt();
			}
			else if (chileElement.nodeName() == "xmax")
			{
				datasequence = chileElement.text();
				QStringList datalist = datasequence.split(' ');
				this->ps_in.xmax = rowvec(datalist.size(), fill::zeros);
				for (int k = 0; k < datalist.size(); k++)
				{
					this->ps_in.xmax(k) = datalist.at(k).toDouble();
				}
			}
			else if (chileElement.nodeName() == "xmin")
			{
				datasequence = chileElement.text();
				QStringList datalist = datasequence.split(' ');
				this->ps_in.xmin = rowvec(datalist.size(), fill::zeros);
				for (int k = 0; k < datalist.size(); k++)
				{
					this->ps_in.xmin(k) = datalist.at(k).toDouble();
				}
			}
			else if (chileElement.nodeName() == "xrange")
			{
				datasequence = chileElement.text();
				QStringList datalist = datasequence.split(' ');
				this->ps_in.xrange = rowvec(datalist.size(), fill::zeros);
				for (int k = 0; k < datalist.size(); k++)
				{
					this->ps_in.xrange(k) = datalist.at(k).toDouble();
				}
			}
			else if (chileElement.nodeName() == "ycol")
			{
				datasequence = chileElement.text();
				this->ps_in.ycol = datasequence.toInt();
			}
			else if (chileElement.nodeName() == "ymax")
			{
				datasequence = chileElement.text();
				this->ps_in.ymax = datasequence.toDouble();
			}
			else if (chileElement.nodeName() == "ymin")
			{
				datasequence = chileElement.text();
				this->ps_in.ymin = datasequence.toDouble();
			}
			else if (chileElement.nodeName() == "yrange")
			{
				datasequence = chileElement.text();
				this->ps_in.yrange = datasequence.toDouble();
			}
			else if (chileElement.nodeName() == "gain")
			{
				datasequence = chileElement.text();
				QStringList datalist = datasequence.split(' ');
				this->ps_in.gain = rowvec(datalist.size(), fill::zeros);
				for (int k = 0; k < datalist.size(); k++)
				{
					this->ps_in.gain(k) = datalist.at(k).toDouble();
				}
			}
			else if (chileElement.nodeName() == "xoffset")
			{
				datasequence = chileElement.text();
				QStringList datalist = datasequence.split(' ');
				this->ps_in.xoffset = rowvec(datalist.size(), fill::zeros);
				for (int k = 0; k < datalist.size(); k++)
				{
					this->ps_in.xoffset(k) = datalist.at(k).toDouble();
				}
			}
			else {}

		}

		//======================================解析PS_OUT结点=====================================================================

		PS = PS.nextSibling();
		if (PS.nodeName() != "PS_out")
		{
			return false;
		}

		QDomNodeList PSoutchildList = PS.childNodes();
		for (int i = 0; i < PSoutchildList.size(); i++)
		{
			QDomNode childNode = PSoutchildList.at(i);
			QDomElement chileElement = childNode.toElement();
			datasequence.clear();
			if (chileElement.nodeName() == "name")
			{
				datasequence = chileElement.text();
				this->ps_out.name = datasequence;
			}
			else if (chileElement.nodeName() == "xcol")
			{
				datasequence = chileElement.text();
				this->ps_out.xcol = datasequence.toInt();
			}
			else if (chileElement.nodeName() == "xmax")
			{
				datasequence = chileElement.text();
				QStringList datalist = datasequence.split(' ');
				this->ps_out.xmax = rowvec(datalist.size(), fill::zeros);
				for (int k = 0; k < datalist.size(); k++)
				{
					this->ps_out.xmax(k) = datalist.at(k).toDouble();
				}
			}
			else if (chileElement.nodeName() == "xmin")
			{
				datasequence = chileElement.text();
				QStringList datalist = datasequence.split(' ');
				this->ps_out.xmin = rowvec(datalist.size(), fill::zeros);
				for (int k = 0; k < datalist.size(); k++)
				{
					this->ps_out.xmin(k) = datalist.at(k).toDouble();
				}
			}
			else if (chileElement.nodeName() == "xrange")
			{
				datasequence = chileElement.text();
				QStringList datalist = datasequence.split(' ');
				this->ps_out.xrange = rowvec(datalist.size(), fill::zeros);
				for (int k = 0; k < datalist.size(); k++)
				{
					this->ps_out.xrange(k) = datalist.at(k).toDouble();
				}
			}
			else if (chileElement.nodeName() == "ycol")
			{
				datasequence = chileElement.text();
				this->ps_out.ycol = datasequence.toInt();
			}
			else if (chileElement.nodeName() == "ymax")
			{
				datasequence = chileElement.text();
				this->ps_out.ymax = datasequence.toDouble();
			}
			else if (chileElement.nodeName() == "ymin")
			{
				datasequence = chileElement.text();
				this->ps_out.ymin = datasequence.toDouble();
			}
			else if (chileElement.nodeName() == "yrange")
			{
				datasequence = chileElement.text();
				this->ps_out.yrange = datasequence.toDouble();
			}
			else if (chileElement.nodeName() == "gain")
			{
				datasequence = chileElement.text();
				QStringList datalist = datasequence.split(' ');
				this->ps_out.gain = rowvec(datalist.size(), fill::zeros);
				for (int k = 0; k < datalist.size(); k++)
				{
					this->ps_out.gain(k) = datalist.at(k).toDouble();
				}
			}
			else if (chileElement.nodeName() == "xoffset")
			{
				datasequence = chileElement.text();
				QStringList datalist = datasequence.split(' ');
				this->ps_out.xoffset = rowvec(datalist.size(), fill::zeros);
				for (int k = 0; k < datalist.size(); k++)
				{
					this->ps_out.xoffset(k) = datalist.at(k).toDouble();
				}
			}
			else {}
		}	
	}
	else
	{
		return false;
	}

	n = n.nextSibling();
	if (n.nodeName() != "weight")
		return false;

	QDomElement chileElement = n.toElement();
	datasequence.clear();
	datasequence = chileElement.text();
	QStringList datalist = datasequence.split(' ');
	vec weight = vec(datalist.size(), fill::zeros);
	for (int k = 0; k < datalist.size(); k++)
	{
		weight(k) = datalist.at(k).toDouble();
	}
	if (!this->createMode(this->InputSize,
		this->active_fun,
		this->hideLayerSize,
		this->maxiters,
		this->params.zeta,
		this->params.alpha))
		return false;
	this->update_weight(weight);
	return true;
}



mat NeuralNetWork::mapminmax(PS &output, const mat X, double ymin , double ymax , QString name)
{
	output.name = name;
	mat Y;
	Y.clear();
	if (name == "mapminmax")
	{
		output.xmin = arma::min(X, 0);
		output.xmax = arma::max(X, 0);
		output.ymin = ymin;
		output.ymax = ymax;
		output.xcol = X.n_cols;
		output.ycol = X.n_cols;
		output.xrange = output.xmax - output.xmin;
		output.yrange = ymax - ymin;
		output.xoffset = output.xmin;
		output.gain = output.yrange / output.xrange;
		if(output.gain.has_inf())
		  output.gain.replace(datum::inf, 0.0);
		if (output.gain.has_nan())
			output.gain.replace(datum::nan, 0.0);

		Y = repmat(output.gain,X.n_rows,1) % (X - repmat(output.xmin, X.n_rows, 1)) + output.ymin;
	}
	else if(name == "apply")
	{
		Y = repmat(output.gain, X.n_rows, 1) % (X - repmat(output.xmin, X.n_rows, 1)) + output.ymin;
	}
	else
	{
		Y = (X - output.ymin) / (repmat(output.gain, X.n_rows, 1));
		if (Y.has_inf())
			Y.replace(datum::inf, 0.0);
		if (Y.has_nan())
			Y.replace(datum::nan, 0.0);
		Y = Y + repmat(output.xmin, X.n_rows, 1);
	}
	return Y;
}


void NeuralNetWork::shuffle_index(int len, int *a)
{
	srand((unsigned int)time(0));
	memset(a, 0, sizeof(int)*len);
	for (int i = 0; i < len; i++)
	{
		a[i] = i;
	}
	random_shuffle(a, a + len);
}

bool NeuralNetWork::convertTomat(const cube xdata, const vec T)
{
	if (xdata.is_empty())
		return false;
	
	mat originX;
	mat originY;
	originX.clear();
	originY.clear();

	traingX.clear();
	traingY.clear();

	originX = xdata.slice(0);
	int count_per = originX.n_rows;
	originY = T(0)*ones(count_per, 1);
	int num = xdata.n_slices;
	for (int i = 1; i < num; i++)
	{
		count_per = xdata.slice(i).n_rows;
		originX = join_cols(originX, xdata.slice(i));
		originY = join_cols(originY, T(i)*ones(count_per, 1));
	}

	traingX = mat(originX.n_rows, originX.n_cols,fill::zeros);
	traingY = mat(originY.n_rows, originY.n_cols, fill::zeros);
	int N = originY.n_rows;
	int *a = new int[N];

	shuffle_index(N, a);
	for (int i = 0; i < N; i++)
	{
		traingX.row(i) = originX.row(a[i]);
		traingY.row(i) = originY.row(a[i]);
	}


	mapminmax(ps_in, traingX, -1, 1);
	mapminmax(ps_out, traingY, -1, 1);

	params.batch_size = traingX.n_rows;
		
	delete[] a;
	return true;
}

void NeuralNetWork::train()
{
	if (traingX.is_empty() || traingY.is_empty())
		return;
	printf("Training on params:\nlearning rate= %.3f\nL2 regularization=%.3f\nmethod= %s\nepochs= %d\n",
		params.alpha, params.zeta, params.method.toStdString().c_str(), maxiters);
	timer.start();
	double weight_sum = 0;
	double delta_sum = 0;
	mat prediction;
	double loss = 0;
	vector<double>loss_history;
	loss_history.clear();
	emit SendCurrentProgress(0, maxiters);
	for (int i = 0; i < maxiters; i++)
	{
		prediction = this->FeedForward(traingX, weight_sum);
		loss = this->loss_function(prediction, traingY, weight_sum);
		loss_history.push_back(loss);
		this->backpropagation(traingY, delta_sum);
		this->update_parameters();
		printf("Step: [%d]   loss:%.4f   delta_sum: %.6f   weight_sum: %.4f\n", i, loss, delta_sum, weight_sum);
		emit SendCurrentProgress(i + 1, maxiters);
	}
	double toc = timer.elapsed() / 1000.0;
	emit SendTimeElapse(toc);
	emit SendTrainingFinished();

}


void NeuralNetWork::trainGlobal()
{
	vec w_int = this->output_weight();
	printf("start training:.....................................\n");
	w_int = fmincg(w_int);
	printf("training finished:.....................................\n");
	this->update_weight(w_int);
}


mat NeuralNetWork::FeedForward(const mat X, double &wsum)
{
	wsum = 0;
	double ws;
	mat inp = X;
	for (int i = 0; i < nodes.size(); i++)
	{
		if (type.at(i) == "Fullconnectlayer")
		{
			Fullconnectlayer *node = (Fullconnectlayer *)(nodes.at(i));
			inp = node->FeedForward(inp, ws);
			wsum += ws;
		}
		else if (type.at(i) == "Tanh")
		{
			Tanh *node = (Tanh *)(nodes.at(i));
			inp = node->FeedForward(inp, ws);
			//wsum += ws;
		}
		else if (type.at(i) == "Sigmoid")
		{
			Sigmoid *node = (Sigmoid *)(nodes.at(i));
			inp = node->FeedForward(inp, ws);
			//wsum += ws;
		}
		else if (type.at(i) == "Relu")
		{
			Relu *node = (Relu *)(nodes.at(i));
			inp = node->FeedForward(inp, ws);
			//wsum += ws;
		}
		else if (type.at(i) == "LinearLayer")
		{
			LinearLayer *node = (LinearLayer *)(nodes.at(i));
			inp = node->FeedForward(inp,ws);
			//wsum += ws;
		}
		else
		{

		}
	}

	return inp;
}
double NeuralNetWork::loss_function(mat pred, mat y, const double wsum)
{
	double loss = (sum(sum(square(pred - y))) + params.zeta*wsum) / (2.0*y.n_rows);
	return loss;
}
void NeuralNetWork::backpropagation(const mat Y, double &delta_sum)
{
	mat delta = Y;
	delta_sum = 0;
	double deltaw;
	for (int i = nodes.size()-1; i >=0; i--)
	{
		if (type.at(i) == "Fullconnectlayer")
		{
			Fullconnectlayer *node = (Fullconnectlayer *)(nodes.at(i));
			delta = node->backpropagation(delta, deltaw);
			delta_sum += deltaw;
		}
		else if (type.at(i) == "Tanh")
		{
			Tanh *node = (Tanh *)(nodes.at(i));
			delta = node->backpropagation(delta); 
		}
		else if (type.at(i) == "Sigmoid")
		{
			Sigmoid *node = (Sigmoid *)(nodes.at(i));
			delta = node->backpropagation(delta);
		}
		else if (type.at(i) == "Relu")
		{
			Relu *node = (Relu *)(nodes.at(i));
			delta = node->backpropagation(delta);
		}
		else if (type.at(i) == "LinearLayer")
		{
			LinearLayer *node = (LinearLayer *)(nodes.at(i));
			delta = node->backpropagation(delta);
		}
		else
		{

		}
	}
}

void NeuralNetWork::update_parameters()
{
	for (int i = 0; i < nodes.size(); i++)
	{
		if (type.at(i) == "Fullconnectlayer")
		{
			Fullconnectlayer *node = (Fullconnectlayer *)(nodes.at(i));
			node->update_kernel(params);
		}
	}
}

void NeuralNetWork::predict()
{
	double wsum;
	mat pred = FeedForward(traingX, wsum);
	pred = mapminmax(ps_out, pred, 0, 1, "reverse");
	mat y = mapminmax(ps_out, traingY, 0, 1, "reverse");
	mat result = join_rows(pred, y);
	mat ARE = mean(abs(pred - y) / y);

	result.print("compare:\n");
	printf("\n");
	ARE.print("ARE: ");
	printf("\n");
	mat res = y - pred;
	mat MSE = res.t()*res/ res.n_rows;
	MSE.print("MSE training Set:\n");
	printf("\n");
}

mat NeuralNetWork::predict(const mat Xtest)
{
	double wsum;
	mat p_test = mapminmax(ps_in, Xtest,-1,1,"apply");
	mat pred = FeedForward(p_test, wsum);
	pred = mapminmax(ps_out,pred, -1, 1, "reverse");
	return pred;
}































double NeuralNetWork::max(double a, double b)
{
	return a > b ? a : b;
}
double NeuralNetWork::min(double a, double b)
{
	return a < b ? a : b;
}

void NeuralNetWork::update_weight(vec w_init)
{
	int startPoint = 0;
	for (int i = 0; i < type.size(); i++)
	{
		if (type.at(i) == "Fullconnectlayer")
		{
			Fullconnectlayer *node = (Fullconnectlayer *)nodes.at(i);
			int length = node->layerSize.at(0)*node->layerSize.at(1) + node->layerSize.at(1);
			vec w = w_init.rows(startPoint, startPoint+length-1);
			node->update_weight(w);
			startPoint = startPoint + length;
		}
	}
}

vec NeuralNetWork::NeuralNetWork::backpropagation()
{
	vec out;
	out.clear();
	for (int i = 0; i < type.size(); i++)
	{
		if (type.at(i) == "Fullconnectlayer")
		{
			Fullconnectlayer *node = (Fullconnectlayer *)nodes.at(i);
			vec delta_w = node->backpropagation(params);
			if (out.is_empty())
			{
				out = delta_w;
			}
			else
			{
				out = join_cols(out, delta_w);
			}
		}
	}
	return out;
}

vec NeuralNetWork::output_weight()
{
	vec out;
	out.clear();
	for (int i = 0; i < type.size(); i++)
	{
		if (type.at(i) == "Fullconnectlayer")
		{
			Fullconnectlayer *node = (Fullconnectlayer *)nodes.at(i);
			vec w = node->output_weight();
			if (out.is_empty())
			{
				out = w;
			}
			else
			{
				out = join_cols(out, w);
			}
		}
	}
	return out;
}



double NeuralNetWork::costFun(vec weight, vec &grad_out)
{
	double delta_w;
	grad_out.clear();
	double weight_sum = 0.0;
	this->update_weight(weight);
	mat prediction = this->FeedForward(traingX, weight_sum);
	double loss = this->loss_function(prediction, traingY, weight_sum);
	this->backpropagation(traingY, delta_w);
	grad_out = backpropagation();
	return loss;
}



vec NeuralNetWork::fmincg(vec X)
{
	int length = maxiters;

	double RHO = 0.01;
	double SIG = 0.5;
	double INT = 0.1;
	double EXT = 3.0;
	double MAX = 20;
	double RATIO = 100;
	//=============================
	vector<double>fX;
	double red = 1;

	int i = 0;
	bool ls_failed = false;
	vec df1;
	//double f1 = f(X, df1, opt_data);
	double f1 = costFun(X, df1);

	i = i + (length < 0);
	vec s = -df1;
	mat d1 = -s.t()*s;
	double z1 = red / (1.0 - d1(0, 0));

	vec X0;
	double f0;
	vec df0;
	vec df2;
	double f2;
	double M;
	while (i < abs(length))
	{
		i = i + (length > 0);
		X0 = X;
		f0 = f1;
		df0 = df1;
		X = X + z1*s;
		//f2 = f(X, df2, opt_data);
		f2 = costFun(X, df2);

		i = i + (length < 0);
		mat d2 = df2.t()*s;
		double f3 = f1;
		mat d3 = d1;
		double z3 = -z1;

		if (length > 0)
			M = MAX;
		else
			M = min(MAX, -length - i);

		bool success = false;
		double limit = -1;
		double z2, A, B;
		vec tmp;
		while (true)
		{
			while ((f2 > (f1 + z1*RHO*d1(0, 0)) || (d2(0, 0) > -SIG*d1(0, 0))) && (M > 0))
			{
				limit = z1;

				if (f2 > f1)
					z2 = z3 - (0.5*d3(0, 0)*z3*z3) / (d3(0, 0)*z3 + f2 - f3);
				else
				{
					A = 6 * (f2 - f3) / z3 + 3 * (d2(0, 0) + d3(0, 0));
					B = 3 * (f3 - f2) - z3*(d3(0, 0) + 2.0*d2(0, 0));

					if ((B*B - A*d2(0, 0)*z3*z3) < 0)
					{
						goto label0;
					}
					z2 = (sqrt(B*B - A*d2(0, 0)*z3*z3) - B) / A;
				}
				/*if isnan(z2) | isinf(z2)
				z2 = z3 / 2;*/      //可能遇到问题
				if (arma::arma_isnan(z2) || arma::arma_isinf(z2))
				{
				label0:    z2 = z3 / 2;
				}

				z2 = max(min(z2, INT*z3), (1 - INT)*z3);

				z1 = z1 + z2;
				X = X + z2*s;
				//f2 = f(X, df2, opt_data);
				f2 = costFun(X, df2);
				M = M - 1;
				i = i + (length < 0);
				d2 = df2.t()*s;
				z3 = z3 - z2;
			}

			if ((f2 >(f1 + z1*RHO*d1(0, 0))) || (d2(0, 0) > -SIG*d1(0, 0)))
				break;
			else if (d2(0, 0) > SIG*d1(0, 0))
			{
				success = true; break;
			}
			else if (M == 0)
				break;
			A = 6 * (f2 - f3) / z3 + 3 * (d2(0, 0) + d3(0, 0));
			B = 3 * (f3 - f2) - z3*(d3(0, 0) + 2 * d2(0, 0));

			double  t1 = B*B - A*d2(0, 0)*z3*z3;
			if (t1 < 0)
			{
				goto label1;
			}
			z2 = -d2(0, 0)*z3*z3 / (B + sqrt(B*B - A*d2(0, 0)*z3*z3));
			//	if ~isreal(z2) | isnan(z2) | isinf(z2) | z2 < 0 % num prob or wrong sign ?   可能出现异常
			if (isnan(z2) || isinf(z2) || z2<0)
			{
			label1:   //goto语句
				if (limit < -0.5)
					z2 = z1*(EXT - 1);
				else
					z2 = (limit - z1) / 2;
			}

			else if ((limit > -0.5) && ((z2 + z1) > limit))
				z2 = (limit - z1) / 2;
			else if ((limit < -0.5) && (z2 + z1 > z1*EXT))
				z2 = z1*(EXT - 1.0);
			else if (z2 < -z3*INT)
				z2 = -z3*INT;
			else if ((limit > -0.5) && (z2 < (limit - z1)*(1.0 - INT)))
				z2 = (limit - z1)*(1.0 - INT);
			f3 = f2; d3 = d2; z3 = -z2;
			z1 = z1 + z2; X = X + z2*s;
			//f2 = f(X, df2, opt_data);
			f2 = costFun(X, df2);
			M = M - 1;
			i = i + (length < 0);
			d2 = df2.t()*s;
		}
		if (success)
		{
			f1 = f2;
			fX.push_back(f1);
			cout << "iteration " << i << " | Cost " << f1 << endl;
			mat tempM = (df2.t()*df2 - df1.t()*df2) / (df1.t()*df1);
			s = tempM(0, 0)*s - df2;
			tmp = df1; df1 = df2; df2 = tmp;
			d2 = df1.t()*s;
			if (d2(0, 0) > 0)
			{
				s = -df1;
				d2 = -s.t()*s;
			}

			z1 = z1* min(RATIO, d1(0, 0) / (d2(0, 0) - 12.2251e-308));
			d1 = d2;
			ls_failed = false;

		}
		else
		{
			X = X0; f1 = f0; df1 = df0;
			if (ls_failed || i > abs(length))
				break;
			tmp = df1;
			df1 = df2;
			df2 = tmp;
			s = -df1;
			d1 = -s.t()*s;
			z1 = 1.0 / (1 - d1(0, 0));
			ls_failed = true;
		}

	}

	return X;
}