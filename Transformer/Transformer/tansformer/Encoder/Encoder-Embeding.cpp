#include "Encoder-Embeding.h"



//----------------------
// EMBEDDING
//----------------------


Embedding::Embedding()
{
}


Embedding::Embedding(size_t embededWordSize ,size_t dictionarySize, double learningRate)
{
	_linearEmbedding  =  FeedForward();
	_linearEmbedding._mlp  =  MlpBuilder()
		.InputSize(dictionarySize)									// tamanho do dicionario;
		.Architecture({
			DenseLayer(embededWordSize, new Linear(), learningRate),//DenseLayer(embededWordSize, new ClipedLinear(-1,1), 0.001),        // world embedded vector
		})
		.Build();
}


Embedding::~Embedding()
{
}


Eigen::MatrixXd Embedding::Forward(Eigen::MatrixXd& inputWords)
{
	Eigen::MatrixXd embeddedWord  =  _linearEmbedding.Forward( inputWords );
	return embeddedWord;
}


Eigen::MatrixXd Embedding::Backward(Eigen::MatrixXd& dL_dEmbeddedWord)
{
	Eigen::MatrixXd dL_dInputWord  =  _linearEmbedding.Backward( dL_dEmbeddedWord );
	return dL_dInputWord;
}



Eigen::MatrixXd Embedding::Positioning(Eigen::MatrixXd& embededInput)
{
	size_t seqLen  =  embededInput.rows();
	size_t dModel  =  embededInput.cols();

	for (size_t pos = 0; pos < seqLen; pos++) {
		for (size_t i = 0; i < dModel; i++) {

			if (i % 2 == 0) {
				double positionEncode = std::sin(pos / std::pow(10'000.0, static_cast<double>(2.0*i) / dModel));
				embededInput(pos,i)  =  embededInput(pos, i) + positionEncode;
			} else {
				double positionEncode = std::cos(pos / std::pow(10'000.0, static_cast<double>(2.0*i) / dModel));
				embededInput(pos,i)  =  embededInput(pos, i) + positionEncode;
			}

		}
	}

	return embededInput;   // positioned Embedding
}
