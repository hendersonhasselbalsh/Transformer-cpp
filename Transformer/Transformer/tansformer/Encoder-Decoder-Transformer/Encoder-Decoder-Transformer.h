#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../../utils/basic-includes.h"

#include "../add-norm/Add-Norm.h"
#include "../Multy-Head-Attention/Attention-Head.h"
#include "../Multy-Head-Attention/Multy-Head-Attention.h"

#include "../Encoder/Encoder-Embeding.h"
#include "../Encoder/Encoder.h"
#include "../Decoder/Decoder.h"
#include "../Scalled-Dot-Product-Attention/Softmax.h"
#include "../feed-forward/FeedForward.h"



struct DL_DTransformerInputs {
	Eigen::MatrixXd encoderInput;
	Eigen::MatrixXd decoderInput;

	DL_DTransformerInputs(){  }
};



class EncodeDecodeTransformer {

	private:
		Embedding _encodeEmbedding;
		Embedding _decodeEmbedding;

		Encoder _encoder;
		Decoder _decoder;

		Softmax _softmax;
		FeedForward _linear;

		ILostFunction* _lossFunc;


	public:
		EncodeDecodeTransformer(size_t embededWordSize, size_t dictionaryInput, size_t dictionaryOutput, size_t h = 4, ILostFunction* lossFunc = new MSE(), double learningRate = 0.001);
		~EncodeDecodeTransformer();

		Eigen::MatrixXd Forward(Eigen::MatrixXd& encoderInput, Eigen::MatrixXd& decoderInput);
		DL_DTransformerInputs Backward(Eigen::MatrixXd& predictedMatrix, Eigen::MatrixXd& correctMatrixXd);



		Eigen::MatrixXd DL_DPredictedMatrix(Eigen::MatrixXd& predictedMatrix, Eigen::MatrixXd& correctMatrixXd);

};





