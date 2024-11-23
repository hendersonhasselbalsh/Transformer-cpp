#include "Encoder-Decoder-Transformer.h"



EncodeDecodeTransformer::EncodeDecodeTransformer(size_t embededWordSize, size_t dictionaryInput, size_t dictionaryOutput, size_t h, ILostFunction* lossFunc, double learningRate)
	: _encoder(embededWordSize, h), _decoder(embededWordSize, h)
{
	_encodeEmbedding  =  Embedding(embededWordSize, dictionaryInput, learningRate);
	_decodeEmbedding  =  Embedding(embededWordSize, dictionaryOutput, learningRate);


	_lossFunc  =  lossFunc;
	_softmax  =  Softmax();


	_linear  =  FeedForward();
	_linear._mlp  =  MlpBuilder()
		.InputSize(embededWordSize)
		.Architecture({
			DenseLayer(dictionaryOutput, new Linear(), learningRate),//DenseLayer(dictionaryOutput, new ClipedLinear(-1,1), 0.001),
		})
		.Build();
}


EncodeDecodeTransformer::~EncodeDecodeTransformer()
{
}





Eigen::MatrixXd EncodeDecodeTransformer::Forward(Eigen::MatrixXd& encoderInput, Eigen::MatrixXd& decoderInput)
{
	Eigen::MatrixXd embeddedEncoderInput  =  _encodeEmbedding.Forward( encoderInput );
	Eigen::MatrixXd positionedEncoderInput  =  _encodeEmbedding.Positioning( embeddedEncoderInput );
	Eigen::MatrixXd encodedAttentionState  =  _encoder.Forward( positionedEncoderInput );


	Eigen::MatrixXd embeddedDecodeInput  =  _decodeEmbedding.Forward( decoderInput );
	Eigen::MatrixXd positionedDecodeInput = _decodeEmbedding.Positioning( embeddedDecodeInput );
	Eigen::MatrixXd decoderOutputMatrix  =  _decoder.Forward( positionedDecodeInput, encodedAttentionState );


	Eigen::MatrixXd wordPredictionMatrix  =  _linear.Forward( decoderOutputMatrix );


	Eigen::MatrixXd outputProbabilityMatriXd  =  _softmax.Forward( wordPredictionMatrix );


	// return outputProbabilityMatriXd;    // the predicted word is only the last line
	size_t lastRow  =  outputProbabilityMatriXd.rows() - 1;
	Eigen::MatrixXd predictedToken  =   outputProbabilityMatriXd.row( lastRow );
	return predictedToken;
}



DL_DTransformerInputs EncodeDecodeTransformer::Backward(Eigen::MatrixXd& predictedMatrix, Eigen::MatrixXd& correctMatrixXd)
{
	//--- DEBUG
	/*std::cout << "\n\n\n\n\n\n-----------------------------------[EncodeDecodeTransformer]--------------------------------------------------\n\n\n";
	std::cout << "[EncodeDecodeTransformer] predictedMatrix:\n" << predictedMatrix << "\n\n";
	std::cout << "[EncodeDecodeTransformer] correctMatrixXd:\n" << correctMatrixXd << "\n\n\n";*/
	//--- END DEBUG

	Eigen::MatrixXd dL_dPredictedMatrix  =  DL_DPredictedMatrix( predictedMatrix, correctMatrixXd );

	//--- DEBUG
	/*if (dL_dPredictedMatrix.array().isNaN().any()) {
		std::cout << "[EncodeDecodeTransformer] dL_dPredictedMatrix:\n" << dL_dPredictedMatrix << "\n\n";
		bool DEBUG_STOP = true;
	}*/
	//--- END DEBUG





	Eigen::MatrixXd dL_dWordPredictionMatrix  =  _softmax.Backward( dL_dPredictedMatrix );

	//--- DEBUG
	/*if (dL_dWordPredictionMatrix.array().isNaN().any()) {
		std::cout << "[EncodeDecodeTransformer] dL_dWordPredictionMatrix:\n" << dL_dWordPredictionMatrix << "\n\n";
		bool DEBUG_STOP = true;
	}*/
	//--- END DEBUG





	Eigen::MatrixXd dL_dDecoderOutputMatrix  =  _linear.Backward( dL_dWordPredictionMatrix );

	//--- DEBUG
	/*if (dL_dDecoderOutputMatrix.array().isNaN().any()) {
		std::cout << "[EncodeDecodeTransformer] dL_dDecoderOutputMatrix:\n" << dL_dDecoderOutputMatrix << "\n\n";
	}*/
	//--- END DEBUG






	DL_DTransformerInputs dL_dTransformerInputs  =  DL_DTransformerInputs();






	DL_DDecoder dL_dDencoderInputs  =  _decoder.Backward( dL_dDecoderOutputMatrix );
	// _decoder.Backward   esta causando erro (gradiente explodente)
	//--- DEBUG
	//std::cout << "[EncodeDecodeTransformer] dL_dDencoderInputs.shiftedOutput:\n" << dL_dDencoderInputs.shiftedOutput << "\n\n"; 
	//std::cout << "[EncodeDecodeTransformer] dL_dDencoderInputs.encoderStates:\n" << dL_dDencoderInputs.encoderStates << "\n\n";
	//--- END DEBUG





	dL_dTransformerInputs.encoderInput  =  _encoder.Backward( dL_dDencoderInputs.encoderStates );
	dL_dTransformerInputs.decoderInput  =  dL_dDencoderInputs.shiftedOutput;

	//--- DEBUG
	//std::cout << "[EncodeDecodeTransformer] dL_dTransformerInputs.encoderInput:\n" << dL_dTransformerInputs.encoderInput << "\n\n";
	//std::cout << "[EncodeDecodeTransformer] dL_dTransformerInputs.decoderInput:\n" << dL_dTransformerInputs.decoderInput << "\n\n";
	//--- END DEBUG



	dL_dTransformerInputs.decoderInput  =  _decodeEmbedding.Backward( dL_dTransformerInputs.decoderInput );
	dL_dTransformerInputs.encoderInput  =  _encodeEmbedding.Backward( dL_dTransformerInputs.encoderInput );

	//--- DEBUG
	//std::cout << "[EncodeDecodeTransformer] dL_dTransformerInputs._decodeEmbedding:\n" << dL_dTransformerInputs.encoderInput << "\n\n";
	//std::cout << "[EncodeDecodeTransformer] dL_dTransformerInputs._encodeEmbedding:\n" << dL_dTransformerInputs.decoderInput << "\n\n";
	//--- END DEBUG



	return dL_dTransformerInputs;
}





Eigen::MatrixXd EncodeDecodeTransformer::DL_DPredictedMatrix(Eigen::MatrixXd& predictedMatrix, Eigen::MatrixXd& correctMatrixXd)
{
	assert(predictedMatrix.rows()==correctMatrixXd.rows() && predictedMatrix.cols()==correctMatrixXd.cols() && "[ERROR]: both matrix must have same dimentions");

	Eigen::MatrixXd dL_dPredictedMatrix  =  Eigen::MatrixXd(predictedMatrix.rows(), predictedMatrix.cols());

	for (size_t row = 0; row < predictedMatrix.rows(); row++) {
		for (size_t col = 0; col < predictedMatrix.cols(); col++) {
			dL_dPredictedMatrix(row,col)  =  _lossFunc->df( predictedMatrix(row,col), correctMatrixXd(row, col) );
		}
	}

	return dL_dPredictedMatrix;
}
