#include "Encoder.h"



Encoder::Encoder(size_t inputMatrixCols, size_t h, SDPAttention::Attrib attrib, double learningRate)
{
	//--- MULTY HEAD ATTENTION
	_multyHeadSelfAttention = MultyHeadAttention(inputMatrixCols, h, attrib, learningRate);


	//--- ADD NORMs
	_feedForward_AddNorm = AddNorm( inputMatrixCols, learningRate);
	_attention_AddNorm = AddNorm( inputMatrixCols, learningRate);


	//--- FEED FORWARD
	_feedForward  =  FeedForward();
	_feedForward._mlp  =  MlpBuilder()
		.InputSize(inputMatrixCols)
		.Architecture({
			DenseLayer(inputMatrixCols, new ReLU(), learningRate),
			DenseLayer(inputMatrixCols, new Linear(), learningRate),//DenseLayer(inputMatrixCols, new ClipedLinear(-1,1), 0.001),
		})
		.Build();


}



Encoder::~Encoder()
{
}



Eigen::MatrixXd Encoder::Forward(Eigen::MatrixXd& positionedInput)
{
	Eigen::MatrixXd multyHeadAttention  =  _multyHeadSelfAttention.Forward(positionedInput, positionedInput, positionedInput);
	Eigen::MatrixXd normalizedAttention =  _attention_AddNorm.Forward(multyHeadAttention, positionedInput);


	Eigen::MatrixXd attentionStates  =  _feedForward.Forward( normalizedAttention );
	Eigen::MatrixXd normalizedAttentionStates  =  _feedForward_AddNorm.Forward(attentionStates, normalizedAttention);


	return normalizedAttentionStates;
}



Eigen::MatrixXd Encoder::Backward(Eigen::MatrixXd& dL_dAttentionStates)
{
	Eigen::MatrixXd dL_dFeedForwardAddNorm = _feedForward_AddNorm.Backward( dL_dAttentionStates );


	Eigen::MatrixXd dL_dNormalizedAttention = _feedForward.Backward( dL_dFeedForwardAddNorm );
	dL_dNormalizedAttention  =  dL_dNormalizedAttention  +  dL_dFeedForwardAddNorm;


	Eigen::MatrixXd dL_dAttention  =  _attention_AddNorm.Backward( dL_dNormalizedAttention );


	DL_DAttention dL_dHead = _multyHeadSelfAttention.Backward( dL_dAttention );


	Eigen::MatrixXd dL_dPositionedInput = dL_dHead.Q + dL_dHead.K + dL_dHead.V;
	dL_dPositionedInput  =  dL_dPositionedInput  +  dL_dNormalizedAttention;


	return dL_dPositionedInput;
}
