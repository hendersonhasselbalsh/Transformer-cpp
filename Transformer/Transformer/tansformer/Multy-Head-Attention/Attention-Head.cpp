#include "Attention-Head.h"

AttentionHead::AttentionHead(size_t linearSize, SDPAttention::Attrib attrib, double learningRate )
{
	_linearQ  =  FeedForward();
	_linearQ._mlp  =  MlpBuilder()
		.InputSize(linearSize)
		.Architecture({
			DenseLayer(linearSize, new Linear(), learningRate),//DenseLayer(linearSize, new ClipedLinear(-1,1), 0.001),
		})
		.Build();


	_linearK  =  FeedForward();
	_linearK._mlp  =  MlpBuilder()
		.InputSize(linearSize)
		.Architecture({
			DenseLayer(linearSize, new Linear(), learningRate),//DenseLayer(linearSize, new ClipedLinear(-1,1), 0.001),
		})
		.Build();


	_linearV  =  FeedForward();
	_linearV._mlp  =  MlpBuilder()
		.InputSize(linearSize)
		.Architecture({
			DenseLayer(linearSize, new Linear(), learningRate),//DenseLayer(linearSize, new ClipedLinear(-1,1), 0.001),
		})
		.Build();


	_scaledDotProductAttention  =  ScaledDotProductAttention(attrib);
}



AttentionHead::~AttentionHead()
{
}



Eigen::MatrixXd AttentionHead::Forward(Eigen::MatrixXd& Q, Eigen::MatrixXd& K, Eigen::MatrixXd& V)
{
	Eigen::MatrixXd QW_q  =  _linearQ.Forward( Q );
	Eigen::MatrixXd KW_k  =  _linearK.Forward( K );
	Eigen::MatrixXd VW_v  =  _linearV.Forward( V );

	Eigen::MatrixXd attentionHead  =  _scaledDotProductAttention.Forward(QW_q, KW_k, VW_v);

	return attentionHead;
}

DL_DAttention AttentionHead::Backward(Eigen::MatrixXd& dL_dAttentionHead)
{
	DL_DAttention dL_dHeadInput = DL_DAttention();


	SDPAttentionBackwardResult SDPABackwardResult =  _scaledDotProductAttention.Backward( dL_dAttentionHead );

	dL_dHeadInput.Q  =  _linearQ.Backward( SDPABackwardResult.dL_dQuery );
	dL_dHeadInput.K  =  _linearK.Backward( SDPABackwardResult.dL_dKey );
	dL_dHeadInput.V  =  _linearV.Backward( SDPABackwardResult.dL_dValue );

	return dL_dHeadInput;
}



//AttentionHeadBackwardResult AttentionHead::Backward(Eigen::MatrixXd& dL_dAttentionHead)
//{
//	AttentionHeadBackwardResult attentionHeadBackwardResult = AttentionHeadBackwardResult();
//
//
//	SDPAttentionBackwardResult SDPABackwardResult =  _scaledDotProductAttention.Backward( dL_dAttentionHead );
//
//	attentionHeadBackwardResult.dL_dQ  =  _linearQ.Forward( SDPABackwardResult.dL_dQuery );
//	attentionHeadBackwardResult.dL_dK  =  _linearK.Forward( SDPABackwardResult.dL_dKey );
//	attentionHeadBackwardResult.dL_dV  =  _linearV.Forward( SDPABackwardResult.dL_dValue );
//
//	return attentionHeadBackwardResult;
//}
