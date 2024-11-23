#include "Decoder.h"



Decoder::Decoder(size_t inputMatrixCols, size_t h, double learningRate)
{
	//--- MULTI HEAD ATTENTIONs
	_maskMultiHeadAttention = MultyHeadAttention(inputMatrixCols, h, SDPAttention::Attrib::USE_MASK, learningRate);
	_multiHeadCrossAttention = MultyHeadAttention(inputMatrixCols, h, SDPAttention::Attrib::DONT_USE_MASK, learningRate);


	//--- ADD & NORMs
	_addNorm_1 = AddNorm( inputMatrixCols, learningRate );
	_addNorm_2 = AddNorm( inputMatrixCols, learningRate );
	_addNorm_3 = AddNorm( inputMatrixCols, learningRate );


	//--- FEED FORWARD
	_feedForward = FeedForward();
	_feedForward._mlp = MlpBuilder()
		.InputSize(inputMatrixCols)
		.Architecture({
			DenseLayer(inputMatrixCols, new ReLU(), learningRate),
			DenseLayer(inputMatrixCols, new Linear(), learningRate),//DenseLayer(inputMatrixCols, new ClipedLinear(-1,1), 0.001),
		})
		.Build();

}



Decoder::~Decoder()
{
}




Eigen::MatrixXd Decoder::Forward(Eigen::MatrixXd& shiftedOutput, Eigen::MatrixXd& encoderStates)
{
	Eigen::MatrixXd maskedAttention = _maskMultiHeadAttention.Forward(shiftedOutput, shiftedOutput, shiftedOutput);
	Eigen::MatrixXd NormalizedMaskAttention  =  _addNorm_1.Forward(maskedAttention, shiftedOutput);


	Eigen::MatrixXd crossAttention  =  _multiHeadCrossAttention.Forward(NormalizedMaskAttention, encoderStates, encoderStates);
	Eigen::MatrixXd normalizedCrossAttention = _addNorm_2.Forward(crossAttention, NormalizedMaskAttention);


	Eigen::MatrixXd decodedOutput  =  _feedForward.Forward( normalizedCrossAttention );
	Eigen::MatrixXd normalizedOutput  =  _addNorm_3.Forward(decodedOutput, normalizedCrossAttention);


	return normalizedOutput;
}




DL_DDecoder Decoder::Backward(Eigen::MatrixXd& dL_dNormalizedOutput)
{
	//--- DEBUG
	//std::cout << "\n\n\n\n--------------------------Decoder::Backward---------------------------------\n\n\n\n";
	//--- END DEBUG



	DL_DDecoder dL_dDencoder = DL_DDecoder();



	Eigen::MatrixXd dL_dDecodedOutput = _addNorm_3.Backward( dL_dNormalizedOutput );
	Eigen::MatrixXd dL_dNormalizedCrossAttention = _feedForward.Backward( dL_dDecodedOutput );
	dL_dNormalizedCrossAttention  =  dL_dNormalizedCrossAttention  +  dL_dDecodedOutput;

	//--- DEBUG
	/*std::cout << "[Decoder] dL_dDecodedOutput:\n" << dL_dDecodedOutput << "\n\n";
	std::cout << "[Decoder] dL_dNormalizedCrossAttention:\n" << dL_dNormalizedCrossAttention << "\n\n";*/
	//--- END DEBUG







	Eigen::MatrixXd dL_dCrossAttention  =  _addNorm_2.Backward( dL_dNormalizedCrossAttention );
	DL_DAttention dL_dNormalizedMaskAttention  =  _multiHeadCrossAttention.Backward( dL_dCrossAttention );
	dL_dDencoder.encoderStates  =  dL_dNormalizedMaskAttention.K + dL_dNormalizedMaskAttention.V;                  // fist backward output
	dL_dNormalizedMaskAttention.Q  =  dL_dNormalizedMaskAttention.Q  +  dL_dCrossAttention;

	//--- DEBUG
	/*std::cout << "[Decoder] dL_dCrossAttention:\n" << dL_dCrossAttention << "\n\n";
	std::cout << "[Decoder] dL_dNormalizedMaskAttention.q:\n" << dL_dNormalizedMaskAttention.Q << "\n\n";
	std::cout << "[Decoder] dL_dNormalizedMaskAttention.k:\n" << dL_dNormalizedMaskAttention.K << "\n\n";
	std::cout << "[Decoder] dL_dNormalizedMaskAttention.v:\n" << dL_dNormalizedMaskAttention.V << "\n\n";
	std::cout << "[Decoder] dL_dDencoder.encoderStates  =  dL_dNormalizedMaskAttention.K + dL_dNormalizedMaskAttention.V;:\n" << dL_dDencoder.encoderStates << "\n\n";
	std::cout << "[Decoder] dL_dNormalizedMaskAttention.Q  =  dL_dNormalizedMaskAttention.Q  +  dL_dCrossAttention:\n" << dL_dNormalizedMaskAttention.Q << "\n\n";*/
	//--- END DEBUG








	Eigen::MatrixXd dL_dMaskedAttention  =  _addNorm_1.Backward( dL_dNormalizedMaskAttention.Q );
	DL_DAttention dL_dAttentionHeads  = _maskMultiHeadAttention.Backward( dL_dMaskedAttention );
	

	dL_dDencoder.shiftedOutput  =  dL_dAttentionHeads.Q + dL_dAttentionHeads.K + dL_dAttentionHeads.V;
	dL_dDencoder.shiftedOutput  =  dL_dDencoder.shiftedOutput  +  dL_dMaskedAttention;                             // fist backward output


	//--- DEBUG
	/*std::cout << "[Decoder] dL_dMaskedAttention:\n" << dL_dMaskedAttention << "\n\n";
	std::cout << "[Decoder] dL_dAttentionHeads.q:\n" << dL_dAttentionHeads.Q << "\n\n";
	std::cout << "[Decoder] dL_dAttentionHeads.k:\n" << dL_dAttentionHeads.K << "\n\n";
	std::cout << "[Decoder] dL_dAttentionHeads.v:\n" << dL_dAttentionHeads.V << "\n\n";
	std::cout << "[Decoder] dL_dDencoder.shiftedOutput  =  dL_dAttentionHeads.Q + dL_dAttentionHeads.K + dL_dAttentionHeads.V + dL_dMaskedAttention:\n" << dL_dDencoder.shiftedOutput << "\n\n";*/
	//--- END DEBUG




	//--- DEBUG
	//std::cout << "\n\n\n\n--------------------------[END] Decoder::Backward---------------------------------\n\n\n\n";
	//--- END DEBUG



	return dL_dDencoder;
}





