#include "Multy-Head-Attention.h"



MultyHeadAttention::MultyHeadAttention()
{
}

MultyHeadAttention::MultyHeadAttention(size_t inputMatrixCols, size_t h, SDPAttention::Attrib attrib, double learningRate)
{
	assert(inputMatrixCols % h == 0);

    _h = h;

	size_t headsQnt  =  inputMatrixCols / h;
	_attentionHeads  =  std::vector<AttentionHead>(headsQnt, AttentionHead(h, attrib, learningRate) );

    _linear  =  FeedForward();
    _linear._mlp  =  MlpBuilder()
        .InputSize( inputMatrixCols )
        .Architecture({
            DenseLayer(inputMatrixCols, new Linear(), learningRate),//DenseLayer(inputMatrixCols, new ClipedLinear(-1,1), 0.001),
        })
        .Build();

}

MultyHeadAttention::~MultyHeadAttention()
{
}



//Eigen::MatrixXd MultyHeadAttention::Forward(Eigen::MatrixXd& input)
//{
//    std::vector<Head> headsInputs  =  SplitMatrixIntoHeads(input, _h);
//
//
//    std::vector<Eigen::MatrixXd> splitedAttentionHeads  =  std::vector<Eigen::MatrixXd>();
//
//    for (size_t i = 0; i < _attentionHeads.size(); i++) {
//
//        Head head  =  headsInputs[i];
//
//        Eigen::MatrixXd attentionHead = _attentionHeads[i].Forward( head.Q, head.K, head.V );
//        splitedAttentionHeads.push_back( attentionHead );
//    }
//
//    Eigen::MatrixXd concatedAttentionHeads  =  ConcatMatrix( splitedAttentionHeads );
//    
//    Eigen::MatrixXd multyHeadAttention  =  _linear.Forward( concatedAttentionHeads );
//
//
//    return multyHeadAttention;
//}



Eigen::MatrixXd MultyHeadAttention::Forward(Eigen::MatrixXd& Q, Eigen::MatrixXd& K, Eigen::MatrixXd& V)
{
   std::vector<Eigen::MatrixXd> splitedQ  =  SplitMatrix(Q, _h);
   std::vector<Eigen::MatrixXd> splitedK  =  SplitMatrix(K, _h);
   std::vector<Eigen::MatrixXd> splitedV  =  SplitMatrix(V, _h);


   std::vector<Eigen::MatrixXd> splittedAttentionHeads  =  std::vector<Eigen::MatrixXd>();

   for (size_t i = 0; i < _attentionHeads.size(); i++) {
       Eigen::MatrixXd partialAttentionHead = _attentionHeads[i].Forward(splitedQ[i], splitedK[i], splitedV[i]);
       splittedAttentionHeads.push_back( partialAttentionHead );
   }


   Eigen::MatrixXd concatedAttentionHeads  =  ConcatMatrix( splittedAttentionHeads );
   Eigen::MatrixXd multyHeadAttention  =  _linear.Forward( concatedAttentionHeads );

   return multyHeadAttention;
}



DL_DAttention MultyHeadAttention::Backward(Eigen::MatrixXd& dL_dMultyHeadAttention)
{
    Eigen::MatrixXd dL_dConcatedAttentionHead  =  _linear.Backward( dL_dMultyHeadAttention );
    std::vector<Eigen::MatrixXd> dL_dSplittedAttentionHead  =  SplitMatrix( dL_dConcatedAttentionHead, _h );


    std::vector<DL_DAttention> dL_dAttentionHeads  =  std::vector<DL_DAttention>();

    for (size_t i = 0; i < _attentionHeads.size(); i++) {
        DL_DAttention dL_dPartialAttentionHead  = _attentionHeads[i].Backward( dL_dSplittedAttentionHead[i] );
        dL_dAttentionHeads.push_back( dL_dPartialAttentionHead );
    }


    DL_DAttention dL_dInput  =  ConcatMatrixIntoHeads( dL_dAttentionHeads );
    return dL_dInput;
}


//Eigen::MatrixXd MultyHeadAttention::Backward(Eigen::MatrixXd& dL_dMultyHeadAttention)
//{
//    
//    Eigen::MatrixXd dL_dConcatedAttentionHeads  =  _linear.Backward( dL_dMultyHeadAttention );
//
//    std::vector<Eigen::MatrixXd> dL_dSplitedAttentionHeads  =  SplitMatrix( dL_dConcatedAttentionHeads, _h );
//
//
//    std::vector<Head> dL_dSplittedHead  =  std::vector<Head>();
//
//    for (size_t i = 0; i < _attentionHeads.size(); i++) {
//
//        Eigen::MatrixXd dL_dAttentionHead  =  dL_dSplitedAttentionHeads[i];
//
//        auto backwardResult = _attentionHeads[i].Backward( dL_dAttentionHead );
//
//        Head dL_dHead  =  Head();
//        dL_dHead.Q  =  backwardResult.dL_dQ;
//        dL_dHead.K  =  backwardResult.dL_dK;
//        dL_dHead.V  =  backwardResult.dL_dV;
//
//        dL_dSplittedHead.push_back(dL_dHead);
//    }
//
//    Head dL_dheadsInputs  =  ConcatMatrixIntoHeads( dL_dSplittedHead );
//
//    Eigen::MatrixXd dL_dInput  =  dL_dheadsInputs.Q  +  dL_dheadsInputs.K  +  dL_dheadsInputs.V;
//    return dL_dInput;
//}










std::vector<DL_DAttention> MultyHeadAttention::SplitMatrixIntoHeads(Eigen::MatrixXd& input, size_t h)
{
    std::vector<DL_DAttention> heads = std::vector<DL_DAttention>();

    size_t headsQnt = input.cols() / h;

    for (size_t i = 0; i < headsQnt; i++) {
        Eigen::MatrixXd subMatrix = input.block(0, i * h, input.rows(), h);

        DL_DAttention head = DL_DAttention();
        head.Q  = subMatrix;
        head.K  = subMatrix;
        head.V  = subMatrix;


        heads.push_back( head );
    }


    return heads;
}



DL_DAttention MultyHeadAttention::ConcatMatrixIntoHeads(std::vector<DL_DAttention> heads)
{
    DL_DAttention result  =  DL_DAttention();


    //--- Q
    std::vector<Eigen::MatrixXd> individualHead  =  std::vector<Eigen::MatrixXd>();

    for (auto& head : heads) {
        individualHead.push_back( head.Q );
    }

    result.Q  =  ConcatMatrix( individualHead );


    //--- K
    individualHead  =  std::vector<Eigen::MatrixXd>();

    for (auto& head : heads) {
        individualHead.push_back(head.K);
    }

    result.K  =  ConcatMatrix(individualHead);


    //--- V
    individualHead  =  std::vector<Eigen::MatrixXd>();

    for (auto& head : heads) {
        individualHead.push_back(head.V);
    }

    result.V  =  ConcatMatrix(individualHead);


    return result;
}

/*

DL_DAttention MultyHeadAttention::ConcatMatrixIntoHeads(std::vector<DL_DAttention> heads)
{
    size_t numRows = heads[0].Q.rows();
    size_t totalCols = 0;

    for (const auto& mat : heads) {
        totalCols += mat.Q.cols();
    }


    DL_DAttention result  =  DL_DAttention();
    result.Q  =  Eigen::MatrixXd(numRows, totalCols);
    result.K  =  Eigen::MatrixXd(numRows, totalCols);
    result.V  =  Eigen::MatrixXd(numRows, totalCols);


    size_t currentCol = 0;
    for (const auto& head : heads) {
        result.Q.block(0, currentCol, numRows, head.Q.cols())  =  head.Q;
        result.K.block(0, currentCol, numRows, head.K.cols())  =  head.K;
        result.V.block(0, currentCol, numRows, head.V.cols())  =  head.V;

        currentCol += head.Q.cols();
    }

    return result;
}

*/






std::vector<Eigen::MatrixXd> MultyHeadAttention::SplitMatrix(Eigen::MatrixXd& input, size_t h)
{
    std::vector<Eigen::MatrixXd> matrixies = std::vector<Eigen::MatrixXd>();

    size_t matrixiesQnt = input.cols() / h;

    for (size_t i = 0; i < matrixiesQnt; i++) {

        Eigen::MatrixXd subMatrix = input.block(0, i * h, input.rows(), h);
        matrixies.push_back( subMatrix );

    }

    return matrixies;
}



Eigen::MatrixXd MultyHeadAttention::ConcatMatrix(std::vector<Eigen::MatrixXd> matrixies)
{
    size_t numRows = matrixies[0].rows();
    size_t totalCols = 0;

    for (const auto& mat : matrixies) {
        totalCols += mat.cols();
    }

    Eigen::MatrixXd result(numRows, totalCols);

    size_t currentCol = 0;
    for (const auto& mat : matrixies) {
        result.block(0, currentCol, numRows, mat.cols()) = mat;
        currentCol += mat.cols();
    }

    return result;
}


