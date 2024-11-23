#include "Add-Norm.h"
#include "../../utils/utils.h"

AddNorm::AddNorm(size_t embeddingSize, double learningRate)
{

    _learningRate  =  learningRate;


    _gammas  =  std::vector<double>(embeddingSize, 1.0);
    _betas  =  std::vector<double>(embeddingSize, 0.0);
}

AddNorm::AddNorm()
{

}

AddNorm::~AddNorm()
{
}



Eigen::MatrixXd AddNorm::Forward(Eigen::MatrixXd& inputMatrix, Eigen::MatrixXd& addMatrix)
{
	assert(inputMatrix.rows() == addMatrix.rows() && inputMatrix.cols() == addMatrix.cols() && "[ERROR]: matrixes must have same dimensions");

    _receivecInput  =  inputMatrix + addMatrix;
    Eigen::MatrixXd normalized  =  LayerNormalization( _receivecInput );

    return normalized;
}


Eigen::MatrixXd AddNorm::Backward(Eigen::MatrixXd& dL_dNormalized)
{
    
    // --- dL_dBetta and dL_dGamma
    for (size_t col = 0; col < _gammas.size(); col++) {

        double dL_dBeta = 0.0;
        double dL_dGamma = 0.0;

        for (size_t row = 0; row < dL_dNormalized.rows(); row++) {
            // --- dL_dBetta
            dL_dBeta  +=  dL_dNormalized(row,col) * _betas[col];

            // --- dL_dGama
            double x  =  _receivecInput(row,col);
            double nii = x - _layerMeans[col];
            dL_dGamma +=  dL_dNormalized(row, col) * (nii / _layerStddev[col]);
        }

        _betas[col]  =  _betas[col]  -  _learningRate * dL_dBeta;
        _gammas[col]  =  _gammas[col]  -  _learningRate * dL_dGamma;
    }
    


    //--- dL_dX
    Eigen::MatrixXd dL_dVariance  =  DL_DVariance( dL_dNormalized );
    Eigen::MatrixXd dL_dNii  =  DL_DNii(dL_dVariance, dL_dNormalized);
    Eigen::MatrixXd dL_dMeans  =  DL_DMeans(dL_dNormalized);
    Eigen::MatrixXd dL_dInputMatrix  =  DL_DInput(dL_dMeans, dL_dNii);


    return dL_dInputMatrix;
}




Eigen::MatrixXd AddNorm::LayerNormalization(Eigen::MatrixXd& addedMatrix)
{
    double epson = 1e-5;

    Eigen::MatrixXd normalized = addedMatrix;

    _layerStddev = std::vector<double>(normalized.cols(), 0.0);
    _layerMeans  =  std::vector<double>(normalized.cols(), 0.0);
    

    for (size_t col = 0; col < addedMatrix.cols(); col++) {
        Eigen::VectorXd column = addedMatrix.col(col);

        _layerMeans[col] = column.mean();
        double variance = (column.array() - _layerMeans[col]).square().mean();
        _layerStddev[col] = std::sqrt(variance + epson);

        normalized.col(col) = (column.array() - _layerMeans[col]) / (_layerStddev[col]);
        normalized.col(col) = (normalized.col(col).array() * _gammas[col] ) + _betas[col];   // DISCOMENT TO USE BETA AND GAMMA
    }

    return normalized;
}







Eigen::MatrixXd AddNorm::DL_DVariance(Eigen::MatrixXd& dL_dNormalized)
{
    Eigen::MatrixXd dL_dVariance = Eigen::MatrixXd(1, dL_dNormalized.cols());

    for (size_t col = 0; col < dL_dNormalized.cols(); col++) {
        double sum  =  0.0;

        for (size_t row = 0; row < dL_dNormalized.rows(); row++) {
            double x = _receivecInput(row,col);
            double nii_element  =  x - _layerMeans[col];

            sum  +=  dL_dNormalized(row,col) * (-nii_element / 2.0*std::pow(_layerStddev[col], 3));
        }

        dL_dVariance(0,col)  =  sum;
    }

    return dL_dVariance;
}


Eigen::MatrixXd AddNorm::DL_DNii(Eigen::MatrixXd& dL_dVariance, Eigen::MatrixXd& dL_dNormalized)
{
    assert(dL_dVariance.rows()==1 && "[ERROR]: dL_dVariance must be a horizontal vector");
    assert(dL_dVariance.cols()==_receivecInput.cols() && "[ERROR]: each col must have their dL_dVariance");


    Eigen::MatrixXd dL_dNii  =  Eigen::MatrixXd(_receivecInput.rows(), _receivecInput.cols());

    double rowSize  =  (double)_receivecInput.rows();

    for (size_t row = 0; row < _receivecInput.rows(); row++) {
        for (size_t col = 0; col < _receivecInput.cols(); col++) {
            double x  =  _receivecInput(row,col);
            double nii  =  x  -  _layerMeans[col];

            dL_dNii(row,col) = (2*nii / rowSize) * dL_dVariance(0, col)   +   dL_dNormalized(row, col) * (_gammas[col]/_layerStddev[col]);
        }
    }

    return dL_dNii;
}


Eigen::MatrixXd AddNorm::DL_DMeans(Eigen::MatrixXd& dL_dNormalized)
{
    Eigen::MatrixXd dL_dMeans = Eigen::MatrixXd(1, dL_dNormalized.cols());

    for (size_t col = 0; col < _receivecInput.cols(); col++) {
        double sum = 0.0;

        for (size_t row = 0; row < _receivecInput.rows(); row++) {
            sum  +=  - dL_dNormalized(row,col) * (_gammas[col] / _layerStddev[col]);
        }

        dL_dMeans(0,col)  =  sum;
    }

    return dL_dMeans;   
}


Eigen::MatrixXd AddNorm::DL_DInput(Eigen::MatrixXd& dL_dMean, Eigen::MatrixXd& dL_dNii)
{
    Eigen::MatrixXd dL_dInput = Eigen::MatrixXd(_receivecInput.rows(), _receivecInput.cols());


    double rowSize  =  (double)_receivecInput.rows();

    for (size_t row = 0; row < _receivecInput.rows(); row++) {
        for (size_t col = 0; col < _receivecInput.cols(); col++) {

            dL_dInput(row,col)  =  dL_dMean(0,col) * (1.0/rowSize)   +   dL_dNii(row,col) * (1.0 - (1.0/rowSize));

        }
    }

    return dL_dInput;
}



