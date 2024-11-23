#include "utils.h"


double Utils::RandomNormalDistributionValue(double min, double max)
{
// Create a random number generator
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Mersenne Twister engine

    // Define mean and standard deviation for the normal distribution
    double mean = (min + max) / 2.0;
    double stddev = (max - min) / 6.0; // approximately 99.7% within [a, b]

    // Create a normal distribution
    std::normal_distribution<double> d(mean, stddev);

    // Generate random numbers within the range [a, b]
    double number = 0.0;
    do {
        number = d(gen);
    } while (number < min || number > max);

    return number;
}


double Utils::RandomUniformDistribution(double min, double max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribuicao(min, max);
    return distribuicao(gen);
}


double Utils::ScalarProduct(std::vector<double> inputs, std::vector<double> weights)
{
    assert(inputs.size() == weights.size());

    double sum = 0.0;

    for (size_t i = 0; i < weights.size(); i++) {
        sum += inputs[i] * weights[i];
    }

    return sum;
}


double Utils::Normalize(double value, double min, double max, double initial, double final)
{
    return ( final - initial) * ( (value - min) / (max - min) ) + initial;
}



double Utils::Mean(std::vector<double> values)
{
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    return mean;
}



double Utils::Variance(std::vector<double> values, double mean)
{
    double somaQuadradosDiferencas = std::accumulate(values.begin(), values.end(), 0.0, [mean](double acumulador, double valor) {
                                                         double diferenca = valor - mean;
                                                         return acumulador + diferenca * diferenca;
                                                     });

    double variance = std::sqrt(somaQuadradosDiferencas / values.size());

    if (variance == 0.0) { variance = std::numeric_limits<double>::min(); }

    return variance;
}



std::vector<double> Utils::BatchNormalization(std::vector<double> originalInput)
{
    double mean =  Utils::Mean(originalInput);
    double variance = Utils::Variance(originalInput, mean);

    for (auto& input : originalInput) {
        input = (input - mean) / variance;
    }

    return originalInput;
}



std::vector<double> Utils::FlatMatrix(Eigen::MatrixXd input)
{
    Eigen::ArrayXXd arr  =  input.array();
    std::vector<double> vec(arr.data(), arr.data() + arr.size());
    /*vec.reserve(input.size());

    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            vec.push_back(input(i, j));
        }
    }*/

    return vec;
}

Eigen::MatrixXd Utils::ReshapeMatrix(std::vector<double> arr, size_t rows, size_t cols)
{
    assert(arr.size() == rows * cols);
    Eigen::MatrixXd reshapedMatrix = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(arr.data(), rows, cols);

    /*Eigen::MatrixXd reshapedMatrix  =  Eigen::MatrixXd::Zero(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            reshapedMatrix(i, j)  =  gradients[i*cols + j];
        }
    }*/

    return reshapedMatrix;
}

std::vector<std::string> Utils::SplitString(const std::string& input, const std::string& delimiter)
{
    std::vector<std::string> result;
    size_t start = 0;
    size_t end = input.find(delimiter);

    while (end != std::string::npos) {
        result.push_back(input.substr(start, end - start));
        start = end + delimiter.length();
        end = input.find(delimiter, start);
    }

    result.push_back(input.substr(start, end));  // Handle the last token

    return result;
}


IActivationFunction* Utils::StringToActivationFunction(std::string functionName)
{
    
	if (functionName == "ReLU")                { return new ReLU(); }
	else if (functionName == "LeakyReLU")      { return new LeakyReLU(); }
	else if (functionName == "Tanh")           { return new Tanh(); }
	else if (functionName == "NormalizedTanh") { return new NormalizedTanh(); }
	else if (functionName == "Sigmoid")        { return new Sigmoid(); }
	else if (functionName == "AdaptedSigmoid") { return new AdaptedSigmoid(); }
	else if (functionName == "Linear")         { return new Linear(); }
	else { return nullptr; }
}


std::vector<double> Utils::Add(std::vector<double> a, std::vector<double> b)
{
    assert(a.size() == b.size());

    std::vector<double> result = std::vector<double>(a.size(), 0.0);
    for (size_t i = 0; i < a.size(); i++) {
        result[i]  =  a[i] + b[i];
    }

    return result;
}

std::vector<std::vector<MLPTrainigData>> Utils::ShuffleBatch(std::vector<std::vector<MLPTrainigData>> batch)
{
    std::vector<MLPTrainigData> set  =  std::vector<MLPTrainigData>(batch[0]);
    for (size_t i = 1; i < 5; i++) {
        set.insert( set.end(), batch[i].begin(), batch[i].end() );
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(set.begin(), set.end(), g);

    size_t batchIndex  =  0;
    size_t batchSize  =  set.size() / 5;
    size_t trainingSetSize  =  set.size();
    std::vector<std::vector<MLPTrainigData>> batchSet  =  std::vector<std::vector<MLPTrainigData>>((size_t)5);

    for (auto& batch : batchSet) {
        for (size_t i = 0; i < batchSize && batchIndex * batchSize + i < trainingSetSize; i++) {
            size_t index = batchIndex * batchSize + i;
            batch.push_back( {set[index].INPUT, set[index].LABEL} );
        }
        batchIndex++;
    }

    return batchSet;
}

std::vector<std::vector<MLPTrainigData>> Utils::ShuffleBatch(std::vector<MLP_DATA> trainigSet, size_t batchSize, std::function<std::vector<double>(size_t)> ParseLabelToVector)
{
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(trainigSet.begin(), trainigSet.end(), g);

    size_t batchIndex  =  0;
    size_t i = 0;
    size_t trainingSetSize  =  trainigSet.size();
    std::vector<std::vector<MLPTrainigData>> batchSet  =  std::vector<std::vector<MLPTrainigData>>(trainigSet.size() / batchSize);

    for (auto& batch : batchSet) {
        for (i = 0; i < batchSize && batchIndex * batchSize + i < trainingSetSize; i++) {
            size_t index = batchIndex * batchSize + i;
            std::vector<double> label  =  ParseLabelToVector( trainigSet[index].labelIndex );
            batch.push_back({ trainigSet[index].input, label });
        }
        batchIndex++;
    }

    if (batchIndex * batchSize < trainingSetSize) {
        auto& lastBatch  =  batchSet[batchSet.size()-1];
        
        for (size_t index = batchIndex * batchSize; index < trainingSetSize; index++) {
            std::vector<double> label  =  ParseLabelToVector( trainigSet[index].labelIndex );
            lastBatch.push_back({ trainigSet[index].input, label });
        }
    }

    return batchSet;
}


std::vector<std::vector<MLPTrainigData>> Utils::ShuffleBatch(std::vector<std::vector<MLPTrainigData>> batchs, size_t batchSize)
{
    std::vector<MLPTrainigData> trainigSet  =  std::vector<MLPTrainigData>(batchs[0]);
    for (size_t i = 1; i < batchs.size(); i++) {
        trainigSet.insert( trainigSet.end(), batchs[i].begin(), batchs[i].end());
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(trainigSet.begin(), trainigSet.end(), g);

    size_t batchIndex  =  0;
    size_t trainingSetSize  =  trainigSet.size();
    std::vector<std::vector<MLPTrainigData>> batchSet  =  std::vector<std::vector<MLPTrainigData>>(trainigSet.size() / batchSize);

    for (auto& batch : batchSet) {
        for (size_t i = 0; i < batchSize && batchIndex * batchSize + i < trainingSetSize; i++) {
            size_t index = batchIndex * batchSize + i;
            batch.push_back({ trainigSet[index].INPUT, trainigSet[index].LABEL });
        }
        batchIndex++;
    }

    if (batchIndex * batchSize < trainingSetSize) {
        auto& lastBatch  =  batchSet[batchSet.size()-1];

        for (size_t index = batchIndex * batchSize; index < trainingSetSize; index++) {
            lastBatch.push_back({ trainigSet[index].INPUT, trainigSet[index].LABEL });
        }
    }

    return batchSet;
}



void Utils::CalculateMeanVector(std::vector<MLPTrainigData> trainigSet, std::vector<double>* meansResult)
{
    size_t trainingSetSize = trainigSet.size();
    size_t inputSize = trainigSet[0].INPUT.size();
    assert(meansResult->size() == inputSize);

    for (size_t i = 0; i < trainingSetSize; i++) {
        for (size_t meanIndex = 0; meanIndex < inputSize; meanIndex++) {
            (*meansResult)[meanIndex]  +=  trainigSet[i].INPUT[meanIndex] / (double)trainingSetSize;
        }
    }

    meansResult->insert( meansResult->begin(), 1.0 );
}

void Utils::CalculateDeviationVector(std::vector<MLPTrainigData> trainigSet, std::vector<double>* means, std::vector<double>* deviationsResult)
{
    size_t trainingSetSize = trainigSet.size();
    size_t inputSize = trainigSet[0].INPUT.size();
    assert(deviationsResult->size() == inputSize);

    for (size_t i = 0; i < trainingSetSize; i++) {
        for (size_t j = 0; j < inputSize; j++) {
            double diff  =  trainigSet[i].INPUT[j]  -  (*means)[j];

            (*deviationsResult)[j]  +=  (diff * diff) / (double)trainingSetSize; /**/
        }
    }

    for (auto& dev : (*deviationsResult)) {
        dev  =  std::sqrt(dev + DBL_MIN);
    }

    deviationsResult->insert( deviationsResult->begin(), 0.0 );
}


void Utils::CalculateMeanVector(std::vector<MLP_DATA> trainigSet, std::vector<double>* meansResult)
{
    size_t trainingSetSize = trainigSet.size();
    size_t inputSize = trainigSet[0].input.size();
    assert(meansResult->size() == inputSize);

    for (size_t i = 0; i < trainingSetSize; i++) {
        for (size_t meanIndex = 0; meanIndex < inputSize; meanIndex++) {
            (*meansResult)[meanIndex]  +=  trainigSet[i].input[meanIndex] / (double)trainingSetSize;
        }
    }
}

void Utils::CalculateDeviationVector(std::vector<MLP_DATA> trainigSet, std::vector<double>* means, std::vector<double>* deviationsResult)
{
    size_t trainingSetSize = trainigSet.size();
    size_t inputSize = trainigSet[0].input.size();
    assert(deviationsResult->size() == inputSize);

    for (size_t i = 0; i < trainingSetSize; i++) {
        for (size_t j = 0; j < inputSize; j++) {
            double diff  =  trainigSet[i].input[j]  -  (*means)[j];

            (*deviationsResult)[j]  +=  (diff * diff) / (double)trainingSetSize; /**/
        }
    }

    for (auto& dev : (*deviationsResult)) {
        dev  =  std::sqrt(dev + DBL_MIN);
    }
}



void Utils::BatchNorm(std::vector<double>* inputs, std::vector<double>* means, std::vector<double>* devs, double* alpha, double* beta)
{
    size_t index = 0;

    inputs->erase( (*inputs).begin() );

    for (auto& input : (*inputs)) {
        double mean = (*means)[index];
        double dev = (*devs)[index++];                         // incrementa apos retornar o elemento do indice

        input  =  (input - mean) / (dev);

        if (alpha != nullptr  &&  beta != nullptr) {
            input  =  (*alpha) * input  +  (*beta);
        }
    }

    inputs->insert( inputs->begin(), 1.0 );
}

void Utils::DataNorm(std::vector<double>* inputs, std::vector<double>* means, std::vector<double>* devs)
{
    size_t index = 0;

    for (auto& input : (*inputs)) {

        double originalInput_DEBUG = input;

        double mean = (*means)[index];
        double dev = (*devs)[index++];                         // incrementa apos retornar o elemento do indice

        if (dev != 0.0  &&  dev >  1e-5) {
            input  =  (input - mean) / dev;
            //if (std::abs((originalInput_DEBUG - input)/originalInput_DEBUG) > 5.0) { input = mean; }
        }

        if (std::abs(input) > 10.0) {
            int stopDebug = 0;     // input too big
        }
    }
}



void Utils::ScalateAndShift(std::vector<double>* inputs, double* alpha, double* beta)
{
    assert(alpha!=nullptr && beta!=nullptr  &&  "[ERROR]: alpha and beta must not be null");

    for (auto& input : (*inputs)) {
        input  =  (*alpha) * input + (*beta);
    }
}



Eigen::MatrixXd Utils::Rotate_180Degree(Eigen::MatrixXd& matrix)
{
    Eigen::MatrixXd flippedMatrix = matrix.colwise().reverse().rowwise().reverse();
    return flippedMatrix;
}



Eigen::MatrixXd Utils::ScalateMatrix(Eigen::MatrixXd& mat, double a, double b) 
{
    Eigen::MatrixXd scaledMat = mat;  // Cria uma cópia para armazenar os valores escalados

    for (int i = 0; i < mat.rows(); ++i) {
        double minVal = mat.row(i).minCoeff();  // Valor mínimo da linha
        double maxVal = mat.row(i).maxCoeff();  // Valor máximo da linha

        // Verifica se maxVal e minVal são iguais para evitar divisão por zero
        if (maxVal == minVal) {
            // Caso especial onde todos os elementos da linha são iguais
            scaledMat.row(i).setConstant((a + b) / 2);
        } else {
            // Escala a linha para o intervalo [a, b]
            scaledMat.row(i) = (b - a) * (mat.row(i).array() - minVal) / (maxVal - minVal) + a;
        }
    }

    return scaledMat;
}

