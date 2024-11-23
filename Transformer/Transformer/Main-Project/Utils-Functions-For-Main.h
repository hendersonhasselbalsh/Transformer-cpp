#pragma once

#include "../utils/basic-includes.h"



Eigen::MatrixXd ConcatMatrix(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B)
{
    //assert(A.cols() == B.cols());
    Eigen::MatrixXd result;

    if (A.size() == 0) {
        result = B;
    } else {
        result  =  Eigen::MatrixXd(A.rows() + B.rows(), A.cols());

        result.block(0, 0, A.rows(), A.cols()) = A;
        result.block(A.rows(), 0, B.rows(), B.cols()) = B;
    }

    return result;
}

Eigen::MatrixXd GaneratedSentence(const Eigen::MatrixXd& sentence, const Eigen::MatrixXd& predictedToken)
{
    size_t maXIndice = 1000;
    Eigen::MatrixXd token  =  Eigen::MatrixXd::Zero(1, predictedToken.cols());
    predictedToken.row(0).maxCoeff(&maXIndice);
    token(0, maXIndice) = 1.0;


    Eigen::MatrixXd newSentence = ConcatMatrix(sentence, token);

    return newSentence;
}


Eigen::MatrixXd WordToToken(std::string& word, std::vector<std::string>& dictionary)
{
    size_t dictionarySize = dictionary.size();
    Eigen::MatrixXd token  =  Eigen::MatrixXd::Zero(1, dictionarySize);

    size_t index = 100'000'000;
    for (size_t i = 0; i < dictionarySize; i++) {
        if (word.compare(dictionary[i]) == 0) { index = i; }
    }

    if (index > dictionary.size()) {
        throw std::runtime_error("token not found");
    } else {
        token(0, index) = 1.0;
    }

    return token;
}

Eigen::MatrixXd SentenceToMatrix(std::string& sentence, std::vector<std::string>& dictionary)
{
    std::vector<std::string> sentenceWord  =  Utils::SplitString(sentence, " ");

    Eigen::MatrixXd sentenceMatrix;
    for (auto word : sentenceWord) {
        Eigen::MatrixXd token  =  WordToToken(word, dictionary);
        sentenceMatrix  =  ConcatMatrix(sentenceMatrix, token);
    }

    return sentenceMatrix;
}


std::string MatrixToSentence(Eigen::MatrixXd& mat, std::vector<std::string>& dictionary)
{
    std::string sentence  =  "";

    for (size_t row = 0; row < mat.rows(); row++) {
        size_t maXIndice = 0;
        mat.row(row).maxCoeff(&maXIndice);
        sentence  +=  dictionary[maXIndice]  +  " ";
    }

    return sentence;
}



