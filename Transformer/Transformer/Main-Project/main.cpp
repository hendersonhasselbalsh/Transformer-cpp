#include "../utils/basic-includes.h"
#include "../tansformer/Encoder-Decoder-Transformer/Encoder-Decoder-Transformer.h"
#include "../tansformer/builder/TransformerBuilder.h"

// //auxiliar functions and constants used only here on main, not part of transformer
#include "Dictionaries-and-Sentences.h"
#include "Utils-Functions-For-Main.h"



int main(int argc, const char** argv)
{
    std::ofstream outputFile("..\\..\\transformer-output.txt");


    EncodeDecodeTransformer transformer  =  TransformerBuilder()
                                                .EmbeddingSize(64*2*2)
                                                .InputDictionarySize(EN_DICTIONARY.size())
                                                .OutputDictionarySize(PT_DICTIONARY.size())
                                                .Heads(1*2*2*2)
                                                .LearningRate(0.0001)
                                                .Build();


    Eigen::MatrixXd INPUT_WORDS = SentenceToMatrix(ORIGINAL_SENTENCE, EN_DICTIONARY);
    Eigen::MatrixXd CORRECT_OUTPUT = SentenceToMatrix(CORRECT_TRANSLATION, PT_DICTIONARY);



    std::string PREVIOUS_TRANSLATION = "";


    size_t epoch = 0;
    bool correctPredictionNotFount = true;

    while (correctPredictionNotFount && epoch < 50'000) {

        Eigen::MatrixXd encoderInput  =  INPUT_WORDS;
        Eigen::MatrixXd decoderInput  = Eigen::MatrixXd::Zero(1, PT_DICTIONARY.size());
        decoderInput(0,0) = 1.0;


        Eigen::MatrixXd predictedSentence;
        for (size_t predictedWords = 0; predictedWords < CORRECT_OUTPUT.rows()-1; predictedWords++) {

            Eigen::MatrixXd predictedToken  =  transformer.Forward(encoderInput, decoderInput);

            predictedSentence  =  ConcatMatrix(predictedSentence, predictedToken);
            decoderInput  =  GaneratedSentence(decoderInput, predictedToken);
        }

        Eigen::MatrixXd expedtedSentence  =  CORRECT_OUTPUT.block(1, 0, CORRECT_OUTPUT.rows()-1, CORRECT_OUTPUT.cols());
        transformer.Backward(predictedSentence, expedtedSentence);


        //-----------------------------------------------------------------------------------------------
        //                  PRINT SENTENCE
        //-----------------------------------------------------------------------------------------------
        std::string PREDICTED_TRANSLATION = MatrixToSentence(decoderInput, PT_DICTIONARY);

        std::cout << "--------------------------- iteration: " << epoch << " ---------------------------\n\n";
        std::cout << "ORIGINAL SENTENCE:    " << ORIGINAL_SENTENCE << "\n";
        std::cout << "CORRET TANSLATION:    " << CORRECT_TRANSLATION << "\n";
        std::cout << "PREDICTED TANSLATION: " << PREDICTED_TRANSLATION << "\n\n\n";


        // write output
        if (PREVIOUS_TRANSLATION.compare(PREDICTED_TRANSLATION)!=0  &&  outputFile.is_open()) {
            outputFile << "--------------------------- iteration: " << epoch << " ---------------------------\n\n";
            outputFile << "ORIGINAL SENTENCE:    " << ORIGINAL_SENTENCE << "\n";
            outputFile << "CORRET TANSLATION:    " << CORRECT_TRANSLATION << "\n";
            outputFile << "PREDICTED TANSLATION: " << PREDICTED_TRANSLATION << "\n\n\n";
            PREVIOUS_TRANSLATION = PREDICTED_TRANSLATION;
        }
        
        
        //-----------------------------------------------------------------------------------------------
        // break while
        if (PREDICTED_TRANSLATION.compare(CORRECT_TRANSLATION) == 0) {
            correctPredictionNotFount = false;
        }

        epoch++;
    }




    outputFile.close();
    std::cout << "\n\n\n[DEBBUGED - SUCESSO!!!!]\n\n\n";
    return 0;
}


