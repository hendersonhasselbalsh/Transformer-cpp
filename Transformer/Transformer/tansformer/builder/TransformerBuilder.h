#pragma once

#include "../Encoder-Decoder-Transformer/Encoder-Decoder-Transformer.h"



class TransformerBuilder {

	private:
		size_t _embeddingSize;
		size_t _inputDictionarySize;
		size_t _outputDictionarySize;
		size_t _heads;
		double _learningRate;

	public:
		TransformerBuilder();

		TransformerBuilder EmbeddingSize(size_t size);
		TransformerBuilder InputDictionarySize(size_t size);
		TransformerBuilder OutputDictionarySize(size_t size);
		TransformerBuilder Heads(size_t size);
		TransformerBuilder LearningRate(double rate);

		EncodeDecodeTransformer Build();

};

