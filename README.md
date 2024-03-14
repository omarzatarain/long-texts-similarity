# long-texts-similarity
A method that obtains the semantic similarity of pairs of texts by using models for sentence similarity.
This repository consists of the following resources:
* A method for semantic similarity on long texts with random size. The method uses large language models (sentence-transformers). This method mitigates the issues of token capacity and low performance due to the positions of words in the context.
* The method is implemented using the following models
     * sentence-transformers: all-MiniLM-L6-v2, all-mpnet-base-v, 2all-MiniLM-L12-v2
     * Longformer
     * BigBird
     * GPT2
* A dataset of random-size texts for semantic similarity created for and aided by the proposed method.
* Results produced by the method on several models and the proposed dataset
The sentence-transformer models are implemented in a python file, The implementations using Longformer, BigBird and GPT2 have separated python files.
