# Long-texts-similarity Repository 
A method that obtains the semantic similarity of pairs of texts by using models for sentence similarity.
This repository consists of the following resources:
* A method for semantic similarity on long texts with random size. The method uses large language models (sentence-transformers). This method mitigates the issues of token capacity and low performance due to the positions of words in the context. See [Method Card](Method_Card.md) 
* The method is implemented using the following models
     * sentence-transformers:
       
            1.- all-MiniLM-L6-v2,
       
            2.- all-MiniLM-L12-v2
       
            3.- all-mpnet-base-v2,

            4.- glove.300D
       
     * Longformer
     * BigBird
     * GPT2
     * BART
* A dataset of random-size texts for semantic similarity created for and aided by the proposed method. The dataset contains 72 documents extracted from Wikipedia.
* Results produced by the method on several models and the proposed dataset.
  
The sentence-transformer models are implemented in a python file, The implementations using Longformer, BigBird and GPT2 have separated python files. Please install each model dependencies before running each implementation of the method. For Installation of the method's implementations please see [Quick_Start](Quick_Start.md), The datasets is found in the folder DATASET, the gold standard is located in the excel file Dataset 72 Docs GoldS-Standard.csv, The list of documents is located at [DatasetListfile.txt](DatasetListfile.txt)
