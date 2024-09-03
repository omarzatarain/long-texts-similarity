# 1- Installation of the  Models tested with the Method for Long Text Similarity
The method is implemented on several language models, based on the features  of each model, the method can be customized for using  GPU or CPU only processuing. Currently the  method  tested with the following  language models:
1.- Sentence Transformers
2.- LongFormer
3.- BigBird
4.- BART
5.- GPT2

## 1.1- Implementation on Sentence Transformers 
 Implementation of the method: Python_LongTextsSimilarity.py
### Dependencies


## 1.2.- Implementation on LongFormer  
 Implementation of the method:  LongFormerTest.py 
 Python Version 3.10.11  with the following package dependencies installed
### Dependencies
 * transformers
 * LongformerModel
 * LongformerTokenizer
 * LongformerConfig
 * numpy
 * numpy.linalg
 * norm
 * torch
 * spacy
 * pandas
 * json
 * math
 * os

## 1.3.- Implementation on BigBird
 Implementation of the method:  TestBigBird.py 
 Python Version 3.10.11  with the following package dependencies installed
### Dependencies

## 1.4.- Implementation on BART
 Implementation of the method:  BART.py 
 Python Version 3.10.11  with the following package dependencies installed
### Dependencies
* transformers
* BartTokenizer
* numpy
* numpy.linalg
* norm 
* torch
* spacy
* pandas 
* json
* math
* os

## 1.5.- Implementation on GPT2
 Implementation of the method: GPT2-Test.py 
 Python Version 3.10.11  with the following package dependencies installed
### Dependencies
* transformers 
* GPT2Tokenizer
* GPT2Model
* numpy 
* numpy.linalg
* norm
* os
* pandas 
* json
* math
* spacy
* torch

  # 2.- Dataset and preprocessed versions
  ## 2.1.- Dataset
  The oroginal dataset of documents includes 72 samples extracted from Wikipedia, the folder DATASET contains a copy in text format. The names of the samples  are enlisted in the file DatasetListfile.txt
  
  ## 2.2- Preprocessing of texts  in sentences
   The dataset is preprocessed by sentences for use with sentence transformers, the folder Sentences contains the versions 

  
  ## 2.3.- Preprocessing of texts in chunks of fixed size
  

# 3.- Comparison of a pairs of texts

# EXAMPLE OF USING A SINGLE COMPARISON USING  SENTENCE TRANSFORMER: Python_LongTextsSimilarity.py

 #CHOOSE ONE OF THE MODELS
 
#themodel = 'all-MiniLM-L6-v2'

#themodel = 'all-MiniLM-L12-v2'

#themodel = 'all-mpnet-base-v2'

##themodel = 'sentence-transformers/average_word_embeddings_glove.6B.300d'

file1 = '2022 Russian invasion of Ukraine.txt'

file2 = '2022 Russian invasion of Ukraine.txt'

Pairdata = DATASET_AttentionOnSentences(file1, file2 , 0.0, themodel, nlp)

SentPairs = selectRepresentativePairs(Pairdata)

saveResults(Pairdata, SentPairs)

data = readResults('Sent_'+ file1 + "_"+ file2 +'.json')

print(data)


# 4.- Systematic Comparison

# 
