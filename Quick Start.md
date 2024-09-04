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

# 3.1.-Sentence Transformers: Python_LongTextsSimilarity.py

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

## 3.2.- LongFormer

nlp = setNLP()
chunksize = 1024

tokenizer = getLongFormerTokenizer()

#DATASET_AttentionOnChunks('2022 Russian invasion of Ukraine Brittanica.txt', '2022 Russian invasion of Ukraine Brittanica.txt', 0.3, tokenizer, nlp, chunksize)

DATASET_AttentionOnSentences('2022 Russian invasion of Ukraine Brittanica.txt', '2022 Russian invasion of Ukraine Brittanica.txt', 0.3, tokenizer, nlp)


## 3.3.- BigBird

nlp= setNLP()

chunksize = 16

DATASET_AttentionOnSentencesBigBird('2022 Russian invasion of Ukraine Brittanica.txt', '2022 Russian invasion of Ukraine.txt', 0.3, tokenizer, nlp)

#DATASET_AttentionOnChunks('2022 Russian invasion of Ukraine Brittanica.txt', '2022 Russian invasion of Ukraine.txt', 0.3, tokenizer, nlp, chunksize)

## 3.4.- BART

nlp= setNLP()

chunksize = 16 #16

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

#DATASET_AttentionOnChunks('2022 Russian invasion of Ukraine Brittanica.txt', '2022 Russian invasion of Ukraine Brittanica.txt', 0.3, tokenizer, nlp, chunksize)

DATASET_AttentionOnSentences('2022 Russian invasion of Ukraine Brittanica.txt', '2022 Russian invasion of Ukraine Brittanica.txt', 0.3, tokenizer, nlp)

## 3.5.- GPT2

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

nlp= setNLP()

DATASET_AttentionOnSentencesGPT2('2022 Russian invasion of Ukraine Brittanica.txt', '2022 Russian invasion of Ukraine.txt', 0.3, tokenizer, nlp)

#chunksize = 16

#DATASET_AttentionOnChunks('2022 Russian invasion of Ukraine Brittanica.txt', '2022 Russian invasion of Ukraine.txt', 0.3, tokenizer, nlp, chunksize)


# 4.- Systematic Comparison

 The systematic comparison of the pairs for each model requires  different setups due to specific features of each model
 
## 4.1.- Sentence Transformers

nlp = setNLP()
##CHOOSE THE MODEL
themodel = 'all-MiniLM-L6-v2'
#themodel = 'all-MiniLM-L12-v2'
#themodel = 'all-mpnet-base-v2'
#themodel = 'sentence-transformers/average_word_embeddings_glove.6B.300d'

model = setModel(themodel)
#CREATION OS SENTENCES VERSIONS
fileSet= 'DatasetListfile.txt'
#preprocessDataset(fileSet, nlp)

#PARAMETERS FOR CLASSIFICATION

#BASELINE
#b = 0.2; 
#c = 0.2; 
#alpha = 0.5# 0.68;
#beta = 0.6 #0.78;
#gamma = 0.8 # 0.82;
#delta = 0.9 #0.92;

#TUNED
b = 0.1; 
c = 0.1; 
alpha = 0.75# 0.68;
beta = 0.77 #0.78;
gamma = 0.79 # 0.82;
delta = 0.85 #0.92;

#SELF COMPARISON
#SystematicPairClassification('DatasetListfile.txt', model,  nlp, a,b,c,d,alpha, beta, gamma, delta)
SystematicSelfClassification('DatasetListfile.txt', model,  nlp, a,b,c,d,alpha, beta, gamma, delta)

## 4.2.- LongFormer



## 4.3.- BigBird



## 4.4.- BART


## 4.5.- GPT2


# 5.- Assessment against the Gold Standard 
 The assessment of the results of each model against the gold standard  in the file "Dataset 72 Docs Gold-Standard.csv"

## 5.1.- Sentence Transformers:  Python_LongTextsSimilarity.py
##APPLY ANALYSIS, The folder can be set to other sentence models results

folder = 'RESULTS_all-MiniLM-L6-v2'

SystematicPairReClassification('DatasetListfile.txt' ,folder, model,  nlp, a,b,c,d,alpha, beta, gamma, delta)

MyDict = GetAnalysis('DatasetListfile.txt', folder)

print(MyDict)

#results = readResultsJson('GlobalResults.json', folder )

#ResultsMatrix = results.get("Matrix")

print(ResultsMatrix)

filename = "Dataset 72 Docs Gold-Standard.xlsx"

path = os.getcwd()

sheet = "Hoja2"

goldstd = importExcelFile(filename, path, sheet)

goldstdmatrix = pd.read_csv('C:\RESEARCH PROJECTS\MIXED_ARCHITECTURE\Dataset 72 Docs Gold-Standard.csv', header=0)

print(goldstdmatrix.iloc[0][1])

CompMatrix = CompareGold_STD(ResultsMatrix, goldstdmatrix, folder)


## 5.2.- LongFormer
## 5.3.- BigBird
## 5.4.- BART
## 5.5.- GPT2

# Tuning
##FINE TUNNING
baseline = { "alpha": alpha, "b": b, "beta": beta,"c": c, "gamma": gamma, "delta": delta}
#folder = 'RESULTS_all-MiniLM-L6-v2'
#folder = 'RESULTS_all-MiniLM-L12-v2'
#folder = 'RESULTS_all-mpnet-base-v2'
#folder = 'RESULTS_glove.300D'
folder = 'RESULTS_CURRENT'
listfile = 'DatasetListfile.txt'
MyDict = GetAnalysis(listfile, folder)
results = readResultsJson('GlobalResults.json', folder)
ResultsMatrix = results.get("Matrix")
print(ResultsMatrix)
GSMatrix = pd.read_csv('C:\RESEARCH PROJECTS\MIXED_ARCHITECTURE\Dataset 72 Docs Gold-Standard.csv', header=0)
print(GSMatrix.iloc[0][1])
#AssesmentTuning(listfile, folder, baseline, GSMatrix, ResultsMatrix)

#AutoTuning(listfile, folder, baseline, GSMatrix, ResultsMatrix)
AutoAssessment(listfile, folder, baseline, GSMatrix, ResultsMatrix)

#SELF COMPARISON SHUFFLED
#SystematicSelfClassificationShuffled('DatasetListfile.txt', model,  nlp, a,b,c,d,alpha, beta, gamma, delta)
#SystematicPairClassificationShuffled('DatasetListfile.txt', model,  nlp, a,b,c,d,alpha, beta, gamma, delta)



