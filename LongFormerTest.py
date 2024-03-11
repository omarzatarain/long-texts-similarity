
from transformers import LongformerTokenizer
import numpy as np
import spacy
import pandas as pd
import json
import math
import os
import matplotlib.pyplot as plt
from numpy.linalg import norm

def ComparebyIndices(sent1_index , sent2_index, embeddings1, embeddings2):
    print(sent1_index)
    print(sent2_index)
    emb1 = embeddings1[sent1_index]
    emb2 = embeddings1[sent2_index]
    cosine = np.dot(emb1,emb2)/(norm(emb1)*norm(emb2))
    print(cosine)
    return cosine


    
def setNLP():
    nlp = spacy.load("en_core_web_sm")
    return nlp
def getLongFormerTokenizer():
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")   
    return tokenizer

def fillTokens(Text, numtokens):
    for i in range(numtokens): 
        Text = Text + " _"
    return Text

def getLongFormerSimilarity(Text1, Text2, tokenizer):
    emb1 = tokenizer(Text1)["input_ids"]
    emb2 = tokenizer(Text2)["input_ids"]
    if len(emb1)==len(emb2):
        cosine = np.dot(emb1,emb2)/(norm(emb1)*norm(emb2))
        #print("Cosine Similarity:", cosine)
        
    if len(emb1)>len(emb2):   
        filler = len(emb1)-len(emb2)
        Text2 = fillTokens(Text2,filler)
        emb2 = tokenizer(Text2)["input_ids"]
        cosine = np.dot(emb1,emb2)/(norm(emb1)*norm(emb2))
        #print("Cosine Similarity:", cosine)
        
    if len(emb1)<len(emb2):   
        filler = len(emb2)-len(emb1)
        Text1 = fillTokens(Text1,filler) 
        emb1 = tokenizer(Text1)["input_ids"]
        cosine = np.dot(emb1,emb2)/(norm(emb1)*norm(emb2))
        #print("Cosine Similarity:", cosine)
        
    return cosine    
    
def getSentencesbyFile(filename):
    # UPDATE THIS ROUTE ACCORDING TO YOUR PATH
    file = os.getcwd() + '/WORKSPACE/Sentences/'+ filename
    file = file.replace("\\","/") 
    f = open(file, "r", encoding="UTF-8")
    text = f.read()
    sentences = text.split('\n')
    ssize = len(sentences);
    print("Sentences size = " + str(ssize))
    mylist  = list()
    for x in sentences:
            mylist.append(x)
    return mylist

def  validSentenceSpacy(text, nlp):
    # Analyze syntax
   flag = False
   doc = nlp(text)
   for token in doc:
       if token.pos_ == "VERB":
           flag = True  
   return flag

def getTextContent(filename):
    file = os.getcwd() + '/KB/'+ filename
    file = file.replace("\\","/") 
    f = open(file, "r", encoding="UTF-8")
    text = f.read()
    words = text.split(None )
    return(words)
    

def DATASET_AttentionOnChunks(file1, file2, threshold, tokenizer, nlp, chunksize):
    words1 = getTextContent(file1)
    words2 = getTextContent(file2)
    w1_len = len(words1)
    print(w1_len)
    w2_len = len(words2)
    print(w2_len)
    chunks1size = math.ceil(w1_len/chunksize)
    chunks2size = math.ceil(w2_len/chunksize)
    chunks1 = list()
    chunks2 = list()
    embeddings1 = list()
    embeddings2 = list()
    Matrix = [[0 for x in range(chunks1size+1)] for y in range(chunks1size+1)]
    Deciles = [0 for x in range(11)]
    SoftDeciles = [0 for x in range(11)]
    
    # Text 1 chunks
    for i in range(chunks1size):
        text = ''
        for k in range(chunksize):
            if i*chunksize+k < w1_len:
                text = text + ' '+ words1[i*chunksize+k]
        chunks1.append(text)
        emb1 = tokenizer(text)["input_ids"]
        embeddings1.append(emb1)
        print(text)
        print('\n\n')
        
    # Text 2 chunks
    for i in range(chunks2size):
        text = ''
        for k in range(chunksize):
            if i*chunksize+k < w2_len:
                text = text + ' '+ words2[i*chunksize+k]
        chunks2.append(text)
        emb2 = tokenizer(text)["input_ids"]
        embeddings2.append(emb2)
        print(text)
        print('\n\n')
        
    for i in range(chunks1size):
        for k in range(chunks2size):
            res = getLongFormerSimilarity(chunks1[i], chunks2[k], tokenizer)
            if res > threshold:
                        if res > Matrix[i][k]:
                             Matrix[i][k] = res
                        if res < 0.1:
                             Deciles[1] = Deciles[1]+1
                        if res >= 0.1 and res <= 0.2:
                             Deciles[2] = Deciles[2]+1
                        if res > 0.2 and res <= 0.3:
                             Deciles[3] = Deciles[3]+1
                        if res > 0.3 and res <= 0.4:
                             Deciles[4] = Deciles[4]+1
                        if res >= 0.4 and res <= 0.5:
                             Deciles[5] = Deciles[5]+1
                        if res > 0.5 and res <= 0.6:
                             Deciles[6] = Deciles[6]+1
                        if res > 0.6 and res <= 0.7:
                             Deciles[7] = Deciles[7]+1
                        if res >= 0.7 and res <= 0.8:
                             Deciles[8] = Deciles[8]+1
                        if res > 0.8 and res <= 0.9:
                             Deciles[9] = Deciles[9]+1
                        if res > 0.9 and res <= 1.0:
                            Deciles[10] = Deciles[10]+1                     
   
    # Produce the rest of variables
    #Compute support   
   
    if chunks1size < chunks2size:
        supportnum = chunks1size
    else:
        supportnum = chunks2size;
    sum1 = 0
    supprt =10
    while supprt > 0 and sum1 < supportnum:
        sum1 = sum1 + Deciles[supprt]
        supprt = supprt -1; 
    
    supprt = supprt + 1;

    # Compute spanning
    span =0 
    for i  in range(11):
        if Deciles[i] > 0:
            span =i; 
    
    # Compute soundness
    sum1 = 0; 
    for i in range(supprt, span+1):
       sum1 = sum1+ Deciles[i]
       
    for i in range(supprt, span+1):  
        SoftDeciles[i] = Deciles[i]/sum1

   # Select the sound  level as
    if supprt ==10:
        soundness = 10;
        SoftDeciles[supprt-1] = 0 
    else:
         soundness = supprt+1;
         SoftDeciles[supprt] = 0

    max =  SoftDeciles[soundness]
    for  i in range(supprt, span+1):  
         if SoftDeciles[i] > max:
           max = SoftDeciles[i]
           soundness = i; 
    print(Deciles)
    PairData = {"file1": file1, "file2": file2, "Matrix": Matrix, "mSize1":  chunks1size, "mSize2":  chunks2size, "Deciles": Deciles, "SoftDeciles": SoftDeciles,"Support": supprt, "Spanning": span, "Soundness": soundness}
    print(PairData)
    return PairData
     


def DATASET_AttentionOnSentences(file1, file2, threshold, tokenizer, nlp):
    filea = file1
    fileb = file2
    file1 ='S_'+ file1;
    file2 ='S_'+ file2;
    Sent1 = getSentencesbyFile(file1)
    Sent2 = getSentencesbyFile(file2)
    size1 = len(Sent1)
    size2 = len(Sent2)
    Matrix = [[0 for x in range(size2+1)] for y in range(size1+1)]
    Deciles = [0 for x in range(11)]
    SoftDeciles = [0 for x in range(11)]
    
# Produce an array of valid sentences on text1
    SVO1 = [0 for x in range(size1)]
    SVO2 = [0 for x in range(size2)]
    for a in range(size1):
         text1 = Sent1[a]
         res1 = validSentenceSpacy(text1, nlp)
         if res1 == True: 
             SVO1[a] = 1
    #print(SVO1)        
    for a in range(size2):
         text2 = Sent2[a]
         res2 = validSentenceSpacy(text2, nlp)
         if res2 == True: 
             SVO2[a] = 1
    #print(SVO2) 
    # Complete processing here
    for a in range(1, size1):
        # Check if the sentence contains verb if not discard
        if SVO1[a] == 1: 
            Text1 = Sent1[a]
            for b in range(1, size2):
                # check if the sentence contains verb if not discard         
                if SVO2[b] == 1: 
                    Text2 = Sent2[b]
                    res = getLongFormerSimilarity(Text1, Text2, tokenizer)
                    if res > threshold:
                        if res > Matrix[a][b]:
                             Matrix[a][b] = res
                        if res < 0.1:
                             Deciles[1] = Deciles[1]+1
                        if res >= 0.1 and res <= 0.2:
                             Deciles[2] = Deciles[2]+1
                        if res > 0.2 and res <= 0.3:
                             Deciles[3] = Deciles[3]+1
                        if res > 0.3 and res <= 0.4:
                             Deciles[4] = Deciles[4]+1
                        if res >= 0.4 and res <= 0.5:
                             Deciles[5] = Deciles[5]+1
                        if res > 0.5 and res <= 0.6:
                             Deciles[6] = Deciles[6]+1
                        if res > 0.6 and res <= 0.7:
                             Deciles[7] = Deciles[7]+1
                        if res >= 0.7 and res <= 0.8:
                             Deciles[8] = Deciles[8]+1
                        if res > 0.8 and res <= 0.9:
                             Deciles[9] = Deciles[9]+1
                        if res > 0.9 and res <= 1.0:
                            Deciles[10] = Deciles[10]+1                     
   
    # Produce the rest of variables
    #Compute support   
   
    if size1 < size2:
        supportnum = size1
    else:
        supportnum = size2;
    sum1 = 0
    supprt =10
    while supprt > 0 and sum1 < supportnum:
        sum1 = sum1 + Deciles[supprt]
        supprt = supprt -1; 
    
    supprt = supprt + 1;

    # Compute spanning
    span =0 
    for i  in range(11):
        if Deciles[i] > 0:
            span =i; 
    
    # Compute soundness
    sum1 = 0; 
    for i in range(supprt, span+1):
       sum1 = sum1+ Deciles[i]
       
    for i in range(supprt, span+1):  
        SoftDeciles[i] = Deciles[i]/sum1

   # Select the sound  level as
    if supprt ==10:
        soundness = 10;
        SoftDeciles[supprt-1] = 0 
    else:
         soundness = supprt+1;
         SoftDeciles[supprt] = 0

    max =  SoftDeciles[soundness]
    for  i in range(supprt, span+1):  
         if SoftDeciles[i] > max:
           max = SoftDeciles[i]
           soundness = i; 
    print(Deciles)
    PairData = {"file1": filea, "file2": fileb, "Matrix": Matrix, "mSize1": size1, "mSize2": size2, "Deciles": Deciles, "SoftDeciles": SoftDeciles,"Support": supprt, "Spanning": span, "Soundness": soundness}
    print(PairData)
    return PairData
        

nlp = setNLP()
chunksize = 1024
tokenizer = getLongFormerTokenizer()
#DATASET_AttentionOnChunks('2022 Russian invasion of Ukraine Brittanica.txt', '2022 Russian invasion of Ukraine Brittanica.txt', 0.3, tokenizer, nlp, chunksize)
DATASET_AttentionOnSentences('2022 Russian invasion of Ukraine Brittanica.txt', '2022 Russian invasion of Ukraine Brittanica.txt', 0.3, tokenizer, nlp)


Text1 = "The woman called her friend"
Text2 = "The girl told her mom to buy some groceries"

getLongFormerSimilarity(Text1, Text2, tokenizer)

#hello2 = tokenizer(" MyText")["input_ids"]
#print(hello2)