
from transformers import  LongformerModel, LongformerTokenizer,  LongformerConfig
import numpy as np
import torch
import spacy
import pandas as pd
import json
import math
import os
#import matplotlib.pyplot as plt
from numpy.linalg import norm


def defuzzyfication(soundness, alpha, b, beta, c, gamma, delta):
    # Test for the fuzzy sets
    NR_membership = 1- (1/ (1+ math.exp(-alpha*10*(soundness*10-alpha*10))))
    print("NR_membership")
    print(NR_membership)
    CR_membership =  math.exp(-(soundness-beta)*(soundness-beta)/2*b*b)
    print("CR_membership")
    print(CR_membership)
    ST_membership =  math.exp(-(soundness-gamma)*(soundness-gamma)/2*c*c)
    print("ST_membership")
    print(ST_membership)    
    I_membership = (1/ (1+ math.exp(-delta*10*(soundness*10-delta*10))))
    print("I_membership")
    print(I_membership)  
    if  NR_membership >= CR_membership and NR_membership > ST_membership and  NR_membership > I_membership:
        label = "NON-RELATED"
    if  CR_membership > NR_membership and CR_membership >= ST_membership and  CR_membership > I_membership:
        label = "CONCEPT-RELATED"
    if  ST_membership > CR_membership and ST_membership > NR_membership and  ST_membership >= I_membership:
        label = "SAME-TOPIC" 
    if  I_membership > ST_membership and I_membership > CR_membership and  I_membership > NR_membership:
        label = "IDENTICAL"   
    return label

def selectRepresentativePairs(PairData):
    print(PairData)
    Matrix= PairData.get("Matrix")
    Deciles = PairData.get("Deciles")
    Support = PairData.get("Support")
    Spanning = PairData.get("Spanning")
    Soundness = PairData.get("Soundness")
    #size1 = PairData.get("mSize1")
    size1 = len(Matrix)
    print(size1)
    size2 = PairData.get("mSize2")
    size2 = int(size2)
    # Collect the number of representative pairs of sentences from deciles
    mysum = 0
    for i in range(Soundness-1, 11):
        mysum = mysum+ Deciles[i]
    # Create the array of pairs indices
    PairA = [0 for x in range(3*mysum+1)]
    PairB = [0 for x in range(3*mysum+1)]
    PairValue = [0 for x in range(3*mysum +1)]
    # Select the relevant pairs' indices and record them
    counter =0
    
    for i in range(1, size1):
        for k in range(1, size2):
            if Matrix[i][k]>=(Soundness)/10:
                counter = counter+1;
                PairA[counter] = i
                PairB[counter] = k
                PairValue[counter] = Matrix[i][k]
                
    SentPairs ={"Index1":PairA,"Index2":PairB, "Value": PairValue, "Counter": counter}
    # retrieve the array of indices 
    #print(SentPairs)   
    return  SentPairs


def saveResults(PairData, SentPairs, PairClass):
    file1 = PairData.get("file1");
    file2 = PairData.get("file2");
    Deciles = PairData.get("Deciles")
    SoftDeciles = PairData.get("SoftDeciles")
    Support = PairData.get("Support")
    Spanning = PairData.get("Spanning")
    Soundness = PairData.get("Soundness")
    Relation = PairClass.get("Relation")
    Membership = PairClass.get("Membership")

    Class ={"Doc1": file1, "Doc2": file2, "Relation": Relation," Membership":  Membership, "Soundness":  Soundness}
    NewPairData = {"Doc1": file1, "Doc2": file2,"Deciles": Deciles,"SoftDeciles": SoftDeciles,"Support": Support, "Spanning": Spanning, "Soundness": Soundness}
    # UPDATE THIS ROUTE ACCORDING TO YOUR PATH
    myfileSent = os.getcwd()+ '/WORKSPACE/RESULTS/Sent_'+ file1 + "_"+ file2 +'.json'
    myfileSent = myfileSent.replace("\\","/") 
    with open(myfileSent, 'w') as convert_file: 
        convert_file.write(json.dumps(SentPairs))
        

    myfileData = os.getcwd()+ '/WORKSPACE/RESULTS/Data_'+ file1 + "_"+ file2 +'.json'
    myfileData = myfileData.replace("\\","/") 
    with open(myfileData, 'w') as convert_file2: 
        convert_file2.write(json.dumps(NewPairData))
        
      
    myfileClass = os.getcwd()+ '/WORKSPACE/RESULTS/Class_'+ file1 + "_"+ file2 +'.json'
    myfileClass = myfileClass.replace("\\","/") 
    with open(myfileClass, 'w') as convert_file3: 
        convert_file3.write(json.dumps(Class))        

def readResults(filename):
    # UPDATE THIS ROUTE ACCORDING TO YOUR PATH 
     myfileSent = os.getcwd()+ '/WORKSPACE/RESULTS/'+ filename
     myfileSent = myfileSent.replace("\\","/")      
     f = open(myfileSent)
     data = json.load(f)
     return data    

def readResultsJson(filename, folder):
     myfile = os.getcwd()+ '/WORKSPACE/' + folder +'/'+ filename
     myfile = myfile.replace("\\","/")      
     f = open(myfile)
     data = json.load(f)
     return data    
  
def classifyPair(Pairdata,b,c,alpha, beta, gamma, delta):

# put in zeros the deciles below the support 
# apply softmax to the deciles 
    Deciles = Pairdata.get("Deciles")
    #print(Deciles)
    soundness = Pairdata.get("Soundness")
    #print(soundness)
    x = soundness/10;
    accuracy = 0.0; 
    if x < alpha:
       NR = 1
    else:
       NR = 1- (1/ (1+ math.exp(-alpha*10*(soundness*10-alpha*10))))
    CR = math.exp(-((x-beta)**2)/(2*b**2))
    ST = math.exp(-((x-gamma)**2)/(2*c**2))
    if x> delta:
       I = 1
    else:
       I = (1/ (1+ math.exp(-delta*10*(soundness*10-delta*10))))
    rel_degree = 'UNKNOWN'   
    if NR >=CR and NR > ST and NR > I:
       rel_degree = 'NON_RELATED'
       accuracy = NR
    if CR > NR and CR >= ST and CR > I:
       rel_degree = 'CONCEPT_RELATED'
       accuracy = CR
    if  ST > NR and ST > CR and ST >= I:
       rel_degree = 'SAME_TOPIC'
       accuracy = ST
    if  I > NR and I > CR  and I >= ST:
       rel_degree = 'IDENTICAL'
       accuracy = I
    data ={"Relation": rel_degree, "Membership": accuracy}
    #print(rel_degree)  
    return data

def getFiles(listfiles):
    path =  os.getcwd() + '/' + listfiles
    path = path.replace("\\","/") 
    f = open(path, "r", encoding="UTF-8")
    text = f.read()
    words = text.split('\n')
    mylist  = list()
    for x in words:
            mylist.append(x)
    return mylist


def SystematicPairClassification(listfile, model,  nlp, b,c,alpha, beta, gamma, delta):
    Files = getFiles(listfile)
    lsize = len(Files)
    print(lsize)    
    print(Files)
    # Matrix for saving 
    Matrix = [['' for x in range(lsize)] for y in range(lsize)]
    for i in range(0,lsize):
        file1 = Files[i]
        print(file1)
        for k in range(i,lsize):
            file2 =  Files[k]
            print(file2)
            # UPDATE THIS ROUTE ACCORDING TO YOUR PATH
            flag= os.path.isfile(os.getcwd()+ '/WORKSPACE/RESULTS/Class_'+ file1 + "_"+ file2 +'.json')
            if flag == False:
                Pairdata = DATASET_AttentionOnSentences(file1, file2 , 0.0, model, nlp)  
                class_data = classifyPair(Pairdata,b,c,alpha, beta, gamma, delta)
                s = class_data.get("Relation")
                Matrix[i][k] =s[:2]
                SentPairs = selectRepresentativePairs(Pairdata)
                saveResults(Pairdata, SentPairs,class_data )
            else:
                print("The pair is already analyzed")
    
    Dictionary ={"Files": Files, "Matrix": Matrix}
    myfileClass = os.getcwd()+ '/WORKSPACE/RESULTS/GlobalResults'+'.json'
    myfileClass = myfileClass.replace("\\","/") 
    with open(myfileClass, 'w') as convert_file: 
        convert_file.write(json.dumps(Dictionary))  

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
    #emb1 = tokenizer(Text1)["input_ids"]
    #emb2 = tokenizer(Text2)["input_ids"]
    emb1 = torch.tensor(tokenizer.encode(Text1 ))
    #print(emb1)
    emb2 = torch.tensor(tokenizer.encode(Text2 ))
    #print(emb2)    
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
    Matrix = [[0 for x in range(chunks1size+1)] for y in range(chunks2size+1)]
    Deciles = [0 for x in range(11)]
    SoftDeciles = [0 for x in range(11)]
    
    # Text 1 chunks
    for i in range(chunks1size):
        text = ''
        for k in range(chunksize):
            if i*chunksize+k < w1_len:
                text = text + ' '+ words1[i*chunksize+k]
        chunks1.append(text)
        #print(text)
        #print('\n\n')
        
    # Text 2 chunks
    for i in range(chunks2size):
        text = ''
        for k in range(chunksize):
            if i*chunksize+k < w2_len:
                text = text + ' '+ words2[i*chunksize+k]
        chunks2.append(text)
        #print(text)
        #print('\n\n')
        
    for a in range(1, chunks1size):
        for b in range(1, chunks2size):
            res = getLongFormerSimilarity(chunks1[a], chunks2[b], tokenizer)
            if res > threshold:
               if res > Matrix[b][a]:
                  Matrix[b][a] = res
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
    print("\n\n PAIR ENDED\n\n")
    return PairData



###### MAIN SECTION ##########################
#        

print("Torch version:",torch.__version__)

print("Is CUDA enabled?",torch.cuda.is_available())
#BASELINE
#b = 0.2; 
#c = 0.2; 
#alpha = 0.5# 0.68;
#beta = 0.6 #0.78;
#gamma = 0.8 # 0.82;
#delta = 0.9 #0.92;

# TUNED
b = 0.1; 
c = 0.1; 
alpha = 0.75# 0.68;
beta = 0.77 #0.78;
gamma = 0.79 # 0.82;
delta = 0.85 #0.92;

#SVO 
nlp = setNLP()

# CREATION OS SENTENCES VERSIONS
fileSet= 'DatasetListfile.txt'
#preprocessDataset(fileSet, nlp)

chunksize = 64 #16

config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerModel(config)
model = model.cuda() 
#input_ids = torch.tensor(tokenizer.encode("The woman called her friend", return_tensors="pt")).clone().detach()
#print(input_ids)



SystematicPairClassification(fileSet, tokenizer,  nlp, b,c,alpha, beta, gamma, delta)
#DATASET_AttentionOnChunks('2022 Russian invasion of Ukraine Brittanica.txt', '2022 Russian invasion of Ukraine.txt', 0.3, tokenizer, nlp, chunksize)
#DATASET_AttentionOnSentences('2022 Russian invasion of Ukraine Brittanica.txt', '2022 Russian invasion of Ukraine.txt', 0.3, tokenizer, nlp)

print(f" \n\n PROCESS ENDED \n \n ")

#Text1 = "The woman called her friend"
#Text2 = "The girl told her mom to buy some groceries"

#getLongFormerSimilarity(Text1, Text2, tokenizer)

#hello2 = tokenizer(" MyText")["input_ids"]
#print(hello2)

#import torch
#from longformer.longformer import Longformer, LongformerConfig
#from longformer.sliding_chunks import pad_to_window_size
#from transformers import RobertaTokenizer
#from transformers import  LongformerModel, LongformerTokenizer,  LongformerConfig


#print("Torch version:",torch.__version__)

#print("Is CUDA enabled?",torch.cuda.is_available())

#configuration = LongformerConfig()
#model = LongformerModel(configuration)
#tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
#input_ids = tokenizer("Hello world")
#print(input_ids)
#configuration = model.config
#print(configuration)
#print(tokenizer)
#model = AutoModel.from_pretrained("allenai/longformer-base-4096")

#config = LongformerConfig.from_pretrained('allenai/longformer-base-4096') 
#model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
#tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

#tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
##tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
##tokenizer.model_max_length = model.config.max_position_embeddings



#config = LongformerConfig.from_pretrained('longformer-base-4096') 
# choose the attention mode 'n2', 'tvm' or 'sliding_chunks'
# 'n2': for regular n2 attantion
# 'tvm': a custom CUDA kernel implementation of our sliding window attention
# 'sliding_chunks': a PyTorch implementation of our sliding window attention
#config.attention_mode = 'sliding_chunks'
#configuration.attention_mode = 'sliding_chunks'
#print(configuration)
#model = Longformer.from_pretrained('longformer-base-4096/', config=config)
#tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#tokenizer.model_max_length = model.config.max_position_embeddings

#SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document

#input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # #batch of size 1

# TVM code doesn't work on CPU. Uncomment this if `config.attention_mode = 'tvm'`
#model = model.cuda(); input_ids = input_ids.cuda()
#input_ids2 = tokenizer(SAMPLE_TEXT, return_tensors="pt").input_ids

# Attention mask values -- 0: no attention, 1: local attention, 2: global attention
#attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
#attention_mask[:, [1, 4, 21,]] =  2  # Set global attention based on the task. For example,
                                     # classification: the <s> token
                                     # QA: question tokens

# padding seqlen to the nearest multiple of 512. Needed for the 'sliding_chunks' attention
##input_ids, attention_mask = pad_to_window_size(
##        input_ids, attention_mask, config.attention_window[0], tokenizer.pad_token_id)

#output = model(input_ids, attention_mask=attention_mask)[0]
#output = model(input_ids, attention_mask=attention_mask)[0]
#print(output)
