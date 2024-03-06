# This script runs only in python not callable in matlab
import stanza
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import pandas as pd
import json
import math
import matplotlib.pyplot as plt
# pip install -U spacy
# python -m spacy download en_core_web_sm
import spacy


    
def getSimilarities():
    sentences1 = getSentences(1)
    sentences2 = getSentences(1)
    embeddings1 =getEmbeddings(sentences1)
    embeddings2 =getEmbeddings(sentences2)
    lsize = len(embeddings1)
    scores = list()
    for i in range(lsize):
        cosine_scores = util.cos_sim(embeddings1[i], embeddings2[i])
        scores.append(cosine_scores)
    return scores

def getEmbeddings(sentences, themodel):
    ssize = len(sentences)
    print("Sentence size" + str(ssize))
    model = SentenceTransformer(themodel)
    emb_list = list()
    for x in sentences:
        embeddings= model.encode(x, convert_to_tensor=True) 
        emb_list.append(embeddings)
    return emb_list

def CompareSets(interfile, spuriousfile):
    emb_inter = getSingleEmbeddings(interfile)
    emb_spurious = getSingleEmbeddings(spuriousfile)
    count_x = 0 
    count_y = 0
    sim1 = list()
    sim2 = list()
    for x in emb_inter:
        count_x = count_x+1
        count_y = 0
        for y  in emb_spurious:
            count_y = count_y+1
            cosine_score = util.cos_sim(x, y)
            if cosine_score > 0.75:
                sim1.append(count_x)
                sim2.append(count_y)

    # compare here the clusters against themselves
    sim3 = list()
    sim4 = list()
    count_x = 0; 
    for x in emb_inter:
        count_x = count_x+1
        count_y = 0
        for y in emb_inter:
            count_y = count_y+1
            if count_x != count_y:
                cosine_score = util.cos_sim(x, y)
                if cosine_score > 0.75:
                    sim3.append(count_x)
                    sim4.append(count_y) 

    # compare here the spurious against themselves
    sim5 = list()
    sim6 = list()
    count_x = 0; 
    for x in emb_spurious:
        count_x = count_x+1
        count_y = 0
        for y in emb_spurious:
            count_y = count_y+1
            if count_x != count_y:
                cosine_score = util.cos_sim(x, y)
                if cosine_score > 0.75:
                    sim5.append(count_x)
                    sim6.append(count_y) 
    return [sim1, sim2, sim3, sim4, sim5, sim6]          

def getSingleEmbeddings(file, modelname):
    words = getWords(file)
    model = SentenceTransformer(modelname)
    emb_list = list()
    for x in words:
        embeddings= model.encode(x, convert_to_tensor=True) 
        emb_list.append(embeddings)
    return emb_list

def getWords(file):
    # UPDATE THIS ROUTE ACCORDING TO YOUR PATH
    path =  os.getcwd() +'/WORKSPACE/' + file
    path = path.replace("\\","/") 
    f = open(path, "r")
    text = f.read()
    words = text.split(' ')
    mylist  = list()
    for x in words:
            mylist.append(x)
    return mylist

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

def getSentences(Sentencesindex):
    number = int(Sentencesindex)
    # UPDATE THIS ROUTE ACCORDING TO YOUR PATH
    file = '/WORKSPACE/Sentences/Sentences/' + str(number) + '.txt'
    file = os.getcwd() + file
    file = file.replace("\\","/") 
    f = open(file, "r")
    text = f.read()
    sentences = text.split('\n')
    ssize = len(sentences);
    print("Sentences size = " + str(ssize))
    mylist  = list()
    for x in sentences:
            mylist.append(x)
    return mylist

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

def ComparebyEmbeddings(embedding1, embedding2):
    cosine_score = util.cos_sim(embedding1, embedding2)
    return cosine_score.item()

def ComparebyIndices(sent1_index , sent2_index, embeddings1, embeddings2):
    emb1 = int(sent1_index)
    emb2 = int(sent2_index)
    cosine_score = util.cos_sim(embeddings1[emb1], embeddings2[emb2])
    return cosine_score.item()

def DATASET_AttentionOnSentences(file1, file2, threshold, model, nlp):
    filea = file1
    fileb = file2
    file1 ='S_'+ file1;
    file2 ='S_'+ file2;
    Sent1 = getSentencesbyFile(file1)
    Sent2 = getSentencesbyFile(file2)
    emb1 = getEmbeddings(Sent1, model)
    emb2 = getEmbeddings(Sent2, model)
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
            for b in range(1, size2):
                # check if the sentence contains verb if not discard         
                if SVO2[b] == 1: 
                    res = ComparebyIndices(a , b, emb1, emb2)
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
    #print(Deciles)
    PairData = {"file1": filea, "file2": fileb, "Matrix": Matrix, "mSize1": size1, "mSize2": size2, "Deciles": Deciles, "SoftDeciles": SoftDeciles,"Support": supprt, "Spanning": span, "Soundness": soundness}
    
    return PairData

def setPipeline():
    #stanza.download('en')
    #nlp = stanza.Pipeline(lang='en')
    nlp = spacy.load("en_core_web_sm")
    return nlp

def setNLP():
    nlp = spacy.load("en_core_web_sm")
    return nlp

def validSentence(Sentence, nlp):
    
     doc = nlp(Sentence)
     print(doc)
     flag = False
     for i, sentence in enumerate(doc.sentences):
         print(f'====== Sentence {i+1} tokens =======')
         print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')         
     for sent in (doc.sentences):
         for word in (sent.words):   
             print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' ], sep='\n')
             if word.upos == 'VERB':
                 flag = True
    
     return flag

def  validSentenceSpacy(text, nlp):
    # Analyze syntax
   flag = False
   doc = nlp(text)
   for token in doc:
       if token.pos_ == "VERB":
           flag = True  
   return flag

def selectRepresentativePairs(PairData):
    Matrix= PairData.get("Matrix")
    Deciles = PairData.get("Deciles")
    Support = PairData.get("Support")
    Spanning = PairData.get("Spanning")
    Soundness = PairData.get("Soundness")
    size1 = PairData.get("mSize1")
    #print(size1)
    size1 = int(size1)
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
  
def classifyPair(Pairdata,a,b,c,d,alpha, beta, gamma, delta):

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
       NR = math.exp(-((x-alpha)**2)/(2*c**2))
    CR = math.exp(-((x-beta)**2)/(2*b**2))
    ST = math.exp(-((x-gamma)**2)/(2*c**2))
    if x> delta:
       I = 1
    else:
       I = math.exp(-((x-delta)**2)/(2*d**2))
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

def SystematicPairClassification(listfile, model,  nlp, a,b,c,d,alpha, beta, gamma, delta):
    Files = getFiles(listfile)
    lsize = len(Files)
    print(lsize)    
    print(Files)
    # hacer una matriz y guardar los resultados 
    Matrix = [['' for x in range(lsize)] for y in range(lsize)]
    for i in range(0,lsize):
        file1 = Files[i]
        print(file1)
        for k in range(i,lsize):
            file2 = Files[k]
            print(file2)
            # UPDATE THIS ROUTE ACCORDING TO YOUR PATH
            flag= os.path.isfile(os.getcwd()+ '/WORKSPACE/RESULTS/Class_'+ file1 + "_"+ file2 +'.json')
            if flag == False:
                Pairdata = DATASET_AttentionOnSentences(file1, file2 , 0.0, model, nlp)
                class_data = classifyPair(Pairdata,a,b,c,d,alpha, beta, gamma, delta)
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


def RetrieveClass(ClassJson):
    ClassJson = ClassJson.replace("\\","/")      
    f = open(ClassJson)
    data = json.load(f)
    return data   

def GetAnalysis(filelist, folder):
    Files = getFiles(filelist)
    lsize = len(Files)
    print(lsize)    
    print(Files)
    Matrix = [['' for x in range(lsize)] for y in range(lsize)]
    for i in range(0,lsize):
        file1 = Files[i]
        print(file1)
        for k in range(i,lsize):
            file2 = Files[k]
            print(file2)
            # UPDATE THIS ROUTE ACCORDING TO YOUR FOLDER
            classfile = os.getcwd()+ '/WORKSPACE/' + folder +'/Class_'+ file1 + "_"+ file2 +'.json'
            flag= os.path.isfile(classfile)
            if flag == True:
                print(classfile)
                class_data =RetrieveClass(classfile)
                s = class_data.get("Relation")
                # Check the class here
                if s =='NON_RELATED':
                   Matrix[i][k] ='NR'
                if s =='CONCEPT_RELATED': 
                    Matrix[i][k] ='CR'
                if s =='SAME_TOPIC': 
                    Matrix[i][k] ='ST'        
                if s =='IDENTICAL': 
                    Matrix[i][k] ='I'                    
            else:
                Matrix[i][k] ='NotFound'
    
    Dictionary ={"Files": Files, "Matrix": Matrix}
    # UPDATE THIS ROUTE ACCORDING TO YOUR PATH
    myfileClass = os.getcwd()+ '/WORKSPACE/'+ folder+'/GlobalResults.json'
    myfileClass = myfileClass.replace("\\","/") 
    with open(myfileClass, 'w') as convert_file: 
        convert_file.write(json.dumps(Dictionary))  
        
    return Dictionary


def CompareGold_STD(ResultsMatrix, Gold_STD, folder):
    rmrows, rmcols = np.shape(ResultsMatrix)
    print(rmrows)
    print(rmcols) 
    gsrows, gscols = np.shape(Gold_STD)
    print(gsrows)
    print(gscols)
    I = 0
    ST = 0
    CR =0;
    NR = 0
    MATCH = 0;
    OverEstimation =0
    UnderEstimation = 0
    FALSES =0
    TOTAL =0;
    CompMatrix =  [['' for x in range(rmrows)] for y in range(rmcols)]
    for rows in range(0, rmrows):
        for cols in range(rows, rmcols):  
            TOTAL = TOTAL+1
            if ResultsMatrix[rows][cols] == Gold_STD.iloc[rows][cols+1]:
                 CompMatrix[rows][cols]= 1
                 MATCH = MATCH +1
                 if ResultsMatrix[rows][cols]== 'I':
                     I=I+1
                 if ResultsMatrix[rows][cols]== 'ST':
                     ST=ST+1                    
                 if ResultsMatrix[rows][cols]== 'CR':
                     CR=CR+1   
                 if ResultsMatrix[rows][cols]== 'NR':
                     NR=NR+1                     
            else:
                 if ResultsMatrix[rows][cols] == 'I' and Gold_STD.iloc[rows][cols+1] =='ST':
                     CompMatrix[rows][cols]= 0.66
                     OverEstimation =OverEstimation+1
                 if ResultsMatrix[rows][cols] == 'ST' and Gold_STD.iloc[rows][cols+1] =='I':
                     CompMatrix[rows][cols]= 0.66    
                     UnderEstimation = UnderEstimation+1
                 if ResultsMatrix[rows][cols] == 'I' and Gold_STD.iloc[rows][cols+1] =='CR':
                     CompMatrix[rows][cols]= 0.33
                     OverEstimation =OverEstimation+1
                 if ResultsMatrix[rows][cols] == 'CR' and Gold_STD.iloc[rows][cols+1] =='I':  
                     CompMatrix[rows][cols]= 0.33  
                     UnderEstimation = UnderEstimation+1
                     
                 if ResultsMatrix[rows][cols] == 'I' and Gold_STD.iloc[rows][cols+1] =='NR':
                     CompMatrix[rows][cols]= 0.0
                     FALSES = FALSES+1
                     
                 if ResultsMatrix[rows][cols] == 'NR' and Gold_STD.iloc[rows][cols+1] =='I':
                     CompMatrix[rows][cols]= 0.0
                     FALSES = FALSES+1
                     
                 if ResultsMatrix[rows][cols] == 'ST' and Gold_STD.iloc[rows][cols+1] =='CR':
                     CompMatrix[rows][cols]= 0.50
                     OverEstimation =OverEstimation+1
                     
                 if ResultsMatrix[rows][cols] == 'CR' and Gold_STD.iloc[rows][cols+1] =='ST':
                     CompMatrix[rows][cols]= 0.50  
                     UnderEstimation = UnderEstimation+1
                     
                 if ResultsMatrix[rows][cols] == 'ST' and Gold_STD.iloc[rows][cols+1] =='NR':
                     CompMatrix[rows][cols]= 0.25
                     OverEstimation =OverEstimation+1
                     
                 if ResultsMatrix[rows][cols] == 'NR' and Gold_STD.iloc[rows][cols+1] =='ST':
                     CompMatrix[rows][cols]= 0.25 
                     UnderEstimation = UnderEstimation+1
                     
                 if ResultsMatrix[rows][cols] == 'CR' and Gold_STD.iloc[rows][cols+1] =='NR':
                     CompMatrix[rows][cols]= 0.66
                     OverEstimation =OverEstimation+1
                     
                 if ResultsMatrix[rows][cols] == 'NR' and Gold_STD.iloc[rows][cols+1] =='CR':
                     CompMatrix[rows][cols]= 0.66
                     UnderEstimation = UnderEstimation+1
                                           
    plotdata = {'CLASS': ['Total', 'Match', 'I','ST', 'CR','NR', 'UnderEst', 'OverEst', 'Falses'],
                 'Number': [TOTAL, MATCH, I,ST, CR,NR, UnderEstimation, OverEstimation, FALSES]}
    matches= MATCH/TOTAL
    under = UnderEstimation/TOTAL
    over = OverEstimation/ TOTAL
    falses= FALSES/TOTAL
    print(folder)
    print(matches)
    print(under)
    print(over)
    print(falses)
    
    dfplot = pd.DataFrame(plotdata)
    dfplot.plot(x='CLASS', y='Number', kind='bar')
    plt.show()
    Dictionary ={ "CompMatrix": CompMatrix}
    # UPDATE THIS ROUTE ACCORDING TO YOUR PATH
    myfileClass = os.getcwd()+ '/WORKSPACE/'+ folder+'/GS_Analysis.json'
    myfileClass = myfileClass.replace("\\","/") 
    with open(myfileClass, 'w') as convert_file: 
        convert_file.write(json.dumps(Dictionary))  
    df= pd.DataFrame(CompMatrix)
    df.to_csv(os.getcwd()+ '/WORKSPACE/'+ folder+'/GS_Analysis.csv')
    
    print(CompMatrix)            
    return CompMatrix        
                
    
    
# Model can be one of the followings
# 'all-MiniLM-L6-v2'
# 'all-MiniLM-L12-v2'
# 'all-mpnet-base-v2'
# 'sentence-transformers/average_word_embeddings_glove.6B.300d'

## CHOOSE THE MODEL
themodel = 'all-MiniLM-L6-v2'
#themodel = 'all-MiniLM-L12-v2'
#themodel = 'all-mpnet-base-v2'
##themodel = 'sentence-transformers/average_word_embeddings_glove.6B.300d'

# PARAMETERS FOR CLASSIFICATION
nlp = setNLP()
a = 0.2;
b = 0.2; 
c = 0.2;
d = 0.2; 
alpha = 0.5;
beta = 0.6;
gamma = 0.8;
delta = 0.9;


## APPLY A SYSTEMATIC  CLASSIFICATION -- CREATE THE FOLDER FIRST!!
folder = 'RESULTS_all-MiniLM-L6-v2'
#folder = 'RESULTS_all-MiniLM-L12-v2'
#folder = 'RESULTS_all-mpnet-base-v2'
#folder = 'RESULTS_glove.300D'
SystematicPairClassification('DatasetListfile.txt', themodel,  nlp, a,b,c,d,alpha, beta, gamma, delta)
MyDict = GetAnalysis('DatasetListfile.txt', folder)


results = readResultsJson('GlobalResults.json', folder )
ResultsMatrix = results.get("Matrix")
print(ResultsMatrix[0][0])
goldstd = pd.read_csv('C:\RESEARCH PROJECTS\MIXED_ARCHITECTURE\Dataset 72 Docs Gold-Standard.csv', header=0)
print(goldstd.iloc[0][1])
CompMatrix = CompareGold_STD(ResultsMatrix, goldstd, folder )


# EXAMPLE OF USING A SINGLE COMPARISON
#file1 = '2022 Russian invasion of Ukraine.txt'
#file2 = '2022 Russian invasion of Ukraine.txt'
#Pairdata = DATASET_AttentionOnSentences(file1, file2 , 0.0, themodel, nlp)
#SentPairs = selectRepresentativePairs(Pairdata)
#saveResults(Pairdata, SentPairs)
#data = readResults('Sent_'+ file1 + "_"+ file2 +'.json')
#print(data)
