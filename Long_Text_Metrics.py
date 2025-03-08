from telnetlib import GA
import stanza
from sentence_transformers import SentenceTransformer, util
from operator import add
import numpy as np
import os
import pandas as pd
import json
import math
import matplotlib.pyplot as plt
from matplotlib import colors, legend
# pip install -U spacy
#python -m spacy download en_core_web_sm
import spacy
import re
import random
from sklearn import metrics
import seaborn as sn
import time

def setNLP():
    nlp = spacy.load("en_core_web_sm")
    return nlp


def setModel(themodel):
    model = SentenceTransformer(themodel)
    return model

def readResultsJson(filename, folder):
     myfile = os.getcwd()+ '/WORKSPACE/' + folder + '/' + filename
     myfile = myfile.replace("\\","/")    
     try:
         f = open(myfile)
         data = json.load(f)
     except: 
         print("File not found")
         data = -1
     return data  

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
    print(I)
    print(ST)
    print(CR)
    print(NR)
    
    
    Dictionary ={ "CompMatrix": CompMatrix}
    # UPDATE THIS ROUTE ACCORDING TO YOUR PATH
    myfileClass = os.getcwd()+ '/WORKSPACE/'+ folder+'/GS_Analysis.json'
    myfileClass = myfileClass.replace("\\","/") 
    with open(myfileClass, 'w') as convert_file: 
        convert_file.write(json.dumps(Dictionary))  
    df= pd.DataFrame(CompMatrix)
    df.to_csv(os.getcwd()+ '/WORKSPACE/'+ folder+'/GS_Analysis.csv')
    df2 = pd.DataFrame(ResultsMatrix)
    df2.to_csv(os.getcwd()+ '/WORKSPACE/'+ folder+'/Results.csv')

    
    print(CompMatrix) 
    dfplot = pd.DataFrame(plotdata)
    dfplot.plot(x='CLASS', y='Number', kind='bar')
    plt.show()


    #return CompMatrix

def GetMatrixfromGS( Gold_STD):
    gsrows, gscols = np.shape(Gold_STD)
    CompMatrix =  [["" for x in range(gsrows)] for y in range(gscols-1)]
    for rows in range(0, gsrows):
       for cols in range(gsrows, gscols-1):
           print(Gold_STD.iloc[rows,cols+1])
           CompMatrix[rows][cols] = Gold_STD[rows][cols+1]

    print(CompMatrix)
    return CompMatrix

def saveResults(PairData, SentPairs, PairClass, Time):
    file1 = PairData.get("file1");
    file2 = PairData.get("file2");
    Deciles = PairData.get("Deciles")
    SoftDeciles = PairData.get("SoftDeciles")
    Support = PairData.get("Support")
    Spanning = PairData.get("Spanning")
    Soundness = PairData.get("Soundness")
    Relation = PairClass.get("Relation")
    Membership = PairClass.get("Membership")
    Matrix =  PairData.get("Matrix")

    Class ={"Doc1": file1, "Doc2": file2, "Relation": Relation," Membership":  Membership, "Soundness":  Soundness, "Time": Time}
    NewPairData = {"Doc1": file1, "Doc2": file2,"Deciles": Deciles,"SoftDeciles": SoftDeciles,"Support": Support, "Spanning": Spanning, "Soundness": Soundness, "Time": Time}
    MatrixData ={"Matrix": Matrix}
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

    myfileMatrix = os.getcwd()+ '/WORKSPACE/RESULTS/Matrix_'+ file1 + "_"+ file2 +'.json'
    myfileMatrix = myfileMatrix.replace("\\","/") 
    with open(myfileMatrix, 'w') as convert_file4: 
        convert_file4.write(json.dumps(MatrixData)) 

        
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


def RetrieveClass(ClassJson):
    ClassJson = ClassJson.replace("\\","/")      
    f = open(ClassJson)
    data = json.load(f)
    return data 

def SystematicPairReClassification(listfile,folder, b, c, alpha, beta, gamma, delta):
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
            myfile = os.getcwd()+ '/WORKSPACE/' + folder + '/Data_' + file1 +'_'+ file2 +'.json'
            Pairdata =RetrieveClass(myfile)
            class_data = classifyPair(Pairdata,b, c, alpha, beta, gamma, delta)
            s = class_data.get("Relation")
            Matrix[i][k] =s[:2]
            SentPairs = selectRepresentativePairs(Pairdata)
            saveResults(Pairdata, SentPairs,class_data )
            
    
    Dictionary ={"Files": Files, "Matrix": Matrix, "b": b, "c": c, "alpha": alpha,"beta": beta, "gamma": gamma, "delta": delta}
    myfileClass = os.getcwd()+ '/WORKSPACE/RESULTS/GlobalResults'+'.json'
    myfileClass = myfileClass.replace("\\","/") 
    with open(myfileClass, 'w') as convert_file: 
        convert_file.write(json.dumps(Dictionary)) 


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
       NR = math.exp(-((x-alpha)**2)/(2*c**2))
    CR = math.exp(-((x-beta)**2)/(2*b**2))
    ST = math.exp(-((x-gamma)**2)/(2*c**2))
    if x> delta:
       I = 1
    else:
       I = 1 /(1 + math.exp(-(delta)*10*(x*10-delta*10 )))
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

def getEmbeddingsSent_Transformer(sentences, model):
    ssize = len(sentences)
    print("Sentence size" + str(ssize))
    emb_list = list()
    for x in sentences:
        embeddings= model.encode(x, convert_to_tensor=True) 
        emb_list.append(embeddings)
    return emb_list

def ComparebyIndices(sent1_index , sent2_index, embeddings1, embeddings2):
    emb1 = int(sent1_index)
    emb2 = int(sent2_index)
    cosine_score = util.cos_sim(embeddings1[emb1], embeddings2[emb2])
    return cosine_score.item()

def  validSentenceSpacy(text, nlp):
    # Analyze syntax
   flag = False
   doc = nlp(text)
   for token in doc:
       if token.pos_ == "VERB":
           flag = True  
   return flag

def DATASET_AttentionOnSentences(file1, file2, threshold, model, nlp):
    filea = file1
    fileb = file2
    file1 ='S_'+ file1;
    file2 ='S_'+ file2;
    Sent1 = getSentencesbyFile(file1)
    Sent2 = getSentencesbyFile(file2)
    emb1 = getEmbeddingsSent_Transformer(Sent1, model)
    emb2 = getEmbeddingsSent_Transformer(Sent2, model)
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


def selectRepresentativePairs(PairData):
   # print(PairData)
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


def SystematicPairClassification(listfile, model,  nlp,b,c,alpha, beta, gamma, delta):
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
                start = time.time()
                Pairdata = DATASET_AttentionOnSentences(file1, file2 , 0.0, model, nlp)
                class_data = classifyPair(Pairdata,b,c,alpha, beta, gamma, delta)
                s = class_data.get("Relation")
                Matrix[i][k] =s[:2]
                SentPairs = selectRepresentativePairs(Pairdata)
                end = time.time()
                Telapsed = end - start
                saveResults(Pairdata, SentPairs,class_data, Telapsed )
            else:
                print("The pair is already analyzed")
    
    Dictionary ={"Files": Files, "Matrix": Matrix, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma":gamma, "delta": delta }
    myfileClass = os.getcwd()+ '/WORKSPACE/RESULTS/GlobalResults'+'.json'
    myfileClass = myfileClass.replace("\\","/") 
    with open(myfileClass, 'w') as convert_file: 
        convert_file.write(json.dumps(Dictionary)) 

def GetConfusionMatrix(ResultsMatrix, Gold_STD, folder):
    rmrows, rmcols = np.shape(ResultsMatrix)
    print(rmrows)
    print(rmcols) 
    gsrows, gscols = np.shape(Gold_STD)
    print(gsrows)
    print(gscols)
    I_I = 0
    I_ST = 0
    I_CR = 0
    I_NR = 0

    ST_ST = 0
    ST_I = 0
    ST_CR = 0
    ST_NR = 0

    CR_CR =0
    CR_I = 0
    CR_ST = 0
    CR_NR = 0

    NR_NR = 0
    NR_I = 0
    NR_ST = 0
    NR_CR = 0

    MATCH = 0;

    FALSES =0
    TOTAL =0;
    #CompMatrix =  [['' for x in range(rmrows)] for y in range(rmcols)]
    ConfusionMatrix =  [[0 for x in range(4)] for y in range(4)]
    for rows in range(0, rmrows):
        for cols in range(rows, rmcols):  
            TOTAL = TOTAL+1
            if ResultsMatrix[rows][cols] == Gold_STD.iloc[rows, cols+1]:
                 
                 MATCH = MATCH +1
                 if ResultsMatrix[rows][cols]== 'I':
                     I_I=I_I+1
                 if ResultsMatrix[rows][cols]== 'ST':
                     ST_ST=ST_ST+1                    
                 if ResultsMatrix[rows][cols]== 'CR':
                     CR_CR=CR_CR+1   
                 if ResultsMatrix[rows][cols]== 'NR':
                     NR_NR=NR_NR+1                     
            else:
                 if ResultsMatrix[rows][cols] == 'I' and Gold_STD.iloc[rows,cols+1] =='ST':
                     I_ST=I_ST+1
                 if ResultsMatrix[rows][cols] == 'ST' and Gold_STD.iloc[rows, cols+1] =='I':
                     ST_I=ST_I+1
                 if ResultsMatrix[rows][cols] == 'I' and Gold_STD.iloc[rows, cols+1] =='CR':
                     I_CR=I_CR+1
                 if ResultsMatrix[rows][cols] == 'CR' and Gold_STD.iloc[rows, cols+1] =='I':  
                     CR_I=CR_I+1   
                 if ResultsMatrix[rows][cols] == 'I' and Gold_STD.iloc[rows, cols+1] =='NR':
                     I_NR=I_NR+1
                 if ResultsMatrix[rows][cols] == 'NR' and Gold_STD.iloc[rows, cols+1] =='I':
                     NR_I=NR_I+1 
                     
                 if ResultsMatrix[rows][cols] == 'ST' and Gold_STD.iloc[rows, cols+1] =='CR':
                     ST_CR=ST_CR+1
                     
                 if ResultsMatrix[rows][cols] == 'CR' and Gold_STD.iloc[rows, cols+1] =='ST':
                     CR_ST=CR_ST
                     
                 if ResultsMatrix[rows][cols] == 'ST' and Gold_STD.iloc[rows, cols+1] =='NR':
                     ST_NR=ST_NR+1
                     
                 if ResultsMatrix[rows][cols] == 'NR' and Gold_STD.iloc[rows, cols+1] =='ST':
                     NR_ST=NR_ST+1 
                     
                 if ResultsMatrix[rows][cols] == 'CR' and Gold_STD.iloc[rows, cols+1] =='NR':
                    CR_NR=CR_NR+1
                     
                 if ResultsMatrix[rows][cols] == 'NR' and Gold_STD.iloc[rows, cols+1] =='CR':
                    NR_CR=NR_CR+1 
                                           
    plotdata = {'CLASS': ['Total', 'Match', 'I_I', 'I_ST', 'I_CR', 'I_NR', 'ST_ST', 'ST_I', 'ST_CR', 'ST_NR', 'CR_CR', 'CR_I', 'CR_ST', 'CR_NR', 'NR_NR', 'NR_I', 'NR_ST' , 'NR_CR'],
                 'Number': [TOTAL, MATCH, I_I, I_ST, I_CR, I_NR, ST_ST, ST_I, ST_CR, ST_NR, CR_CR, CR_I, CR_ST, CR_NR, NR_NR, NR_I, NR_ST , NR_CR]}
    matches= MATCH/TOTAL
    
    
    ConfusionMatrix[0][0] = I_I
    ConfusionMatrix[0][1] = I_ST
    ConfusionMatrix[0][2] = I_CR
    ConfusionMatrix[0][3] = I_NR

    ConfusionMatrix[1][0] = ST_I
    ConfusionMatrix[1][1] = ST_ST
    ConfusionMatrix[1][2] = ST_CR
    ConfusionMatrix[1][3] = ST_NR

    ConfusionMatrix[2][0] = CR_I
    ConfusionMatrix[2][1] = CR_ST
    ConfusionMatrix[2][2] = CR_CR
    ConfusionMatrix[3][3] = CR_NR

    ConfusionMatrix[3][0] = NR_I
    ConfusionMatrix[3][1] = NR_ST
    ConfusionMatrix[3][2] = NR_CR
    ConfusionMatrix[3][3] = NR_NR

    print(ConfusionMatrix) 
    Dictionary ={ "ConfusionMatrix": ConfusionMatrix }
    # UPDATE THIS ROUTE ACCORDING TO YOUR PATH
    myfileClass = os.getcwd()+ '/WORKSPACE/'+ folder+'/GS_Analysis_Metrics.json'
    myfileClass = myfileClass.replace("\\","/") 
    with open(myfileClass, 'w') as convert_file: 
        convert_file.write(json.dumps(Dictionary))  
    df= pd.DataFrame(ConfusionMatrix)
    df.to_csv(os.getcwd()+ '/WORKSPACE/'+ folder+'/GS_Analysis_Metrics.csv')
    df2 = pd.DataFrame(ResultsMatrix)
    df2.to_csv(os.getcwd()+ '/WORKSPACE/'+ folder+'/Results_Metrics.csv')
    #dfplot = pd.DataFrame(plotdata)
    #dfplot.plot(x='CLASS', y='Number', kind='bar')
    #plt.show()

    #gsmatrx = GetMatrixfromGS(Gold_STD)
    #colormap = colors.ListedColormap(["white","green", "blue"])
    df_cm = pd.DataFrame(ConfusionMatrix, index = ['identical', 'same topic', 'concept related', 'nonrelated'], columns = ['identical', 'same topic', 'concept related', 'nonrelated'])
    sn.set(font_scale=1) # for label size

    plt.figure(figsize=(4,4))
    ax = sn.heatmap(df_cm, cmap="crest", annot=True, linewidth=.5)
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    
    plt.show() 
    
    return ConfusionMatrix


nlp = setNLP()
themodel = 'all-MiniLM-L6-v2'
model = setModel(themodel)

#BASELINE
b = 0.2; 
c = 0.2; 
alpha = 0.5# 0.68;
beta = 0.6 #0.78;
gamma = 0.8 # 0.82;
delta = 0.9 #0.92;

# TUNED 1
#b = 0.1
#c = 0.1
#alpha = 0.75# 0.68;
#beta = 0.77 #0.78;
#gamma = 0.79 # 0.82;
#delta = 0.85 #0.92;
folder = 'RESULTS_all-MiniLM-L6-v2'
listfile = 'DatasetListfile.txt'
filename = "Dataset 72 Docs Gold-Standard.xlsx"
path = os.getcwd()
SystematicPairClassification(listfile, model, nlp, b, c, alpha, beta, gamma, delta)
#SystematicPairReClassification(listfile,folder, b, c, alpha, beta, gamma, delta)
results = readResultsJson('GlobalResults.json', folder )
ResultsMatrix = results.get("Matrix")
goldstdmatrix = pd.read_csv('C:\RESEARCH PROJECTS\MIXED_ARCHITECTURE\Dataset 72 Docs Gold-Standard.csv', header=0)
print(results)
print(goldstdmatrix.iloc[0,1])
#CompMatrix = CompareGold_STD(ResultsMatrix, goldstdmatrix, folder )
ConfusionMatrix = GetConfusionMatrix(ResultsMatrix, goldstdmatrix, folder)
print(ConfusionMatrix)
TP = ConfusionMatrix[0][0] + ConfusionMatrix[1][1] +ConfusionMatrix[2][2] +ConfusionMatrix[3][3]
