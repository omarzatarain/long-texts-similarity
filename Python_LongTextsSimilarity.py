# This script runs only in python not callable in matlab
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
# pip install -U spacy
# python -m spacy download en_core_web_sm
import spacy
import re
import random

# Randomize Texts
def RandomizeText(filename, path, nlp):
    mytext = getFile(filename, path, nlp)
    print("TEXT BEFORE SHUFFLE \n \n ")
    print(mytext)
    random.shuffle(mytext)
    print("\n\n TEXT AFTER SHUFFLE \n \n ")
    print(mytext)
    return mytext
    
def importExcelFile(filename, path, sheet):
    file = path +"/" + filename
    file = file.replace("\\","/")	
    print(file)
    
    df = pd.read_excel(file, sheet_name=sheet)
    dim_x,dim_y = df.shape
   # Matrix for saving 
    Matrix = [['' for x in range(dim_x+1)] for y in range(dim_y+1)]
    for i in range(dim_x):
        for j in range(dim_y):
            Matrix[i][j] = df.iat[i,j]
    print(dim_x)
    print(dim_y)
    print(Matrix)
    Dictionary ={"Matrix": Matrix}
    myfileClass =path+ '/Gold_Standard'+'.json'
    myfileClass = myfileClass.replace("\\","/") 
    with open(myfileClass, 'w') as convert_file: 
        convert_file.write(json.dumps(Dictionary)) 
    return Matrix

## FOR CREATING THE PIPELINE FOR THE KNOWLEDGE BASE ITS NECESSARY TO GET THE 

def getDistribution(Values, reference):
    centrallimit = 0;
    counter =1
    centiles = [0 for x in range(11)]
    vsize = len(Values)
    while Values[counter]> 0 and counter <vsize:
        centil = Values[counter] -reference
        print(Values[counter])
        print(reference)
        print(centil)
        if centil >0 and centil < 0.01:
            centiles[1]= centiles[1]+1
        if centil >0.01 and centil < 0.02:
            centiles[2]= centiles[2]+1      
        if centil >0.02 and centil < 0.03:
            centiles[3]= centiles[3]+1
        if centil >0.03 and centil < 0.04:
            centiles[4]= centiles[4]+1 
        if centil >0.04 and centil < 0.05:
            centiles[5]= centiles[5]+1
        if centil >0.05 and centil < 0.06:
            centiles[6]= centiles[6]+1 
        if centil >0.06 and centil < 0.07:
            centiles[7]= centiles[7]+1
        if centil >0.07 and centil < 0.08:
            centiles[8]= centiles[8]+1 
        if centil >0.08 and centil < 0.09:
            centiles[9]= centiles[9]+1
        if centil >0.09 and centil <=1:
            centiles[10]= centiles[10]+1             
            ## PROCESS THE REST OF CENTILES
        counter = counter +1
    ## GET THE CENTRAL CENTILE
    print("CENTILES")  
    print(centiles)
    return centiles
            
def AssesmentTuning(listfile, folder, baseline, GSMatrix, ResultsMatrix):
    # retrieve files set
    Files = getFiles(listfile)
    lsize = len(Files)
    print(lsize)    
    print(Files)
    # get starting baseline parameters and create new assessing parameters
    a = baseline.get("a")
    b = baseline.get("b")
    c = baseline.get("c")
    d = baseline.get("d")
    alpha = baseline.get("alpha")
    beta = baseline.get("beta")
    gamma = baseline.get("gamma")
    delta = baseline.get("delta")    
    
   # create the indices of files for  tunning parameters with pairs of documents
   # USE CODE FROM GOLD STD COMPARISON TO REDO THE ANALYSIS OF DOCUMENT PAIRS WHEN UNDERESTIMATION OR OVERESTIMATION

    rmrows, rmcols = np.shape(ResultsMatrix)
    print(rmrows)
    print(rmcols) 
    gsrows, gscols = np.shape(GSMatrix)
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
    counter =0;
    Icentiles = [0 for x in range(11)]
    STcentiles = [0 for x in range(11)]
    CRcentiles = [0 for x in range(11)]
    NRcentiles = [0 for x in range(11)]
    
    LR_Icentiles = [0 for x in range(11)]
    LR_STcentiles = [0 for x in range(11)]
    LR_CRcentiles = [0 for x in range(11)]  
    LR_NRcentiles = [0 for x in range(11)]
    for rows in range(0, rmrows):
        file1 =Files[rows]
        
        for cols in range(rows, rmcols):  
            file2 =Files[cols]
            datafile ='/Sent_'+ file1 + "_"+ file2 +'.json'
            # Get deciles according for size sample estimation
            pair_data =readResultsJson(datafile, folder)
            values = pair_data.get("Value")
            print(values)
            TOTAL = TOTAL+1
            # SETTING PARAMETERS alpha, beta gamma and delta through ttahe central limit theorem
            if ResultsMatrix[rows][cols] == GSMatrix.iloc[rows][cols+1]:
                 CompMatrix[rows][cols]= 1
                 MATCH = MATCH +1
                 if ResultsMatrix[rows][cols]== 'I':
                     I=I+1
                     # get the minimum delta (central limit)
                     decile = delta
                     distribution = getDistribution(values, decile)
                     Icentiles =list(map(add, Icentiles, distribution))
                     print("I CASE")
                     print(Icentiles)
                     
                 if ResultsMatrix[rows][cols]== 'ST':
                     ST=ST+1   
                     # get the central limit gamma
                     decile = gamma 
                     distribution = getDistribution(values, decile)
                     STcentiles =list(map(add, STcentiles, distribution))
                     print("ST CASE")
                    # print(STcentiles)
                     
                 if ResultsMatrix[rows][cols]== 'CR':
                     CR=CR+1   
                      # get the central limit beta 
                     decile = beta
                     distribution = getDistribution(values, decile)
                     CRcentiles =list(map(add, CRcentiles, distribution))
                     print("CR CASE")
                     #print(CRcentiles)
                     

                 if ResultsMatrix[rows][cols]== 'NR':
                     NR=NR+1 
                     # get the maximum limit alpha
                     decile = alpha
                     distribution = getDistribution(values, decile)
                     NRcentiles =list(map(add, NRcentiles, distribution))
                     print("NR CASE")
                    # print( NRcentiles)
                     
            # LEARNING

            else:
                 counter = counter+1 
                 print("LEARNING") 
                 if ResultsMatrix[rows][cols] == 'I' and GSMatrix.iloc[rows][cols+1] =='ST':
                     CompMatrix[rows][cols]= 0.66
                     OverEstimation =OverEstimation+1
                     #increase delta
                     decile = delta
                     distribution = getDistribution(values, decile)
                     LR_Icentiles =list(map(add, LR_Icentiles, distribution))
                     print("OVERESTIMATION ST_I")
                     
                 if ResultsMatrix[rows][cols] == 'ST' and GSMatrix.iloc[rows][cols+1] =='I':
                     CompMatrix[rows][cols]= 0.66  
                     print("UNDERESTIMATION I_ST")
                     UnderEstimation = UnderEstimation+1
                     # increase gamma
                     decile = gamma
                     distribution = getDistribution(values, decile)
                     LR_STcentiles =list(map(add, LR_STcentiles, distribution))
                     
                 if ResultsMatrix[rows][cols] == 'I' and GSMatrix.iloc[rows][cols+1] =='CR':
                     CompMatrix[rows][cols]= 0.33
                     OverEstimation =OverEstimation+1
                     print("OVERESTIMATION CR_I")
                      ## to far, model bias

                 if ResultsMatrix[rows][cols] == 'CR' and GSMatrix.iloc[rows][cols+1] =='I':  
                     CompMatrix[rows][cols]= 0.33  
                     UnderEstimation = UnderEstimation+1
                     print("UNDERESTIMATION I_CR")
                      ## to far, model bias

                 if ResultsMatrix[rows][cols] == 'I' and GSMatrix.iloc[rows][cols+1] =='NR':
                     CompMatrix[rows][cols]= 0.0
                     FALSES = FALSES+1
                     print("FALSE I")
                     ## too far, model bias

                 if ResultsMatrix[rows][cols] == 'NR' and GSMatrix.iloc[rows][cols+1] =='I':
                     CompMatrix[rows][cols]= 0.0
                     FALSES = FALSES+1
                     print("FALSE NR")
                     ## to far, model bias

                 if ResultsMatrix[rows][cols] == 'ST' and GSMatrix.iloc[rows][cols+1] =='CR':
                     CompMatrix[rows][cols]= 0.50
                     OverEstimation =OverEstimation+1
                     print("OVERESTIMATION ST_CR")
                     # increase beta
                     decile = beta
                     distribution = getDistribution(values, decile)
                     LR_CRcentiles =list(map(add, LR_CRcentiles, distribution))

                 if ResultsMatrix[rows][cols] == 'CR' and GSMatrix.iloc[rows][cols+1] =='ST':
                     CompMatrix[rows][cols]= 0.50  
                     UnderEstimation = UnderEstimation+1
                     print("UNDERESTIMATION CR_ST")
                     # decrease gamma
                     decile = gamma
                     distribution = getDistribution(values, decile)
                     LR_STcentiles =list(map(add, LR_STcentiles, distribution))

                 if ResultsMatrix[rows][cols] == 'ST' and GSMatrix.iloc[rows][cols+1] =='NR':
                     CompMatrix[rows][cols]= 0.25
                     OverEstimation =OverEstimation+1
                     print("OVERESTIMATION ST_NR")
                     ## too far, model bias

                 if ResultsMatrix[rows][cols] == 'NR' and GSMatrix.iloc[rows][cols+1] =='ST':
                     CompMatrix[rows][cols]= 0.25 
                     UnderEstimation = UnderEstimation+1
                     print("UNDERESTIMATION NR_ST")
                     ## to far 

                 if ResultsMatrix[rows][cols] == 'CR' and GSMatrix.iloc[rows][cols+1] =='NR':
                     CompMatrix[rows][cols]= 0.66
                     OverEstimation =OverEstimation+1
                     print("OVERESTIMATION CR_NR")
                     # increase alpha
                     decile = alpha
                     distribution = getDistribution(values, decile)
                     LR_NRcentiles =list(map(add, LR_NRcentiles, distribution))
                     
                 if ResultsMatrix[rows][cols] == 'NR' and GSMatrix.iloc[rows][cols+1] =='CR':
                     CompMatrix[rows][cols]= 0.66
                     UnderEstimation = UnderEstimation+1
                     print("UNDERESTIMATION NR_CR") 
                     # decrease beta
                     decile = beta
                     distribution = getDistribution(values, decile)
                     LR_CRcentiles =list(map(add, LR_CRcentiles, distribution))
      
    print("counter: ", counter)
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
    print("I CASE")
    print(Icentiles)
    print("ST CASE")
    print(STcentiles)   
    print("CR CASE")
    print(CRcentiles)
    print("NR CASE")
    print(NRcentiles)
    
    print("LEARNING: I CASE")
    print(LR_Icentiles)
    print("LEARNING: ST CASE")
    print(LR_STcentiles)   
    print("LEARNING: CR CASE")
    print(LR_CRcentiles)
    print("LEARNING: NR CASE")
    print(LR_NRcentiles)
    
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

    
    #print(CompMatrix) 
    dfplot = pd.DataFrame(plotdata)
    dfplot.plot(x='CLASS', y='Number', kind='bar')
    plt.show()
    #return CompMatrix     
    #
    #
    ############################################################################   

    # get goldstd

    #for each pair check gldstd vs assesment
    # if over assesment then increase lower class parameters
    #if  sub assessment then decrease upper class parameters
    # create a new baseline using the  tuning
    # Test new estimation against goldstd
    # if new estimation outperforms baseline then return new estimation


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

def AutoTuning(listfile, folder, baseline,  GSMatrix, ResultsMatrix):
    # retrieve files set
    Files = getFiles(listfile)
    lsize = len(Files)
    print(lsize)    
    print(Files)
    # get starting baseline parameters and create new assessing parameters
    a = baseline.get("a")
    b = baseline.get("b")
    c = baseline.get("c")
    d = baseline.get("d")
    alpha = baseline.get("alpha")
    beta = baseline.get("beta")
    gamma = baseline.get("gamma")
    delta = baseline.get("delta")    
    
   # create the indices of files for  tunning parameters with pairs of documents
   # USE CODE FROM GOLD STD COMPARISON TO REDO THE ANALYSIS OF DOCUMENT PAIRS WHEN UNDERESTIMATION OR OVERESTIMATION

    rmrows, rmcols = np.shape(ResultsMatrix)
    print(rmrows)
    print(rmcols) 
    gsrows, gscols = np.shape(GSMatrix)
    print(gsrows)
    print(gscols)
    I = 0
    ST = 0
    CR =0;
    NR = 0
    MATCH = 0;
    OverEstimation_33 =0
    OverEstimation_66 =0
    UnderEstimation_33 = 0
    UnderEstimation_66 = 0
    FALSES =0
   
    TOTAL =0;
    CompMatrix =  [['' for x in range(rmrows)] for y in range(rmcols)]
    counter =0;
    grad_learn = 0.001
    for i in range(5):
        for rows in range(0, rmrows):
            file1 =Files[rows] 
            for cols in range(rows, rmcols):  
                file2 =Files[cols]
                #datafile ='/Sent_'+ file1 + "_"+ file2 +'.json'
                # Get deciles according for size sample estimation
                #pair_data =readResultsJson(datafile, folder)
                #values = pair_data.get("Value")
                #print(values)
                datafile ='/Data_'+ file1 + "_"+ file2 +'.json'
                pair_data =readResultsJson(datafile, folder)
                soundness =pair_data.get("Soundness")/10
                support =pair_data.get("Support")/10
                label = defuzzyfication(soundness, alpha, b, beta, c, gamma, delta)
                # SETTING PARAMETERS alpha, beta gamma and delta through ttahe central limit theorem
                if  GSMatrix.iloc[rows][cols+1] != " ":
                   # LEARNING
                     counter = counter+1 
                     print("LEARNING")  
                     if GSMatrix.iloc[rows][cols+1] =='I' and label !="IDENTICAL":
                         if soundness > delta:
                             delta = delta + grad_learn
                             if delta > 0.98:
                                 delta = 0.98
                             # Backwards  
                             if gamma > delta:
                                 gamma = delta -grad_learn
                             if beta > gamma:
                                 beta = gamma -grad_learn
                             if alpha > beta:
                                 alpha = beta -grad_learn
                     
                                 
                     if  GSMatrix.iloc[rows][cols+1] =='ST' and label !="SAME-TOPIC":
                         if soundness > gamma:
                             gamma = gamma + grad_learn
                             #Forward
                             if gamma > delta:
                                 delta = gamma + 2*grad_learn
                             #Backwards
                             if beta > gamma:
                                 beta = gamma -grad_learn
                             if alpha > beta:
                                 alpha = beta -grad_learn  
                     
                     if GSMatrix.iloc[rows][cols+1] =='CR' and label != "CONCEPT-RELATED":
                         if soundness > beta:
                             beta = beta + grad_learn
                             # Forward
                             if delta < 0.98:
                                 if beta > gamma:
                                    gamma =beta + 2*grad_learn
                                 if gamma > delta:
                                    delta = gamma + 2*grad_learn   
                             #Backwards
                             if alpha > beta:
                                 alpha = beta -grad_learn
                         else: 
                             if beta > alpha+ 3*grad_learn:
                                beta = beta -grad_learn

                     if GSMatrix.iloc[rows][cols+1] =='NR' and label != "NON-RELATED":
                         if soundness > alpha:
                             alpha = alpha + grad_learn
                         else:
                             if alpha > grad_learn:
                                 alpha = alpha - grad_learn
                
    
    PREMATCH = MATCH
    FALSES = 0
    MATCH = 0
    TOTAL = 0
    ## TEST AGAIN 

   # a = baseline.get("a")
   # b = baseline.get("b")
   # c = baseline.get("c")
   # d = baseline.get("d")
    alpha = baseline.get("alpha")
    beta = baseline.get("beta")
    gamma = baseline.get("gamma")
    delta = baseline.get("delta") 

    for rows in range(0, rmrows):
        file1 =Files[rows] 
        for cols in range(rows, rmcols):  
            file2 =Files[cols]
            datafile ='/Data_'+ file1 + "_"+ file2 +'.json'
            pair_data =readResultsJson(datafile, folder)
            soundness =(pair_data.get("Soundness"))/10
            support =pair_data.get("Support")/10
            TOTAL = TOTAL+1
            label = defuzzyfication(soundness, alpha, b, beta, c, gamma, delta)
               
            # TESTING

             #PÜT THE MATCHES AND TEST CASES HERE
            counter = counter+1 
            print("TESTING") 
            if GSMatrix.iloc[rows][cols+1] =='I':
                if label == "IDENTICAL":
                    MATCH = MATCH +1
                    I = I+1
                if label == "SAME-TOPIC":  
                    UnderEstimation_33 = UnderEstimation_33 +1
                if label == "CONCEPT-RELATED":  
                    UnderEstimation_66 = UnderEstimation_66 +1                       
                if label == "NON-RELATED":  
                    FALSES = FALSES +1                         
                     
            if GSMatrix.iloc[rows][cols+1] =='ST':
                if label == "IDENTICAL":
                    OverEstimation_33 = OverEstimation_33 +1
                if label == "SAME-TOPIC": 
                    MATCH = MATCH +1
                    ST = ST+1
                if label == "CONCEPT-RELATED":  
                    UnderEstimation_33 = UnderEstimation_33 +1                       
                if label == "NON-RELATED":  
                    UnderEstimation_66 = UnderEstimation_66 +1                          
                     
                     
            if GSMatrix.iloc[rows][cols+1] =='CR':
                if label == "IDENTICAL":
                     OverEstimation_66 = OverEstimation_66 +1
                if label == "SAME-TOPIC": 
                     OverEstimation_33 = OverEstimation_33 +1                      
                if label == "CONCEPT-RELATED":  
                     MATCH = MATCH +1
                     CR = CR+1                  
                if label == "NON-RELATED":  
                     UnderEstimation_33 = UnderEstimation_33 +1       

            if  GSMatrix.iloc[rows][cols+1] =='NR':
                if label == "IDENTICAL":
                     FALSES = FALSES +1
                if label == "SAME-TOPIC": 
                     OverEstimation_66 = OverEstimation_66 +1                      
                if label == "CONCEPT-RELATED":  
                     OverEstimation_33 = OverEstimation_33 +1                     
                if label == "NON-RELATED":  
                     MATCH = MATCH +1
                     NR = NR+1  
                        

    print("counter: ", counter)
    plotdata = {'CLASS': ['Total', 'Match', 'I','ST', 'CR','NR', 'UnderEst_33', 'UnderEst_66','OverEst_33', 'OverEst_66','Falses'],
                 'Number': [TOTAL, MATCH, I,ST, CR,NR, UnderEstimation_33, UnderEstimation_66, OverEstimation_33,OverEstimation_66, FALSES]}
    matches= MATCH/TOTAL
    under33 = UnderEstimation_33/TOTAL
    under66 = UnderEstimation_66/TOTAL
    over33 = OverEstimation_33/ TOTAL
    over66 = OverEstimation_66/ TOTAL
    falses= FALSES/TOTAL
    print(folder)
    print("MATCH % :")
    print(matches)
    print("UNDER 33 % :")
    print(under33)
    print("UNDER 66 % :")
    print(under66)
    print("OVER 33 % :")
    print(over33)
    print("OVER 66 % :")
    print(over66)
    print("FALSES % :")
    print(falses)
    print(I)
    print(ST)
    print(CR)
    print(NR)

    
    print("PREMATCH")
    print(PREMATCH)
    print("MATCH")
    print(MATCH)
    print("LEARNING: I CASE")
    print(delta)
    print("LEARNING: ST CASE")
    print(gamma)   
    print("LEARNING: CR CASE")
    print(beta)
    print("LEARNING: NR CASE")
    print(alpha)
    
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

    
    #print(CompMatrix) 
    dfplot = pd.DataFrame(plotdata)
    dfplot.plot(x='CLASS', y='Number', kind='bar')
    plt.show()
    #return CompMatrix     
    #
    #
    ############################################################################   


def AutoAssessment(listfile, folder, baseline,  GSMatrix, ResultsMatrix):
    # retrieve files set
    Files = getFiles(listfile)
    lsize = len(Files)
    print(lsize)    
    print(Files)
    # get starting baseline parameters and create new assessing parameters
    a = baseline.get("a")
    b = baseline.get("b")
    c = baseline.get("c")
    d = baseline.get("d")
    alpha = baseline.get("alpha")
    beta = baseline.get("beta")
    gamma = baseline.get("gamma")
    delta = baseline.get("delta")    
    
   # create the indices of files for  tunning parameters with pairs of documents
   # USE CODE FROM GOLD STD COMPARISON TO REDO THE ANALYSIS OF DOCUMENT PAIRS WHEN UNDERESTIMATION OR OVERESTIMATION

    rmrows, rmcols = np.shape(ResultsMatrix)
    print(rmrows)
    print(rmcols) 
    gsrows, gscols = np.shape(GSMatrix)
    print(gsrows)
    print(gscols)
    I_match = 0
    I_under =0
    I_over =0
    ST_match = 0
    ST_under =0
    ST_over =0    
    CR_match =0
    CR_under =0
    CR_over =0
    NR_match = 0
    NR_under =0
    NR_over =0
    MATCH = 0;
    OverEstimation_33 =0
    OverEstimation_66 =0
    UnderEstimation_33 = 0
    UnderEstimation_66 = 0
    FALSES =0
   
    TOTAL =0;
    CompMatrix =  [['' for x in range(rmrows)] for y in range(rmcols)]
    counter =0;
    grad_learn = 0.001
    for i in range(5):
        for rows in range(0, rmrows):
            file1 =Files[rows] 
            for cols in range(rows, rmcols):  
                file2 =Files[cols]
                #datafile ='/Sent_'+ file1 + "_"+ file2 +'.json'
                # Get deciles according for size sample estimation
                #pair_data =readResultsJson(datafile, folder)
                #values = pair_data.get("Value")
                #print(values)
                datafile ='/Data_'+ file1 + "_"+ file2 +'.json'
                pair_data =readResultsJson(datafile, folder)
                soundness =pair_data.get("Soundness")/10
                support =pair_data.get("Support")/10
                label = defuzzyfication(soundness, alpha, b, beta, c, gamma, delta)
                # SETTING PARAMETERS alpha, beta gamma and delta through ttahe central limit theorem
                if  GSMatrix.iloc[rows][cols+1] != " ":
                   # LEARNING
                     counter = counter+1 
                     print("LEARNING")  
                     if GSMatrix.iloc[rows][cols+1] =='I' and label !="IDENTICAL":
                         if soundness > delta:
                             delta = delta + grad_learn
                             if delta > 0.98:
                                 delta = 0.98
                             # Backwards  
                             if gamma > delta:
                                 gamma = delta -grad_learn
                             if beta > gamma:
                                 beta = gamma -grad_learn
                             if alpha > beta:
                                 alpha = beta -grad_learn
                     
                                 
                     if  GSMatrix.iloc[rows][cols+1] =='ST' and label !="SAME-TOPIC":
                         if soundness > gamma:
                             gamma = gamma + grad_learn
                             #Forward
                             if gamma > delta:
                                 delta = gamma + 2*grad_learn
                             #Backwards
                             if beta > gamma:
                                 beta = gamma -grad_learn
                             if alpha > beta:
                                 alpha = beta -grad_learn  
                     
                     if GSMatrix.iloc[rows][cols+1] =='CR' and label != "CONCEPT-RELATED":
                         if soundness > beta:
                             beta = beta + grad_learn
                             # Forward
                             if delta < 0.98:
                                 if beta > gamma:
                                    gamma =beta + 2*grad_learn
                                 if gamma > delta:
                                    delta = gamma + 2*grad_learn   
                             #Backwards
                             if alpha > beta:
                                 alpha = beta -grad_learn
                         else: 
                             if beta > alpha+ 3*grad_learn:
                                beta = beta -grad_learn

                     if GSMatrix.iloc[rows][cols+1] =='NR' and label != "NON-RELATED":
                         if soundness > alpha:
                             alpha = alpha + grad_learn
                         else:
                             if alpha > grad_learn:
                                 alpha = alpha - grad_learn
                
    
    PREMATCH = MATCH
    FALSES = 0
    MATCH = 0
    TOTAL = 0
    ## TEST AGAIN 

   # a = baseline.get("a")
   # b = baseline.get("b")
   # c = baseline.get("c")
   # d = baseline.get("d")
    alpha = baseline.get("alpha")
    beta = baseline.get("beta")
    gamma = baseline.get("gamma")
    delta = baseline.get("delta") 

    for rows in range(0, rmrows):
        file1 =Files[rows] 
        for cols in range(rows, rmcols):  
            file2 =Files[cols]
            datafile ='/Data_'+ file1 + "_"+ file2 +'.json'
            pair_data =readResultsJson(datafile, folder)
            soundness =(pair_data.get("Soundness"))/10
            support =pair_data.get("Support")/10
            TOTAL = TOTAL+1
            label = defuzzyfication(soundness, alpha, b, beta, c, gamma, delta)
               
            # TESTING

             #PÜT THE MATCHES AND TEST CASES HERE
            counter = counter+1 
            print("TESTING") 
            if GSMatrix.iloc[rows][cols+1] =='I':
                if label == "IDENTICAL":
                    MATCH = MATCH +1
                    I_match = I_match +1
                   
                if label == "SAME-TOPIC":  
                    UnderEstimation_33 = UnderEstimation_33 +1
                    I_under = I_under +1
                if label == "CONCEPT-RELATED":  
                    UnderEstimation_66 = UnderEstimation_66 +1 
                    I_under = I_under +1
                if label == "NON-RELATED":  
                    I_under = I_under +1
                    FALSES = FALSES +1                         
                     
            if GSMatrix.iloc[rows][cols+1] =='ST':
                if label == "IDENTICAL":
                    ST_over = ST_over +1
                    OverEstimation_33 = OverEstimation_33 +1
                if label == "SAME-TOPIC": 
                    MATCH = MATCH +1
                    ST_match = ST_match+1
                if label == "CONCEPT-RELATED": 
                    ST_under = ST_under +1
                    UnderEstimation_33 = UnderEstimation_33 +1                       
                if label == "NON-RELATED":
                    ST_under = ST_under +1
                    UnderEstimation_66 = UnderEstimation_66 +1                          
                     
                     
            if GSMatrix.iloc[rows][cols+1] =='CR':
                if label == "IDENTICAL":
                     CR_over = CR_over +1 
                     OverEstimation_66 = OverEstimation_66 +1
                if label == "SAME-TOPIC": 
                     CR_over = CR_over +1 
                     OverEstimation_33 = OverEstimation_33 +1                      
                if label == "CONCEPT-RELATED":  
                     MATCH = MATCH +1
                     CR_match = CR_match +1
                                      
                if label == "NON-RELATED":  
                     CR_under = CR_under +1 
                     UnderEstimation_33 = UnderEstimation_33 +1       

            if  GSMatrix.iloc[rows][cols+1] =='NR':
                if label == "IDENTICAL":
                     NR_over = NR_over + 1 
                     FALSES = FALSES +1
                if label == "SAME-TOPIC":
                     NR_over = NR_over + 1 
                     OverEstimation_66 = OverEstimation_66 +1                      
                if label == "CONCEPT-RELATED": 
                     NR_over = NR_over + 1 
                     OverEstimation_33 = OverEstimation_33 +1                     
                if label == "NON-RELATED":  
                     MATCH = MATCH +1
                     NR_match = NR_match+1  
                        

    print("counter: ", counter)
    plotdata = {'CLASS': ['Total', 'Match', 'I_Match', 'I_Over', 'I_under','ST_match', 'ST_Over', 'ST_Under','CR_Match', 'CR_over', 'CR_under','NR_Match', 'NR_over', 'NR_under', 'UnderEst_33', 'UnderEst_66','OverEst_33', 'OverEst_66','Falses'],
                 'Number': [TOTAL, MATCH, I_match, I_over,I_under, ST_match,ST_over, ST_under, CR_match, CR_over, CR_under,NR_match, NR_over, NR_under , UnderEstimation_33, UnderEstimation_66, OverEstimation_33,OverEstimation_66, FALSES]}
    matches= MATCH/TOTAL
    under33 = UnderEstimation_33/TOTAL
    under66 = UnderEstimation_66/TOTAL
    over33 = OverEstimation_33/ TOTAL
    over66 = OverEstimation_66/ TOTAL
    falses= FALSES/TOTAL
    I_match = I_match/TOTAL
    I_under = I_under/TOTAL
    ST_match = ST_match/TOTAL
    ST_over = ST_over/TOTAL
    ST_under = ST_under/TOTAL
    CR_match = CR_match/TOTAL
    CR_over = CR_over/TOTAL
    CR_under = CR_under/TOTAL
    NR_match = NR_match/TOTAL
    NR_over = NR_over/TOTAL
    

    print(f"\n\n\n\n")
    print(folder)
    
    print("FALSES % :")
    print(falses)

    print("UNDER-ESTIMATION 33 % :")
    print(under33)
    print("UNDER-ESTIMATION 66 % :")
    print(under66)
    print("OVER-ESTIMATION 33 % :")
    print(over33)
    print("OVER-ESTIMATION 66 % :")
    print(over66)
    
    print("MATCH % :")
    print(matches)
    print("I-MATCH % :")
    print(I_match)
    print("I-UNDER-ESTIMATION % :")
    print(I_under)

    print("ST-MATCH % :")
    print(ST_match)
    print("ST OVER-ESTIMATION % :")
    print(ST_over)
    print("ST UNDER-ESTIMATION % :")
    print(ST_under)
 
          
    print("CR-MATCH % :")
    print(CR_match)
    print("CR OVER-ESTIMATION % :")
    print(CR_over)
    print("CR UNDER-ESTIMATION % :")
    print(CR_under)
 
    print("NR-MATCH % :")
    print(NR_match)
    print("NR OVER-ESTIMATION % :")
    print(NR_over)


    print(f"\n\n\n\n")
    
    print("PREMATCH")
    print(PREMATCH)
    print("MATCH")
    print(MATCH)
    print("LEARNING: I CASE")
    print(delta)
    print("LEARNING: ST CASE")
    print(gamma)   
    print("LEARNING: CR CASE")
    print(beta)
    print("LEARNING: NR CASE")
    print(alpha)
    
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

    
    #print(CompMatrix) 
    dfplot = pd.DataFrame(plotdata)
    dfplot.plot(x='CLASS', y='Number', kind='bar', fontsize=20)  
    plt.title('MODEL: '+ folder)
    plt.legend(loc = 'best')
    plt.show()
    


    #return CompMatrix     
    #
    #
    ############################################################################   



def compareTextsSmallModelsbySentences(file1, file2, model, nlp, threshold): 
    # Produce the sentences from texts
    path = os.getcwd() + "/WORKSPACE/TEXTS/"
    sent1= getFile(file1, path, nlp)
    sent2= getFile(file2, path, nlp)
    # Produce the embeddings from sentences
    emb1 = getEmbeddingsSent_Transformer(sent1, model)
    emb2 = getEmbeddingsSent_Transformer(sent2, model)
    size1 = len(sent1)
    size2 = len(sent2)
    Matrix = [[0 for x in range(size2+1)] for y in range(size1+1)]
    Deciles = [0 for x in range(11)]
    SoftDeciles = [0 for x in range(11)]
    # Get the assessment from pairs of sentences
    # Complete processing here
    for a in range(1, size1):
        # Check if the sentence contains verb if not discard
        for b in range(1, size2):
            # check if the sentence contains verb if not discard         
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
    PairData = {"file1": file1, "file2": file2, "Matrix": Matrix, "mSize1": size1, "mSize2": size2, "Deciles": Deciles, "SoftDeciles": SoftDeciles,"Support": supprt, "Spanning": span, "Soundness": soundness}
    
    return PairData
    # Get the estimation of the similarity of documents 
    

def getFile(filename, path, nlp):
    if path == "":
       path = os.getcwd() 
    filename = path + "/"+ filename   
    filename = filename.replace("\\","/") 
    f = open(filename, "r", encoding="UTF-8")
    text = f.read()
    #print(text)
    sentences = re.split('[.\n]', text)
    #sentences = text.split(".")
   # print(sentences)
    ssize = len(sentences);
    print("Sentences size = " + str(ssize))
    mylist  = list()
    counter = 0
    buffer = ""
    for x in sentences:
        # test if x contains at least one SVO  
        flag = validSentenceSpacy(x, nlp) 
        if flag==True:
            x = buffer+ " " + x
            buffer = ""
            mylist.append(x)
            counter = counter+1
        else:
            buffer = buffer + " "+ x
    print("Sentences size = " + str(counter))       
    print(mylist)
    return mylist
    
def setNLP():
    nlp = spacy.load("en_core_web_sm")
    return nlp

def setModel(themodel):
    model = SentenceTransformer(themodel)
    return model

def getEmbeddingsSent_Transformer(sentences, model):
    ssize = len(sentences)
    print("Sentence size" + str(ssize))
    emb_list = list()
    for x in sentences:
        embeddings= model.encode(x, convert_to_tensor=True) 
        emb_list.append(embeddings)
    return emb_list

def getEmbeddings(sentences, themodel):
    ssize = len(sentences)
    print("Sentence size" + str(ssize))
    model = SentenceTransformer(themodel)
    emb_list = list()
    for x in sentences:
        embeddings= model.encode(x, convert_to_tensor=True) 
        emb_list.append(embeddings)
    return emb_list

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

def getTextDetails(listfiles):
    path =  os.getcwd() + '/' + listfiles
    path = path.replace("\\","/") 
    f = open(path, "r", encoding="UTF-8")
    text = f.read()
    filenames = text.split('\n')
    numwords  = list()
    numsentences  = list()
    fileList = list()
    for x in filenames:
        filestr = os.getcwd() + '/WORKSPACE/DATASET/' + x   
        filestr = filestr.replace("\\","/") 
        file = open(filestr, "r", encoding="UTF-8")
        text = file.read()
        words = text.split(None)
        sentences= text.split(".")
        print(x)
        print(len(words))
        print(len(sentences))
        fileList.append(x)
        numwords.append(len(words))
        numsentences.append(len(sentences))
    MyDict = {"Files":fileList, "Sent_size": numsentences,  "Wordsize":numwords }
    myfileClass = os.getcwd()+ '/WORKSPACE/DatasetDetails.json'
    myfileClass = myfileClass.replace("\\","/") 
    with open(myfileClass, 'w') as convert_file: 
        convert_file.write(json.dumps(MyDict))  
    return MyDict

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

def ComparebyIndices(sent1_index , sent2_index, embeddings1, embeddings2):
    emb1 = int(sent1_index)
    emb2 = int(sent2_index)
    cosine_score = util.cos_sim(embeddings1[emb1], embeddings2[emb2])
    return cosine_score.item()

def preprocessDatasetwithShuffle(fileSet, nlp):
    path= os.getcwd() + '/WORKSPACE/DATASET/'
    sentpath = os.getcwd() + '/WORKSPACE/SENTENCES/'
    filesetpath = os.getcwd() +"/" + fileSet
    filesetpath = filesetpath.replace("\\","/") 
    f = open(filesetpath, "r", encoding="UTF-8")
    text = f.read()
    files = text.split('\n')
    ssize = len(files );
    print("Number of Files  = " + str(ssize))
    mylist  = list()
    for x in files:
        sentences= RandomizeText(x, path, nlp)
        mylist.append(" S_Shuffled_"+ x)
        #sentences = getFile(x, path, nlp)
        #print(sentences)
        
        myoutputfile = os.getcwd()+ '/WORKSPACE/SENTENCES/S_Shuffled_'+ x
        myoutputfile = myoutputfile.replace("\\","/") 
        with open(myoutputfile, 'w', encoding="UTF-8") as convert_file2: 
            for sent in sentences:
                convert_file2.write(sent+ "\n")
    return mylist

def preprocessDataset(fileSet, nlp):
    path= os.getcwd() + '/WORKSPACE/DATASET/'
    sentpath = os.getcwd() + '/WORKSPACE/SENTENCES/'
    filesetpath = os.getcwd() +"/" + fileSet
    filesetpath = filesetpath.replace("\\","/") 
    f = open(filesetpath, "r", encoding="UTF-8")
    text = f.read()
    files = text.split('\n')
    ssize = len(files );
    print("Number of Files  = " + str(ssize))
    mylist  = list()
    for x in files:
        mylist.append(x)
        sentences = getFile(x, path, nlp)
        print(sentences)
        myoutputfile = os.getcwd()+ '/WORKSPACE/SENTENCES/S_'+ x
        myoutputfile = myoutputfile.replace("\\","/") 
        with open(myoutputfile, 'w', encoding="UTF-8") as convert_file2: 
            for sent in sentences:
                convert_file2.write(sent+ "\n")
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

def SystematicPairClassificationShuffled(listfile, model,  nlp, a,b,c,d,alpha, beta, gamma, delta):
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
            file2 = "Shuffled_"+ Files[k]
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

def SystematicPairClassification(listfile, model,  nlp, a,b,c,d,alpha, beta, gamma, delta):
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

def SystematicSelfClassificationShuffled(listfile, model,  nlp, a,b,c,d,alpha, beta, gamma, delta):
    Files = getFiles(listfile)
    lsize = len(Files)
    print(lsize)    
    print(Files)
    for i in range(0,lsize):
        file1 = Files[i]
        file2 = "Shuffled_"+ file1
        print(file1)
        print(file2)
        
        Pairdata = DATASET_AttentionOnSentences(file1, file2 , 0.0, model, nlp)
        class_data = classifyPair(Pairdata,a,b,c,d,alpha, beta, gamma, delta)
        SentPairs = selectRepresentativePairs(Pairdata)
        saveResults(Pairdata, SentPairs,class_data )
        
def SystematicSelfClassification(listfile, model,  nlp, a,b,c,d,alpha, beta, gamma, delta):
    Files = getFiles(listfile)
    lsize = len(Files)
    print(lsize)    
    print(Files)
    for i in range(0,lsize):
        file1 = Files[i]
        print(file1)
        Pairdata = DATASET_AttentionOnSentences(file1, file1 , 0.0, model, nlp)
        class_data = classifyPair(Pairdata,a,b,c,d,alpha, beta, gamma, delta)
        SentPairs = selectRepresentativePairs(Pairdata)
        saveResults(Pairdata, SentPairs,class_data )


# CREATE ASSESSMENT WITH PREVIOUS  CLASSIFICATION ANALYSIS

def SystematicPairReClassification(listfile,folder, model,  nlp, a,b,c,d,alpha, beta, gamma, delta):
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
            class_data = classifyPair(Pairdata,a,b,c,d,alpha, beta, gamma, delta)
            s = class_data.get("Relation")
            Matrix[i][k] =s[:2]
            SentPairs = selectRepresentativePairs(Pairdata)
            saveResults(Pairdata, SentPairs,class_data )
            
    
    Dictionary ={"Files": Files, "Matrix": Matrix}
    myfileClass = os.getcwd()+ '/WORKSPACE/RESULTS/GlobalResults'+'.json'
    myfileClass = myfileClass.replace("\\","/") 
    with open(myfileClass, 'w') as convert_file: 
        convert_file.write(json.dumps(Dictionary)) 

# CREATE A FUNCTION THAT USES PAIRDATA TO CLASSIFY AGAIN 
    

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
                
# FOR SVO 
nlp = setNLP()
## CHOOSE THE MODEL
#themodel = 'all-MiniLM-L6-v2'
#themodel = 'all-MiniLM-L12-v2'
#themodel = 'all-mpnet-base-v2'
##themodel = 'sentence-transformers/average_word_embeddings_glove.6B.300d'

#model = setModel(themodel)



# CREATION OS SENTENCES VERSIONS
#fileSet= 'DatasetListfile.txt'
#preprocessDataset(fileSet, nlp)


## TESTING PAIRS FROM SCRATCH WITHOUT PREVIOUS PREPROCESSING
#path = os.getcwd() +"/WORKSPACE/TEXTS/"
#filename1 = "Logic .txt"
#filename2 = "Immigration .txt"
#content = getFile(filename, path, nlp)
#mycomparison= compareTextsSmallModelsbySentences(filename1, filename2, model, nlp, 0) 
#print(mycomparison)  


    
# Model can be one of the followings
# 'all-MiniLM-L6-v2'
# 'all-MiniLM-L12-v2'
# 'all-mpnet-base-v2'
# 'sentence-transformers/average_word_embeddings_glove.6B.300d'



# SHUFFLE TEXT

#fileSet= 'DatasetListfile.txt'
#preprocessDatasetwithShuffle(fileSet, nlp)


# PARAMETERS FOR CLASSIFICATION

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

#alpha = 0.25;
#beta = 0.3;
#gamma = 0.6;
#delta =0.7;

## FINE TUNNING
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

#SELF COMPARISON
#SystematicSelfClassification('DatasetListfile.txt', model,  nlp, a,b,c,d,alpha, beta, gamma, delta)




## APPLY A SYSTEMATIC  CLASSIFICATION -- CREATE THE FOLDER FIRST!!
#folder = 'RESULTS_all-MiniLM-L6-v2'
#folder = 'RESULTS_all-MiniLM-L12-v2'
#folder = 'RESULTS_all-mpnet-base-v2'
#folder = 'RESULTS_glove.300D'
#SystematicPairClassification('DatasetListfile.txt', model,  nlp, a,b,c,d,alpha, beta, gamma, delta)

## APPLY ANALYSIS
#folder = 'RESULTS_all-MiniLM-L6-v2'
##SystematicPairReClassification('DatasetListfile.txt' ,folder, model,  nlp, a,b,c,d,alpha, beta, gamma, delta)
#MyDict = GetAnalysis('DatasetListfile.txt', folder)
##Dict = GetAnalysis('DatasetListfile.txt', 'RESULTS')
#print(MyDict)
#results = readResultsJson('GlobalResults.json', folder )
##results = readResultsJson('GlobalResults.json', 'RESULTS' )
#ResultsMatrix = results.get("Matrix")
#print(ResultsMatrix)

#filename = "Dataset 72 Docs Gold-Standard.xlsx"
#path = os.getcwd()
#sheet = "Hoja2"
#goldstd = importExcelFile(filename, path, sheet)
#goldstdmatrix = pd.read_csv('C:\RESEARCH PROJECTS\MIXED_ARCHITECTURE\Dataset 72 Docs Gold-Standard.csv', header=0)
#print(goldstdmatrix.iloc[0][1])
#CompMatrix = CompareGold_STD(ResultsMatrix, goldstdmatrix, folder )





## GET SENTENCES

#List =getNumSentences('DatasetListfile.txt')

# EXAMPLE OF USING A SINGLE COMPARISON
#file1 = '2022 Russian invasion of Ukraine.txt'
#file2 = '2022 Russian invasion of Ukraine.txt'
#Pairdata = DATASET_AttentionOnSentences(file1, file2 , 0.0, themodel, nlp)
#SentPairs = selectRepresentativePairs(Pairdata)
#saveResults(Pairdata, SentPairs)
#data = readResults('Sent_'+ file1 + "_"+ file2 +'.json')
#print(data)
