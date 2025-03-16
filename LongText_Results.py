import numpy as np
import os
import pandas as pd
import json
import seaborn as sn
import matplotlib.pyplot as plt

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

def getTimePerformance(filelist):
    Files = getFiles(filelist)
    lsize = len(Files)
    print(lsize)    
    print(Files)
    # Matrix for saving 
    Matrix = [['' for x in range(lsize)] for y in range(lsize)]
    sumTime =0
    iterations = 0
    for i in range(0,lsize):
        file1 = Files[i]
        print(file1)
        for k in range(i,lsize):
            file2 =  Files[k]
            print(file2)
            myfile = os.getcwd()+ '/WORKSPACE/' + folder + '/Data_' + file1 +'_'+ file2 +'.json'
            myfile2 = os.getcwd()+ '/WORKSPACE/' + folder + '/Matrix_' + file1 +'_'+ file2 +'.json'
            print(myfile2)
            Pairdata =RetrieveClass(myfile)
            time = Pairdata.get("Time")
            sumTime = sumTime+time
            iterations = iterations+1

            #PairMatrix =RetrieveClass(myfile2)
    avg_time = sumTime/iterations
    print("Total time: ", sumTime)
    print("Iterations: ", iterations)
    print("Average time: ", avg_time)


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
    NON_CATCHED = 0
    #CompMatrix =  [['' for x in range(rmrows)] for y in range(rmcols)]
    ConfusionMatrix =  [[0 for x in range(4)] for y in range(4)]
    catched = 0
    for rows in range(0, rmrows):
        for cols in range(rows, rmcols):  
            TOTAL = TOTAL+1
            #print("ResultsMatrix")
            #print(ResultsMatrix[rows][cols])
            #print("Gold_STD")
            #print(Gold_STD.iloc[rows, cols+1])
            catched = 0
            if ResultsMatrix[rows][cols] == Gold_STD.iloc[rows, cols+1]:
                 
                 MATCH = MATCH +1
                 if ResultsMatrix[rows][cols]== 'I':
                     catched = 1
                     I_I=I_I+1
                 if ResultsMatrix[rows][cols]== 'ST':
                     ST_ST=ST_ST+1     
                     catched = 1
                 if ResultsMatrix[rows][cols]== 'CR':
                     CR_CR=CR_CR+1  
                     catched = 1
                 if ResultsMatrix[rows][cols]== 'NR':
                     NR_NR=NR_NR+1 
                     catched = 1
            else:
                 if ResultsMatrix[rows][cols] == 'I' and Gold_STD.iloc[rows,cols+1] =='ST':
                     I_ST=I_ST+1
                     catched = 1
                 if ResultsMatrix[rows][cols] == 'ST' and Gold_STD.iloc[rows, cols+1] =='I':
                     ST_I=ST_I+1
                     catched = 1
                 if ResultsMatrix[rows][cols] == 'I' and Gold_STD.iloc[rows, cols+1] =='CR':
                     I_CR=I_CR+1
                     catched = 1
                 if ResultsMatrix[rows][cols] == 'CR' and Gold_STD.iloc[rows, cols+1] =='I':  
                     CR_I=CR_I+1 
                     catched = 1
                 if ResultsMatrix[rows][cols] == 'I' and Gold_STD.iloc[rows, cols+1] =='NR':
                     I_NR=I_NR+1
                     catched = 1
                 if ResultsMatrix[rows][cols] == 'NR' and Gold_STD.iloc[rows, cols+1] =='I':
                     NR_I=NR_I+1 
                     catched = 1
                 if ResultsMatrix[rows][cols] == 'ST' and Gold_STD.iloc[rows, cols+1] =='CR':
                     ST_CR=ST_CR+1
                     catched = 1
                 if ResultsMatrix[rows][cols] == 'CR' and Gold_STD.iloc[rows, cols+1] =='ST':
                     CR_ST=CR_ST
                     catched = 1
                 if ResultsMatrix[rows][cols] == 'ST' and Gold_STD.iloc[rows, cols+1] =='NR':
                     ST_NR=ST_NR+1
                     catched = 1
                 if ResultsMatrix[rows][cols] == 'NR' and Gold_STD.iloc[rows, cols+1] =='ST':
                     NR_ST=NR_ST+1 
                     catched = 1
                 if ResultsMatrix[rows][cols] == 'CR' and Gold_STD.iloc[rows, cols+1] =='NR':
                    CR_NR=CR_NR+1
                    catched = 1
                 if ResultsMatrix[rows][cols] == 'NR' and Gold_STD.iloc[rows, cols+1] =='CR':
                    NR_CR=NR_CR+1
                    catched = 1

            if catched ==0:
                NON_CATCHED = NON_CATCHED+1
                                           
    plotdata = {'CLASS': ['Total', 'Match', 'I_I', 'I_ST', 'I_CR', 'I_NR', 'ST_ST', 'ST_I', 'ST_CR', 'ST_NR', 'CR_CR', 'CR_I', 'CR_ST', 'CR_NR', 'NR_NR', 'NR_I', 'NR_ST' , 'NR_CR'],
                 'Number': [TOTAL, MATCH, I_I, I_ST, I_CR, I_NR, ST_ST, ST_I, ST_CR, ST_NR, CR_CR, CR_I, CR_ST, CR_NR, NR_NR, NR_I, NR_ST , NR_CR]}
    matches= MATCH/TOTAL
    print("NON_CATCHED")
    print(NON_CATCHED)
    
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
    ConfusionMatrix[2][3] = CR_NR

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


#folder = 'RESULTS_all-MiniLM-L6-v2'
#folder = 'RESULTS_all-MiniLM-L12-v2'
#folder = 'RESULTS_all-mpnet-base-v2'
#folder = 'RESULTS_glove.300D'
#folder = 'RESULTS_CURRENT'
#folder = 'RESULTS_LongFormer_1K'
#folder = 'RESULTS_BigBird_1K'

folder = 'RESULTS__all-MiniLM-L6-v2_2_2_5_6_8_9'

#folder = 'RESULTS'


listfile = 'DatasetListfile.txt'
#getTimePerformance(listfile)
filename = "Dataset 72 Docs Gold-Standard.xlsx"
path = os.getcwd()
#SystematicPairClassification(listfile, model, nlp, b, c, alpha, beta, gamma, delta)
#SystematicPairReClassification(listfile,folder, b, c, alpha, beta, gamma, delta)
results = readResultsJson('GlobalResults.json', folder )
#results = readResultsJson('GlobalResults_NFirst.json', folder )
ResultsMatrix = results.get("Matrix")
goldstdmatrix = pd.read_csv('C:\RESEARCH PROJECTS\MIXED_ARCHITECTURE\Dataset 72 Docs Gold-Standard.csv', header=0)
print(results)
print(goldstdmatrix.iloc[0,1])
#CompMatrix = CompareGold_STD(ResultsMatrix, goldstdmatrix, folder )
ConfusionMatrix = GetConfusionMatrix(ResultsMatrix, goldstdmatrix, folder)
print(ConfusionMatrix)

TP_I  = ConfusionMatrix[0][0] 
TP_ST = ConfusionMatrix[1][1] 
TP_CR = ConfusionMatrix[2][2] 
TP_NR = ConfusionMatrix[3][3]

FP_I  = ConfusionMatrix[1][0] + ConfusionMatrix[2][0] + ConfusionMatrix[3][0]
FP_ST = ConfusionMatrix[0][1] + ConfusionMatrix[2][1] + ConfusionMatrix[3][1]
FP_CR = ConfusionMatrix[0][2] + ConfusionMatrix[1][2] + ConfusionMatrix[3][2]
FP_NR = ConfusionMatrix[0][3] + ConfusionMatrix[1][3] + ConfusionMatrix[2][3]

FN_I  = ConfusionMatrix[0][1] + ConfusionMatrix[0][2] + ConfusionMatrix[0][3]
FN_ST = ConfusionMatrix[1][0] + ConfusionMatrix[1][2] + ConfusionMatrix[1][3]
FN_CR = ConfusionMatrix[2][0] + ConfusionMatrix[2][1] + ConfusionMatrix[2][3]
FN_NR = ConfusionMatrix[3][0] + ConfusionMatrix[3][1] + ConfusionMatrix[3][2]

TN_I =  ConfusionMatrix[1][1] + ConfusionMatrix[1][2] + ConfusionMatrix[1][3] +ConfusionMatrix[2][1]+ ConfusionMatrix[2][2] + ConfusionMatrix[2][3]+ ConfusionMatrix[3][1]+ ConfusionMatrix[3][2] + ConfusionMatrix[3][3]
TN_ST = ConfusionMatrix[0][0] + ConfusionMatrix[0][2] + ConfusionMatrix[0][3] +ConfusionMatrix[2][0]+ ConfusionMatrix[2][2] + ConfusionMatrix[2][3]+ ConfusionMatrix[3][0]+ ConfusionMatrix[3][2] + ConfusionMatrix[3][3]
TN_CR = ConfusionMatrix[0][0] + ConfusionMatrix[0][1] + ConfusionMatrix[0][3] +ConfusionMatrix[2][0]+ ConfusionMatrix[2][1] + ConfusionMatrix[2][3]+ ConfusionMatrix[3][0]+ ConfusionMatrix[3][1] + ConfusionMatrix[3][3]
TN_NR = ConfusionMatrix[0][0] + ConfusionMatrix[0][1] + ConfusionMatrix[0][2] +ConfusionMatrix[1][0]+ ConfusionMatrix[1][1] + ConfusionMatrix[1][2]+ ConfusionMatrix[2][0]+ ConfusionMatrix[2][1] + ConfusionMatrix[2][2]

accuraccy_I = (TP_I + FP_I )/ (TP_I + FP_I + FN_I+ TN_I)

prec_I = TP_I / (TP_I + FP_I )
print("PREC_I =", prec_I)
prec_ST = TP_ST / (TP_ST + FP_ST )
print("PREC_ST =", prec_ST)
prec_CR = TP_CR / (TP_CR + FP_CR )
print("PREC_CR =", prec_CR)
prec_NR = TP_NR / (TP_NR + FP_NR )
print("PREC_NR =", prec_NR)

if (TP_I + FN_I)==0:
    recall_I =0
else:
    recall_I = TP_I / (TP_I + FN_I )
print("RECALL_I =", recall_I)
if (TP_ST + FN_ST)==0:
    recall_ST =0
else:
    recall_ST = TP_ST / (TP_ST + FN_ST )
print("RECALL_ST =", recall_ST)
if (TP_CR + FN_CR)==0:
    recall_CR =0
else:
    recall_CR = TP_CR / (TP_CR + FN_CR )
print("RECALL_CR =", recall_CR)
if (TP_CR + FN_CR)==0:
    recall_NR =0
else:
   recall_NR = TP_NR / (TP_NR + FN_NR )
print("RECALL_NR =", recall_CR)


if (prec_I+recall_I)==0:
    F1_I  =0
else:
    F1_I = 2*(prec_I*recall_I)/(prec_I+recall_I)
print("F1_I =", F1_I)
if (prec_ST + recall_ST)==0:
    F1_ST  =0
else:    
    F1_ST = 2*(prec_ST*recall_ST)/(prec_ST+recall_ST)
print("F1_ST =", F1_ST)
if (prec_CR +recall_CR)==0:
    F1_CR  =0
else:
    F1_CR = 2*(prec_CR*recall_CR)/(prec_CR+recall_CR)
print("F1_CR =", F1_CR)
if (prec_NR +recall_CR)==0:
    F1_NR  =0
else:
    F1_NR = 2*(prec_NR*recall_NR)/(prec_NR+recall_NR)
print("F1_NR =", F1_NR)

#accuraccy_I = (TP_I + FP_I )/ (TP_I + FP_I + FN_I+ TN_I)
#print("accuraccy_I =", accuraccy_I)
#accuraccy_ST = (TP_ST + FP_ST )/ (TP_ST + FP_ST + FN_ST+ TN_ST)
#print("accuraccy_ST =", accuraccy_ST)
#accuraccy_CR = (TP_CR + FP_CR )/ (TP_CR + FP_CR + FN_CR+ TN_CR)
#print("accuraccy_CR =", accuraccy_CR)
#accuraccy_NR = (TP_NR + FP_NR )/ (TP_NR + FP_NR + FN_NR+ TN_NR)
#print("accuraccy_NR =", accuraccy_NR)

accuraccy_I = (TP_I  )/ (TP_I + FP_I + FN_I)
print("accuraccy_I =", accuraccy_I)
accuraccy_ST = (TP_ST  )/ (TP_ST + FP_ST + FN_ST)
print("accuraccy_ST =", accuraccy_ST)
accuraccy_CR = (TP_CR  )/ (TP_CR + FP_CR + FN_CR)
print("accuraccy_CR =", accuraccy_CR)
accuraccy_NR = (TP_NR  )/ (TP_NR + FP_NR + FN_NR)
print("accuraccy_NR =", accuraccy_NR)

df_cm = pd.DataFrame(ConfusionMatrix, index = ['identical', 'same topic', 'concept related', 'nonrelated'], columns = ['identical', 'same topic', 'concept related', 'nonrelated'])
sn.set(font_scale=1) # for label size
plt.figure(figsize=(4,4))
ax = sn.heatmap(df_cm, cmap="crest", annot=True, linewidth=.5)
ax.set(xlabel="", ylabel="")
ax.xaxis.tick_top()
plt.show() 

