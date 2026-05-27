import numpy as np, os
from sklearn.metrics import *
# from Metric import *
import sys

def MacroAUC(output, label):
    y_pred = output
    y_true = label
    num_instance,num_class = y_pred.shape
    count = np.zeros((num_class,1))
    num_P_instance =  np.zeros((num_class,1)) 
    num_N_instance =  np.zeros((num_class,1)) 
    auc = np.zeros((num_class,1))
    count_valid_label = 0
    for  i in range(num_class):
        num_P_instance[i,0] = sum(y_true[:,i] == 1)
        num_N_instance[i,0] = num_instance - num_P_instance[i,0]
        if num_P_instance[i,0] == 0 or num_N_instance[i,0] == 0:
            auc[i,0] = 0
            count_valid_label = count_valid_label + 1
        else:
            temp_P_Outputs = np.zeros((int(num_P_instance[i,0]), num_class))
            temp_N_Outputs = np.zeros((int(num_N_instance[i,0]), num_class))
  
            temp_P_Outputs[:,i] = y_pred[y_true[:,i]==1,i]
            temp_N_Outputs[:,i] = y_pred[y_true[:,i]==0,i]    
            for m in range(int(num_P_instance[i,0])):
                for n in range(int(num_N_instance[i,0])):
                    if(temp_P_Outputs[m,i] > temp_N_Outputs[n,i] ):
                        count[i,0] = count[i,0] + 1
                    elif(temp_P_Outputs[m,i] == temp_N_Outputs[n,i]):
                        count[i,0] = count[i,0] + 0.5
            
            auc[i,0] = count[i,0]/(num_P_instance[i,0]*num_N_instance[i,0])  
    macroAUC1 = sum(auc)/(num_class-count_valid_label)
    return  float(macroAUC1)

def ptbxl_Result(class_num, y_pred, y_test, baseline = 0.5):
    output_labels=[]
    
    for i,key in enumerate(y_pred):
        output_label=[]
        for j in range(len(key)):
            if(key[j] >= baseline):
                output_label.append(1)
            else:
                output_label.append(0)
        output_label=np.array(output_label)
        output_labels.append(output_label)
    
    output_labels=np.array(output_labels)
    output=output_labels
    label=y_test

    auc = MacroAUC(output,label) 
 
    y_pred=np.where(output>0.5,1,0)
    acc=accuracy_score(y_pred,label)

    return acc, auc

def Sen(con_mat,n=4):
    
    sen = []
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)
        
    return sen

def Spe(con_mat,n=4):
    
    spe = []
    temp = 0
    for i in range(n):
        temp += con_mat[i][i]
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = (temp - tp) / (tn + fp )
        #print(spe1)
        spe.append(spe1)
    
    return spe
    
def ACC(con_mat,n=4):
    
    acc = []
    number = np.sum(con_mat[:,:])
    temp = 0
    for i in range(n):
        temp += con_mat[i][i]
    acc = temp / number    
    return acc

def ptb_Result(class_num,y_pred,y_test):
    y_pred = np.argmax(y_pred,axis=1)
    if(int(class_num)==7):
        target_names = ['AMI','ASMI','ALMI','IMI','ILMI','other','NORM']
    else:
        print("num_class error!")
        sys.exit()

    con_mat = confusion_matrix(y_test, y_pred)  # 把输入转变为混淆矩阵
    n = con_mat.shape[0]
    sen = Sen(con_mat,n)
    spe = Spe(con_mat,n)
    acc = ACC(con_mat,n)
    return np.sum(sen)/n, np.sum(spe)/n, acc