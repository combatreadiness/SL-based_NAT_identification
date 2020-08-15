import pandas
import random
import csv
import numpy as np
import graphviz 

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from scipy import interp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc


#import matplotlib.pyplot as plt

train = pandas.read_csv('final_data_only_10_highest.csv')
y = train['nat']
var =[]
for i in train:
        var.append(i) 
del var[-1]
X = train[var]


out = open('./final_roc_top10.csv','w',newline='')
csvWriter =csv.writer(out)
column = []
out2 = open('./final_pr_c_top10.csv','w',newline='')
csvWriter2 =csv.writer(out2)
column = []
#column.extend(['LR.P','LR.R','LR.A','LR.F1','LR.t','LR.t2'])
#column.extend(['SVM.P','SVM.R','SVM.A','SVM.F1','SVM.t','SVM.t2'])
#column.extend(['TR.P','TR.R','TR.A','TR.F1','TR.t','TR.t2'])
#column.extend(['KNN.P','KNN.R','KNN.A','KNN.F1','KNN.t','KNN.t2'])
#column.extend(['MLP.P','MLP.R','MLP.A','MLP.F1','MLP.t','MLP.t2'])
#column.extend(['RF.P','RF.R','RF.A','RF.F1','RF.t','Rf.t2'])

csvWriter.writerow(column)

LR_tprs = []
LR_aucs = []
LR_mean_fpr = np.linspace(0, 1, 100)
LR_prc  = []
LR_rec = []
LR_av_prc=[]
sv_tprs = []
sv_aucs = []
sv_mean_fpr = np.linspace(0, 1, 100)
sv_prc  = []
sv_rec = []
sv_av_prc=[]
tr_tprs = []
tr_aucs = []
tr_mean_fpr = np.linspace(0, 1, 100)
tr_prc = []
tr_rec = []
tr_av_prc=[]
knn_tprs = []
knn_aucs = []
knn_mean_fpr = np.linspace(0, 1, 100)
knn_prc = []
knn_rec = []
knn_av_prc=[]
mlp_tprs = []
mlp_aucs = []
mlp_mean_fpr = np.linspace(0, 1, 100)
mlp_prc = []
mlp_rec = []
mlp_av_prc=[]
rf_tprs = []
rf_aucs = []
rf_mean_fpr = np.linspace(0, 1, 100)
rf_prc = []
rf_rec = []
rf_av_prc=[]

for n in range(1000):

    
    RANDOM_SEED = random.randrange(0,2147483647)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=RANDOM_SEED)
    
#    print(y_test)
        #Logistic Regression 
#    lr = LogisticRegression(C=1.0, penalty = 'l1', tol=1e-6)
   
    
    lr = LogisticRegression()
    probas_=lr.fit(X_train, y_train).predict_proba(X_test)
   
#    score_=lr.fit(X_train, y_train).decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    LR_tprs.append(interp(LR_mean_fpr, fpr, tpr))
    LR_tprs[-1][0] = 0.0
    LR_roc_auc = auc(fpr, tpr)
    LR_aucs.append(LR_roc_auc)
#    y_score=lr.decision_function(X_test)
    y_score=probas_[:, 1]
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    if(len(LR_prc)<len(precision)):
        LR_prc=precision
        LR_rec=recall
    LR_av_prc.append(average_precision)
        
    
    sv = svm.SVC( probability=True)
    probas_=sv.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    sv_tprs.append(interp(sv_mean_fpr, fpr, tpr))
    sv_tprs[-1][0] = 0.0
    sv_roc_auc = auc(fpr, tpr)
    sv_aucs.append(sv_roc_auc)
#    y_score=sv.decision_function(X_test)
    y_score=probas_[:, 1]
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    if(len(sv_prc)<len(precision)):
        sv_prc=precision
        sv_rec=recall
    sv_av_prc.append(average_precision)

    tr = tree.DecisionTreeClassifier()  
    probas_=tr.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    tr_tprs.append(interp(tr_mean_fpr, fpr, tpr))
    tr_tprs[-1][0] = 0.0
    tr_roc_auc = auc(fpr, tpr)
    tr_aucs.append(tr_roc_auc)
    y_score=probas_[:, 1]
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    if(len(tr_prc)<len(precision)):
        tr_prc=precision
        tr_rec=recall
    tr_av_prc.append(average_precision)
        
    n_neighbors = 3
    knn = neighbors.KNeighborsClassifier(n_neighbors)
    probas_=knn.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    knn_tprs.append(interp(knn_mean_fpr, fpr, tpr))
    knn_tprs[-1][0] = 0.0
    knn_roc_auc = auc(fpr, tpr)
    knn_aucs.append(knn_roc_auc)
    y_score=probas_[:, 1]
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    if(len(knn_prc)<len(precision)):
        knn_prc=precision
        knn_rec=recall
    knn_av_prc.append(average_precision)
        
    mlp= MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(15,), random_state=1)
    probas_=mlp.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    mlp_tprs.append(interp(mlp_mean_fpr, fpr, tpr))
    mlp_tprs[-1][0] = 0.0
    mlp_roc_auc = auc(fpr, tpr)
    mlp_aucs.append(mlp_roc_auc)
    y_score=probas_[:, 1]
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    if(len(mlp_prc)<len(precision)):
        mlp_prc=precision
        mlp_rec=recall
    mlp_av_prc.append(average_precision)
    

    rf = RandomForestClassifier()
    probas_=rf.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    rf_tprs.append(interp(rf_mean_fpr, fpr, tpr))
    rf_tprs[-1][0] = 0.0
    rf_roc_auc = auc(fpr, tpr)
    rf_aucs.append(rf_roc_auc)
    y_score=probas_[:, 1]
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    if(len(rf_prc)<len(precision)):
        rf_prc=precision
        rf_rec=recall
    rf_av_prc.append(average_precision)


#LR
mean_tpr = np.mean(LR_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(LR_mean_fpr, mean_tpr)
std_auc = np.std(LR_aucs)
row=[]
row.append("Logistic Regression")
row.append(mean_auc)
row.append(std_auc)
csvWriter.writerow(row)
csvWriter.writerow(LR_mean_fpr)
csvWriter.writerow(mean_tpr)
row=[]
row.append("Logistic Regression")
row.append(np.mean(LR_av_prc))
csvWriter2.writerow(row)
csvWriter2.writerow(LR_prc)
csvWriter2.writerow(LR_rec)

#SVM
mean_tpr = np.mean(sv_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(sv_mean_fpr, mean_tpr)
std_auc = np.std(sv_aucs)
row=[]
row.append("Support Vector Machine")
row.append(mean_auc)
row.append(std_auc)
csvWriter.writerow(row)
csvWriter.writerow(sv_mean_fpr)
csvWriter.writerow(mean_tpr)
row=[]
row.append("Support Vector Machine")
row.append(np.mean(LR_av_prc))
csvWriter2.writerow(row)
csvWriter2.writerow(sv_prc)
csvWriter2.writerow(sv_rec)

#TR
mean_tpr = np.mean(tr_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(tr_mean_fpr, mean_tpr)
std_auc = np.std(tr_aucs)
row=[]
row.append("Decision Tree")
row.append(mean_auc)
row.append(std_auc)
csvWriter.writerow(row)
csvWriter.writerow(tr_mean_fpr)
csvWriter.writerow(mean_tpr)
row=[]
row.append("Decision Tree")
row.append(np.mean(tr_av_prc))
csvWriter2.writerow(row)
csvWriter2.writerow(tr_prc)
csvWriter2.writerow(tr_rec)

mean_tpr = np.mean(knn_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(knn_mean_fpr, mean_tpr)
std_auc = np.std(knn_aucs)
row=[]
row.append("K-Nearest Neighbor")
row.append(mean_auc)
row.append(std_auc)
csvWriter.writerow(row)
csvWriter.writerow(knn_mean_fpr)
csvWriter.writerow(mean_tpr)
row=[]
row.append("K-Nearest Neighbor")
row.append(np.mean(knn_av_prc))
csvWriter2.writerow(row)
csvWriter2.writerow(knn_prc)
csvWriter2.writerow(knn_rec)


mean_tpr = np.mean(mlp_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mlp_mean_fpr, mean_tpr)
std_auc = np.std(mlp_aucs)
row=[]
row.append("Multilayer Perceptron")
row.append(mean_auc)
row.append(std_auc)
csvWriter.writerow(row)
csvWriter.writerow(mlp_mean_fpr)
csvWriter.writerow(mean_tpr)
row=[]
row.append("Multilayer Perceptron")
row.append(np.mean(mlp_av_prc))
csvWriter2.writerow(row)
csvWriter2.writerow(mlp_prc)
csvWriter2.writerow(mlp_rec)

mean_tpr = np.mean(rf_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(rf_mean_fpr, mean_tpr)
std_auc = np.std(rf_aucs)
row=[]
row.append("Random Forest")
row.append(mean_auc)
row.append(std_auc)
csvWriter.writerow(row)
csvWriter.writerow(rf_mean_fpr)
csvWriter.writerow(mean_tpr)
row=[]
row.append("Random Forest")
row.append(np.mean(rf_av_prc))
csvWriter2.writerow(row)
csvWriter2.writerow(rf_prc)
csvWriter2.writerow(rf_rec)

out.close()
out2.close()