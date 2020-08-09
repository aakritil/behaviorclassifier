#!/usr/bin/env python
# coding: utf-8

# In[81]:


# Pandas is used for data manipulation
import pandas as pd
from sklearn.model_selection import cross_validate
import sklearn.preprocessing
import time

# Use numpy to convert to arrays
import numpy as np

print('Starting the data pull label file  %s'%time.asctime())

# Labels are the values we want to predict, these are all the csv input labels
alllabelsold = pd.read_csv('//Users//aakritilakshmanan//Desktop//vole_contour_data//outputmanual.aakriti.csv')
alllabels = pd.read_csv('//Users//aakritilakshmanan//Desktop//vole_contour_data//outputmanualnew...aakriti.csv')
newlabels=pd.read_csv("//Users//aakritilakshmanan//Desktop//vole_contour_data//finalmn.aakriti.csv")
finalLabel = pd.concat([alllabelsold, alllabels])
    
print('Starting the data file')

# features csv input files
featuresdfold = pd.read_csv('//Users//aakritilakshmanan//Desktop//vole_contour_data//output4.aakriti.csv',index_col=0)
featuresdf = pd.read_csv('//Users//aakritilakshmanan//Desktop//vole_contour_data//outputnewdata.aakriti.csv',index_col=0)
areaoverlapfeature= pd.read_csv('//Users//aakritilakshmanan//Desktop//vole_contour_data//outputareaoverlap.aakriti.csv',index_col=0)

finalFeature = featuresdf

#resent the index so that they are the same
newlabels.reset_index(level=0, inplace=True)
newlabels=newlabels.drop(['index'],axis=1)

finalFeature.reset_index(level=0, inplace=True)
finalFeature=finalFeature.drop(['index'],axis=1)

print('We have {} frames of data with {} variables'.format(*finalFeature.shape))
print('We have {} lables of data with {} variables'.format(*newlabels.shape))


#add in additional columns to feature df
finalFeature["CareaOverlap20"]=areaoverlapfeature['C_areaoverlap_10frames']
finalFeature["label"]=newlabels['huddle total']


#filling in the first 20 empty frames
huddleavg=0.0
nonhuddleavg=0.0
for i in range(len(finalFeature['frame num']-1)):
    if finalFeature.loc[i,'label'] ==0:
        nonhuddleavg= nonhuddleavg+ finalFeature.loc[i,'CareaOverlap20']
    else: 
        huddleavg= huddleavg+ finalFeature.loc[i,'CareaOverlap20']

for i in range(len(finalFeature['frame num']-1)):
    if 0<= finalFeature.loc[i,'frame num'] <=20 and finalFeature.loc[i,'label'] ==0:
        finalFeature.loc[i, "CareaOverlap20"] = (nonhuddleavg/(len(finalFeature['frame num']-1)))
    elif 0<= finalFeature.loc[i,'frame num'] <=20 and finalFeature.loc[i,'label'] ==1:
        finalFeature.loc[i, "CareaOverlap20"] = (huddleavg/(len(finalFeature['frame num']-1)))

# dropping columns from features 
finalFeature=finalFeature.drop(['frame num'], axis=1)
finalFeature=finalFeature.drop(['RTogetherFlag'],axis=1)
finalFeature=finalFeature.drop(['LTogetherFlag'],axis=1)

finalFeature=finalFeature.drop(['RcummulativeDist'],axis=1)
finalFeature=finalFeature.drop(['CcummulativeDist'],axis=1)
finalFeature=finalFeature.drop(['LcummulativeDist'],axis=1)


# copy lable and then drop the label column
labels = np.array(finalFeature['label'])
finalFeature=finalFeature.drop(['label'],axis=1)

#check for NAN
print(' Labels non zero:', np.count_nonzero(labels))
print(finalFeature["CareaOverlap20"])


# convert to nparray
features = np.array(finalFeature)

print('We have {} frames of data with {} variables'.format(*finalFeature.shape))
print('We have %s lables of data '%(len(labels)))

print('fnished the data pull label file  %s'%time.asctime())


# In[108]:



#calculate feature importance
feature_importances = pd.DataFrame(rfsmall.feature_importances_,
                                   index = goodfeature_list,
                                   columns=['importance']).sort_values('importance', ascending=False)

#calculate standard deviation for the error bars
std = np.std([tree.feature_importances_ for tree in rfsmall.estimators_],
             axis=0)

#reset index
feature_importances.reset_index(level=0, inplace=True)

#rename/shorten the feature names
feature_importances.iat[6,0] = "Dist from Previous"
feature_importances.iat[0,0] = "Together Flag"
feature_importances.iat[1,0] = "C Area"
feature_importances.iat[2,0] = "C Minor Length"
feature_importances.iat[3,0] = "C Eccentricity"
feature_importances.iat[4,0] = "C Area Overlap"
feature_importances.iat[5,0] = "C Velocity"
feature_importances.iat[7,0] = "C Acceleration"

#plot the figure importance

plt.figure(figsize=(10,7))
# Barplot: Add bars
plt.bar(range(len(goodfeature_list)), feature_importances['importance'], yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10, color='#6D99B4')
# Add feature names as x-axis labels
plt.xticks(range(len(goodfeature_list)), feature_importances['index'], rotation=70, fontsize = 18)
plt.yticks(fontsize=15)
# Create plot title
plt.title("Feature Importance", fontsize=23)
plt.savefig('Feature_Importance.png')
# Show plot
plt.show()


# In[88]:


import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


print('starting data validation %s'%time.asctime())

#set up the df of shortened list of features
onlygoodfeatures =pd.concat( [
    finalFeature['CTogetherFlag'],
    finalFeature['CMinorLength'],
    finalFeature['CdistFromPreviousCont'],
    finalFeature["CareaOverlap20"],
    finalFeature['CSmoothened Acceleration'],
    finalFeature['CSmoothened Velocity'],
    finalFeature['Ceccentricity1'],
    finalFeature['Carea']
    ], axis=1, keys=['together','CMinorLength','C_distFromPreviousCont_avg5','Careaoverlap','Cacceleration','Cvelocity','Ceccentricity2','Carea'])


# List of features for later use
goodfeature_list = list(onlygoodfeatures.columns)

#get the index of the labels for later use when analyzing false n/p
Labels = pd.DataFrame(data=labels)

index=np.array(Labels.index)
index=index.astype(int)

#split data into training/testing modules
train_features2, test_features2, train_labels2, test_labels2,indices_train,indices_test = train_test_split(onlygoodfeatures, labels, index, test_size=0.2, shuffle=True, random_state=0)
train_features2,val_features2, train_labels2, val_labels2 = train_test_split(train_features2,train_labels2, test_size=0.25,shuffle=True, random_state=0)


#set up the random forest
rfsmall =  RandomForestClassifier(n_estimators=200, bootstrap=True, class_weight= None, criterion='gini',
                                max_depth=10, max_features='auto', max_leaf_nodes=None,
                                min_samples_leaf=2,
                                min_samples_split=2, min_weight_fraction_leaf=0.0,
                                n_jobs=1, oob_score=False, random_state=5,
                                verbose=0, warm_start=False)

#oversampling for unbalanced datasets 
#sm = SMOTE()
#X_train_oversampled, y_train_oversampled = sm.fit_sample(train_features2, train_labels2)
    
#fit the model
rfsmall.fit(train_features2, train_labels2);

print ('Finished fitting  model validation %s'%time.asctime())
                                                                                
# Make predictions on test data
predictions = rfsmall.predict(val_features2)

from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(val_labels2, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc*100)

print('done at %s'%time.asctime())


# In[94]:


from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import random


#X_train, X_test, y_train, y_test
#train_features, test_features, train_labels, test_labels 

#split into 5 folds for cv
skf = KFold(n_splits=5, shuffle=True)

X=pd.DataFrame(onlygoodfeatures)
y=pd.DataFrame(labels)


#run cv
for train_index, test_index in skf.split(X,y): 
    print("Train:", train_index, "Validation:", test_index) 
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #shuffle annotations
    #random.shuffle(y_train.values.ravel())
    rfsmall.fit(np.array(X_train),y_train.values.ravel())
    predictions = rfsmall.predict(X_test)
    y_scores = rfsmall.predict_proba(X_test)[:, 1]
    y_pred_adj = adjusted_classes(y_scores, .61)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test.values.ravel(),  y_pred_adj).ravel()
    print(tn)
    print(fp)
    print(fn)
    print(tp)
    print(sklearn.metrics.classification_report(y_test.values.ravel(), y_pred_adj))




# In[93]:


# final statistical tests that can be run for all models 
import sklearn.metrics

def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]

#prediction probabilities for test, val, and train data
y_scores = rfsmall.predict_proba(val_features2)[:, 1]
y_scores2 = rfsmall.predict_proba(test_features2)[:, 1]
y_scores3 = rfsmall.predict_proba(train_features2)[:, 1]

#original predictions on data
predicted_labels = rfsmall.predict(val_features2)
predicted_labels2 = rfsmall.predict(test_features2)
predicted_labels3 = rfsmall.predict(train_features2)

#adjusted predictions with decision threshold of .61
y_pred_adj = adjusted_classes(y_scores, .61)
y_pred_adj2 = adjusted_classes(y_scores2, .61)


#print classification metrics
tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_labels2, y_pred_adj2).ravel()
print(tn)
print(fp)
print(fn)
print(tp)

tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_labels2, predicted_labels2).ravel()
print(tn)
print(fp)
print(fn)
print(tp)

print(sklearn.metrics.classification_report(val_labels2, predicted_labels))
print(sklearn.metrics.classification_report(val_labels2, y_pred_adj))

print(sklearn.metrics.classification_report(test_labels2, predicted_labels2))
print(sklearn.metrics.classification_report(test_labels2, y_pred_adj2))

print(sklearn.metrics.classification_report(train_labels2, predicted_labels3))


# In[95]:


from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt

def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]

def precision_recall_threshold(p, r, thresholds, trainLabels, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_scores, t)
    print(pd.DataFrame(confusion_matrix(trainLabels, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'], 
                       index=['neg', 'pos']))
    
    # plot the curve
    plt.figure(figsize=(8,8))
    plt.title("Precision and Recall curve ^ = current threshold (test)")
    plt.step(r, p, color='b', alpha=0.2,
             where='post')
    plt.fill_between(r, p, step='post', alpha=0.2,
                     color='b')
    plt.ylim([0.5, 1.01]);
    plt.xlim([0.5, 1.01]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    #plt.savefig('precisionrecalltest.png')  
    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k',
            markersize=15)
    


#X_train, X_test, y_train, y_test
#train_features, test_features, train_labels, test_labels 

y_scores = rfsmall.predict_proba(test_features2)[:, 1]

p, r, thresholds = precision_recall_curve(test_labels2, y_scores)

print('done')


# In[109]:



def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(6, 4))
    plt.title("Precision and Recall Scores (Test)", fontsize=19)
    plt.plot(thresholds, precisions[:-1], "--", label="Precision", color="#6D99B4")
    plt.plot(thresholds, recalls[:-1], "-", label="Recall",color="#2d4a65")
    plt.ylabel("Score", fontsize=19)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlabel("Decision Threshold", fontsize=19)
    plt.legend(loc='best', fontsize=15)
    #plt.savefig('prtestfinal.pdf')  
    
plot_precision_recall_vs_threshold(p, r, thresholds)


# In[106]:


def plotPrecision_recall(model,testX,testy,y):
    # predict probabilities
    yhat = model.predict_proba(testX)
    # retrieve just the probabilities for the positive class
    pos_probs = yhat[:, 1]
    # calculate the no skill line as the proportion of the positive class
    no_skill = len(y[y==1]) / len(y)
    # plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill',color="#467096")
    # calculate model precision-recall curve
    precision, recall, _ = precision_recall_curve(testy, pos_probs)
    # plot the model precision-recall curve
    plt.plot(recall, precision, marker='.', label='Logistic', color = '#6D99B4')
    # axis labels
    plt.xlabel('Recall',fontsize=19)
    plt.ylabel('Precision',fontsize=19)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.title("Precision and Recall Scores (test)",fontsize=21)
    # show the legend
    plt.legend(fontsize=16)
    plt.savefig('Precision_recallvalfinal.png')
    # show the plot
    plt.show()

def plotROC(model,testX,testy):
    # predict probabilities
    yhat = model.predict_proba(testX)
    pos_probs = yhat[:, 1]
    # plot no skill roc curve
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill', color="#467096")
    # calculate roc curve for model
    fpr, tpr, _ = roc_curve(testy, pos_probs)
    # plot model roc curve
    plt.plot(fpr, tpr, marker='.', label='Logistic', color = '#6D99B4')
    # axis labels
    plt.xlabel('False Positive Rate',fontsize=19)
    plt.title("ROC Curve (test)",fontsize=21)
    plt.savefig('rocvalfinal.png')
    plt.ylabel('True Positive Rate',fontsize=19)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    # show the legend
    plt.legend(fontsize=16)
    # show the plot
    plt.show()
    
plotPrecision_recall(rfsmall, val_features2, val_labels2,labels)
plotROC(rfsmall, val_features2, val_labels2)


# In[44]:


#ANALYSIS OF FALSE POS AND FALSE NEG

# Dataframe with true values and frame num
true_datadf = pd.concat([featuresdf['frame num'], alllabels['huddle total']], axis=1, keys=['frame', 'behavior'])
true_data = pd.DataFrame(true_datadf)

# Dataframe with predictions and frame num
predictions_data = pd.DataFrame(data = {'frame num': test_features.iloc[:,0], 'prediction': predictions})

#finalLabel.reset_index(level=0, inplace=True)
labelarray = newlabels.to_numpy()
array=labelarray.tolist()
testarray=[]
print('start')
for i in range(0,len(indices_test)):
    testarray.append(labelarray[indices_test[i]])
print('end')


# In[47]:


# false negative positive analysis

#index of the test 
labeltestdf=pd.DataFrame(testarray,columns=['index','random', 'left','center','right', 'interact right','interact left', 'huddle left','attack right','huddle right','huddle total'])

#predictions
predicted_labels = rfsmall.predict(test_features2)
y_scores = rfsmall.predict_proba(test_features2)[:, 1]
y_pred_adj = adjusted_classes(y_scores, .675)
print(sklearn.metrics.classification_report(test_labels2,  y_pred_adj ))

labeltestdf.loc[:,"predictions"] = pd.Series(y_pred_adj, index= labeltestdf.index)

#fact checking, where are they occuring, more indepth than confusion matrix
tp=0.0
tn=0.0
fpi=0.0
fpnt=0.0
fn=0.0
fpnon=0.0

print('start')
for i in range(0,len(labeltestdf)):
    if labeltestdf.loc[i,'predictions'] ==1 and labeltestdf.loc[i,'huddle total'] ==1:
        tp=tp+1
    elif labeltestdf.loc[i,'predictions'] ==0 and labeltestdf.loc[i,'huddle total'] ==0:
        tn=tn+1
    elif labeltestdf.loc[i,'predictions'] ==0 and labeltestdf.loc[i,'huddle total'] ==1:
        fn=fn+1
    elif labeltestdf.loc[i,'predictions'] ==1 and labeltestdf.loc[i,'huddle total'] ==0:
        if labeltestdf.loc[i,'interact right'] ==1 or labeltestdf.loc[i,'interact left'] ==1:
            fpi = fpi+1
        elif labeltestdf.loc[i,'center'] ==1 or labeltestdf.loc[i,'left'] ==1 or labeltestdf.loc[i,'right'] ==1:
            fpnt = fpi+1
        else:
            fpnon=fpnon+1

print(tp)
print(tn)
print(fpi)
print(fpnt)
print(fn)
print(fpnon)
        
#read to csv file
labeltestdf.to_csv("//Users//aakritilakshmanan//Desktop//vole_contour_data//falseposneg.aakriti.csv" , sep=',')

print('done')


# In[138]:


# RUN GRID SEARCH FOR FINDING BEST HYPERPARAMTERS

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 60, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 2, verbose=2, random_state=42, n_jobs = 2)
# Fit the random search model
rf_random.fit(train_features2, train_labels2)


# In[139]:


rf_random.best_params_


# In[8]:


# graph the actual behaviors against the predicted behaviors

import pandas as pd
import matplotlib.pyplot as plt
alllabelstestf = pd.read_csv("//Users//aakritilakshmanan//Desktop//vole_contour_data//falseposneg.aakriti.csv" )

alllabelstestf.reset_index(level=0, inplace=True)
alllabelstestf=alllabelstestf.drop(['index'],axis=1)

#print(alllabelstestf)

x=range(0,len(alllabelstestf))

print('start')
f, (ax1, ax2, ax3, ax4, ax5,ax6,ax7,ax8,ax9) = plt.subplots(9, sharex=True, sharey=False)
ax1.bar(x, alllabelstestf.iloc[:,3])
ax1.set_ylabel(r'Left', size =10)
ax1.set_title('Features')
ax2.bar(x, alllabelstestf.iloc[:,4], color = 'g')
ax2.set_ylabel(r'Center', size =10)

ax3.bar(x, alllabelstestf.iloc[:,5], color='r')
ax3.set_ylabel(r'Right', size =10)

ax4.bar(x, alllabelstestf.iloc[:,6], color='r')
ax4.set_ylabel(r'Interact Right', size =10)

ax5.bar(x, alllabelstestf.iloc[:,7], color='r')
ax5.set_ylabel(r'Int Left', size =10)

ax6.bar(x, alllabelstestf.iloc[:,8], color='r')
ax6.set_ylabel(r'Huddle Left', size =10)

ax7.bar(x,  alllabelstestf.iloc[:,10])
ax7.set_ylabel(r'Huddle Right', size =10)

ax8.bar(x,  alllabelstestf.iloc[:,11])
ax8.set_ylabel(r'Predicted Huddles', size =10)

ax9.bar(x,  alllabelstestf.iloc[:,12])
ax9.set_ylabel(r'Predicted Huddles', size =10)



f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
ax1.set_axis_off
plt.setp([a.get_yticklabels() for a in f.axes[:4]], visible=False)

print('end')


# In[ ]:




