#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator,RegressorMixin
import spacy
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    import spacy.cli 
    print("Model not found. Downloading.")
    spacy.cli.download("en_core_web_md")
    import en_core_web_md
    nlp = en_core_web_md.load()
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import sklearn   
import scipy.io as scio
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report



def explore(df):
    print('The shape of the dataset is:{shape}'.format(shape =df.shape))
    print('\n')
    df.info()
    print('\n')
    print('The number of null value of the dataset is:')
    print(df.isna().sum())
    print('\n')
    return df.head()


def drop_column(df,column):
    df = df.drop(column, axis = 1)
    return df


def lowercase(df,column):
    df[column] = df[column].str.lower()


def date_time_split(df,column):
    df[column+'_date'] = pd.to_datetime(df[column]).dt.date

def check_equal(a, b):
    if a == b:
        return 1
    else:
        return b-a

def conflict_correct(df,des,column,a):
    for x in des:
        df.loc[df['description']==x,column] = a
        
def rule(df,i,y):
    df['global'+str(i)] = [i if x == True else 0 for x in df.description.str.contains(y, regex= True, na=False)]
    
def conf_des_gen(df,column,category):
    return df.description[df[column]==category].to_list()    

def rule_gen(i):
    rule_i = global_rule[global_rule['standard_transaction_type_id']==i]
    rule_i_str = "|".join(rule_i.regex_pattern)
    return 'r' + rule_i_str


#Text-preprocessing

#remove URLs
def remove_url(df,column):
    return df[column].str.replace(r'(www|\bhttp|\b)\S+(.co.uk|.com)',' ',regex = True)

#remove punctuations
def remove_punc(df,column):
    return df[column].str.replace(r"[\-*:()@<>#&/,.?_]",' ',regex = True)

#remove numbers
def remove_num(df,column):
    return df[column].str.replace(r'\d+(.\d+)?',' ',regex = True)

#remove stop words
def remove_stopword(df,stop_df,column):
    stop_df['stopwords'] = r'\b' + stop_df['stopwords'] + r'\b'
    regex = '|'.join(stop_df['stopwords'])
    return df[column].str.replace(regex, ' ', regex=True)

def removals(df,stop_df,column):
    df[column] = remove_url(df,column)
    df[column] = remove_num(df,column)
    df[column] = remove_stopword(df,stop_df,column)
    df[column] = remove_punc(df,column)
    df[column] = df[column].str.strip()
    return 'finished'

def abbrivations(df,column):
    df[column] = df[column].str.replace(r'\b(xfer|tfr)\b','transfer',regex = True)
    df[column] = df[column].str.replace(r'\b(pymt|payt|pyt)\b','payment',regex = True)
    df[column] = df[column].str.replace(r'\b(st|stre)\b','street',regex = True)
    df[column] = df[column].str.replace(r'\b(dd|ddr)\b','direct debit',regex = True)
    df[column] = df[column].str.replace(r'\b(bbp|bp)\b','bill payment',regex = True)
    df[column] = df[column].str.replace(r'\bbgc\b','bank giro credit',regex = True)
    df[column] = df[column].str.replace(r'\bbdc\b','bank debit card',regex = True)
    df[column] = df[column].str.replace(r'\bbac\b','banker automated clearing service',regex = True)
    df[column] = df[column].str.replace(r'\bsto\b','standing order',regex = True)
    df[column] = df[column].str.replace(r'\brp\b','repurchase agreement',regex = True)
    df[column] = df[column].str.replace(r'\bcsc\b','common sense compliance',regex = True)
    df[column] = df[column].str.replace(r'\bcc\b','cash credit',regex = True)
    df[column] = df[column].str.replace(r'\bdpc\b','direct banking',regex = True)
    df[column] = df[column].str.replace(r'\bexp\b','expense',regex = True)
    df[column] = df[column].str.replace(r"\bint'l\b",'international',regex = True)
    df[column] = df[column].str.replace(r'\bfp\b','faster payment',regex = True)
    df[column] = df[column].str.replace(r'\binc\b','incorporated',regex = True)
    df[column] = df[column].str.replace(r'\bdeb\b','debit card',regex = True)
    df[column] = df[column].str.replace(r'\b(\scd)+(\sdebit card|)$\b','card transaction',regex = True)
    df[column] = df[column].str.replace(r'\bre(pymt|pyt|payt)\b','repayment',regex = True)


#tokenisation
def tokenise_withfilter(token,min_length=1):
    return not (token.is_punct | token.is_space) and (len(token.text) > min_length)

def tokenise_lemmatise(df,column):
    texts = df[column].tolist()
    lemmatise = []
    for text in nlp.pipe(texts):
        tokens = [token.lemma_ for token in text if tokenise_withfilter(token)]
        lemmatise.append(tokens)
    return lemmatise



#Models_semi_supervised

#Tri-training

class TriTraining:
    def __init__(self, classifier):
        if sklearn.base.is_classifier(classifier):
            self.classifiers = [sklearn.base.clone(classifier) for i in range(3)]
        else:
            self.classifiers = [sklearn.base.clone(classifier[i]) for i in range(3)]
            
    def fit(self, L_X, L_y, U_X):
        for i in range(3):
            sample = sklearn.utils.resample(L_X, L_y)  
            self.classifiers[i].fit(*sample)   
        e_prime = [0.5]*3
        l_prime = [0]*3
        e = [0]*3
        update = [False]*3
        Li_X, Li_y = [[]]*3, [[]]*3#to save proxy labeled data
        improve = True
        self.iter = 0
        
        while improve:
            self.iter += 1#count iterations 
            
            for i in range(3):    
                j, k = np.delete(np.array([0,1,2]),i)
                update[i] = False
                e[i] = self.measure_error(L_X, L_y, j, k)
                if e[i] < e_prime[i]:
                    U_y_j = self.classifiers[j].predict(U_X)
                    U_y_k = self.classifiers[k].predict(U_X)
                    Li_X[i] = U_X[U_y_j == U_y_k]#when two models agree on the label, save it
                    Li_y[i] = U_y_j[U_y_j == U_y_k]
                    if l_prime[i] == 0:#no updated before
                        l_prime[i]  = int(e[i]/(e_prime[i] - e[i]) + 1)
                    if l_prime[i] < len(Li_y[i]):
                        if e[i]*len(Li_y[i])<e_prime[i] * l_prime[i]:
                            update[i] = True
                        elif l_prime[i] > e[i]/(e_prime[i] - e[i]):
                            L_index = np.random.choice(len(Li_y[i]), int(e_prime[i] * l_prime[i]/e[i] -1))
                            Li_X[i], Li_y[i] = Li_X[i][L_index], Li_y[i][L_index]
                            update[i] = True
            for i in range(3):
                if update[i]:
                    a = np.append(L_X,Li_X[i],axis=0)
                    b = np.append(L_y, Li_y[i], axis=0)
                    self.classifiers[i].fit(np.append(L_X,Li_X[i],axis=0), np.append(L_y, Li_y[i], axis=0))
                    e_prime[i] = e[i]
                    l_prime[i] = len(Li_y[i])
    
            if update == [False]*3:
                improve = False#if no classifier was updated, no improvement


    def predict(self, X):
        pred = np.asarray([self.classifiers[i].predict(X) for i in range(3)])
        pred[0][pred[1]==pred[2]] = pred[1][pred[1]==pred[2]]
        return pred[0]
        
    def score(self, X, y):
        return sklearn.metrics.accuracy_score(y, self.predict(X))
        
    def measure_error(self, X, y, j, k):
        j_pred = self.classifiers[j].predict(X)
        k_pred = self.classifiers[k].predict(X)
        wrong_index =np.logical_and(j_pred != y, k_pred==j_pred)
        #wrong_index =np.logical_and(j_pred != y_test, k_pred!=y_test)
        return sum(wrong_index)/sum(j_pred == k_pred)


# Pseudo labelling

class PseudoLabeler(BaseEstimator, RegressorMixin):
    '''
     Sci-kit learn wrapper for creating pseudo-lebeled estimators.
     '''

    def __init__(self, model, unlabled_data, features, target, sample_rate=0.2, seed=42):
        '''
     @sample_rate - percent of samples used as pseudo-labelled data
     from the unlabelled dataset
     '''
        assert(sample_rate <= 1.0)
        self.sample_rate = sample_rate
        self.seed = seed
        self.model = model
        self.model.seed = seed

        self.unlabled_data = unlabled_data
        self.features = features
        self.target = target

    def get_params(self, deep=True):
         return {
     "sample_rate": self.sample_rate,
     "seed": self.seed,
     "model": self.model,
     "unlabled_data": self.unlabled_data,
     "features": self.features,
     "target": self.target
     }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
                setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        '''
     Fit the data using pseudo labeling.
     '''

        augemented_train = self.__create_augmented_train(X, y)
        self.model.fit(augemented_train[self.features],augemented_train[self.target])
        return self

    def __create_augmented_train(self, X, y):
        '''
     Create and return the augmented_train set that consists
     of pseudo-labeled and labeled data.
     '''
        num_of_samples = int(len(self.unlabled_data) * self.sample_rate)

    # Train the model and creat the pseudo-labels
        self.model.fit(X, y)
        pseudo_labels = self.model.predict(self.unlabled_data[self.features])

    # Add the pseudo-labels to the test set
        pseudo_data = self.unlabled_data.copy(deep=True)
        pseudo_data[self.target] = pseudo_labels

    # Take a subset of the test set with pseudo-labels and append in onto
     # the training set
        sampled_pseudo_data = pseudo_data.sample(n=num_of_samples)
        temp_train = pd.concat([X, y], axis=1)
        augemented_train = pd.concat([sampled_pseudo_data, temp_train])

        return shuffle(augemented_train)

    def predict(self, X):
        '''
     Returns the predicted values.
     '''
        return self.model.predict(X)

    def get_model_name(self):
        return self.model.__class__.__name__


#Model hyperparameter tuning

def gridsearch(model,param):
    kfold = StratifiedKFold(n_splits=5,random_state=200,shuffle=True)
    np.random.seed(0)
    clf=GridSearchCV(model,param, cv=kfold, n_jobs=-1,scoring='f1_macro')
    clf.fit(X_train, y_train)
    return clf.best_params_



# Model evluation

def confusion(predict,true,name):
    cm = confusion_matrix(true, predict)

    print("{}".format(name))
    print(classification_report(true, predict,digits=4))
    print("{} Confusion matrix".format(name))
    print(cm)
    cm_df = pd.DataFrame(cm)
    cm_df.index =  range(1,17)
    cm_df.columns =  range(1,17)
    plt.figure(figsize = (10,10))
    plt.title('Confusion Matrix Plot',fontsize=15)
    sns.heatmap(cm_df, annot=True, fmt='g',cmap=plt.cm.Blues_r)
    plt.xlabel('Predicted Labels',fontsize=14)
    plt.ylabel('True Labels',fontsize=14)
    plt.show()


def confusion_final(predict,true):
    cm = confusion_matrix(true, predict)

    print(classification_report(true, predict,digits=5))
    print("Confusion matrix")
    print(cm)
    cm_df = pd.DataFrame(cm)
    cm_df.index =  range(0,17)
    cm_df.columns =  range(0,17)
    plt.figure(figsize = (10,10))
    plt.title('Confusion Matrix Plot',fontsize=15)
    sns.heatmap(cm_df, annot=True, fmt='g',cmap=plt.cm.Blues_r)
    plt.xlabel('Predicted Labels',fontsize=14)
    plt.ylabel('True Labels',fontsize=14)
    plt.show()

def model_evaluation(train,test,finalmodel):
    finalmodel.fit(train,y_train)
    predict_test = finalmodel.predict(test)
    confusion(predict_test,y_test)


