#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
pd.set_option('max_colwidth', 1000)
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate,StratifiedKFold,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import sklearn   
import os
import scipy.io as scio
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator,RegressorMixin


#Create another column in order to to modify the labels 
def gen_rule_label_column(df,label_column):
    df[label_column] = df['standard_transaction_type_id']

#match local rules with given date batch and label the others as 0
def partial_rule_labelling(df,date,label_column,value):
    df.loc[df['local_rule_creation_date'] >= date, label_column] = value
    df.loc[df['local_rule_creation_date'].isna(),label_column] = value


def semi_supervised_set(df,date,label_column,value):
    gen_rule_label_column(df,label_column)
    partial_rule_labelling(df,date,label_column,value)
    df_semi = df[(df['global']!=0)&(df[label_column]==-1)|(df[label_column]!=-1)]
    return df_semi
   
    
#Checking n-grams in frequency-based models
svc = SVC(random_state=200)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=200)

def vect_selection(df_des,y):
    ngram = list()
    for n in range(1,6):
        bow =  CountVectorizer(ngram_range=(1,n))
        tfidf=  TfidfVectorizer(ngram_range=(1,n))
        i = (1,n)
            
        tfidf_matrix = tfidf.fit_transform(df_des)
        bow_matrix = bow.fit_transform(df_des)
        tfidf_cv = cross_val_score(svc, tfidf_matrix, y, cv=kfold, scoring='f1_macro').mean()
        bow_cv = cross_val_score(svc, bow_matrix, y, cv=kfold, scoring='f1_macro').mean()
        ngram.append({'n': i, 'score': tfidf_cv, 'type': 'TF-iDF','model':TfidfVectorizer})
        ngram.append({'n': i, 'score': bow_cv, 'type': 'Bag of Words','model':CountVectorizer})
    
    ngram_df = pd.DataFrame(ngram)
    max_score = ngram_df['score'].max()
    a = ngram_df[ngram_df['score'] == max_score]
    optimal_ngram = a.iloc[0]['n']
    optimal_model_name = a.iloc[0]['type']
    optimal_score = a.iloc[0]['score']
    optimal_model = a.iloc[0]['model']
    return optimal_model,optimal_ngram


#Selecting maximum features for the better performed model
def feature_num_selection(df_des,y,model,index):
    vec =  model(ngram_range=index)
    vec_matrix = vec.fit_transform(df_des)
    a = vec_matrix.shape
    
    vec_result = list()
    for n in range(100, a[1], 100):
        vec = model(ngram_range=index,max_features=n)
        vec_matrix = vec.fit_transform(df_des)
        score_countVec = cross_val_score(svc, vec_matrix, y, cv=kfold, scoring='f1_macro').mean()
        vec_result.append({'max_feature': n,  'score': score_countVec})
    
    vec_result_df = pd.DataFrame(vec_result)
    max_score = vec_result_df['score'].max()
    a = vec_result_df[vec_result_df['score'] == max_score]
    max_feat = a.iloc[0]['max_feature']
    return max_feat


def final_model_vectoriser(df_des,model,indices):
    des_trans = model.transform(df_des)
    des_df = des_trans.toarray()
    des_df = pd.DataFrame(des_df)
    des_df.index = indices
    return des_df


from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
def form_final_df(des_df,df):
    #Normalise amount and balance between 0 and 1
    df['amount_std'] = scaler.fit_transform(df['amount'].values.reshape(-1,1))
    df['balance_std'] = scaler.fit_transform(df['balance'].values.reshape(-1,1))
    X = pd.concat([des_df,df[['amount_std','balance_std','debit_binary', 'foreign_binary','financing_binary','account_binary','hmrc_vat','hmrc_corp','hmrc_paye','week_of_month_2','week_of_month_3','week_of_month_4','token_length']]], axis=1)
    return X



def semi_df_for_training(df_semi,label_column):
    df_semi_labeled = df_semi[df_semi[label_column]!=-1]
    df_semi_unlabeled = df_semi[df_semi[label_column]==-1]
    
    X_label_des = [' '.join(x) for x in df_semi_labeled['description']]
    X_unlabel_des = [' '.join(x) for x in df_semi_unlabeled['description']]
    X_train_des = [' '.join(x) for x in df_semi['description']]
    y_labeled = df_semi_labeled[label_column]
    y_training = df_semi[label_column]
    training_indices = df_semi.index
    labeled_indices = df_semi_labeled.index
    unlabeled_indices = df_semi_unlabeled.index
    
    #Word representation selection
    vector_model,n_gram_index = vect_selection(X_label_des,y_labeled)
    max_feat = feature_num_selection(X_label_des,y_labeled,vector_model,n_gram_index)

    #Use the final model selected to form word representations of training and unlabelled data descriptions    
    final_vec_model= vector_model(max_features=int(max_feat),ngram_range=n_gram_index)
    final_vec_model.fit_transform(X_label_des)
    text_df_training = final_model_vectoriser(X_train_des,final_vec_model,training_indices)
    text_df_labeled = final_model_vectoriser(X_label_des,final_vec_model,labeled_indices)
    text_df_unlabeled = final_model_vectoriser(X_unlabel_des,final_vec_model,unlabeled_indices)
    
    X_labelled = form_final_df(text_df_labeled,df_semi_labeled)
    X_unlabelled = form_final_df(text_df_unlabeled,df_semi_unlabeled)
    X_training = form_final_df(text_df_training,df_semi)
    
    return X_labelled,X_unlabelled,X_training,y_labeled, y_training,df_semi_labeled,df_semi_unlabeled



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
        
        
        
def semi_modelling(X_train,X_l,X_u,y_l,y_train,label_column):
    lp_model = LabelPropagation(gamma=1, kernel='rbf', max_iter=100000, n_jobs=-1,n_neighbors=7, tol=0.001)
    tt = TriTraining([ExtraTreesClassifier(max_depth= None,max_features='sqrt',min_samples_leaf= 1,min_samples_split= 10,n_estimators= 100,random_state=200),
                  RandomForestClassifier(max_depth= 50,max_features= 'sqrt',min_samples_leaf=1,min_samples_split= 2,n_estimators= 100),
                  XGBClassifier(max_depth= 50, n_estimators= 50,random_state=200)])
    pseudo = PseudoLabeler(ExtraTreesClassifier(max_depth= None,max_features='log2',min_samples_leaf= 1,min_samples_split= 10,n_estimators= 100,random_state=200),
    X_u,X_u.columns,label_column,sample_rate=0.3)
    
    lp_model.fit(X_train.values,y_train)
    tt.fit(X_l.values,y_l.values,X_u.values)
    pseudo.seed = 42
    pseudo.fit(X_l, y_l)
    
    lp_predict = lp_model.predict(X_u.values)
    tt_predict = tt.predict(X_u.values)
    pse_predict = pseudo.predict(X_u)
    prediction_combine = np.vstack((lp_predict, tt_predict,pse_predict)).T
    return prediction_combine
    
    
    
def semi_vote(prediction,df_semi_unlab,label_column,df):
    vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=prediction)
    vote_df = pd.DataFrame(vote)
    vote_df.index = df_semi_unlab.index

    a = pd.DataFrame(prediction)
    a.index = df_semi_unlab.index
    non_vote_index = a[(a.iloc[:,0]!=a.iloc[:,1])&(a.iloc[:,2]!=a.iloc[:,1])&(a.iloc[:,0]!=a.iloc[:,2])].index

    global_non_vote = df_semi_unlab[df_semi_unlab.index.isin(non_vote_index)]
    global_non_vote = global_non_vote['global']

    for index in non_vote_index:
        vote_df.loc[index] = global_non_vote[index]

    df['weak_label'] = df[label_column]
    for index in df_semi_unlab.index:
        df.loc[index,'weak_label'] = vote_df.loc[index].values[0]
   
    df = df.drop(label_column, axis = 1)
    return df

        
        
        
        

