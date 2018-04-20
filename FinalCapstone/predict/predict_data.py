import configparser
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools
from Entity import Word_Analysis
from predict.src.make_feature import *
from predict.src.train_utils import *
from itertools import zip_longest
import pickle
import heapq
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os


def getIndexValue(pred_values, checkAllZeros = None):
    result = []
    if checkAllZeros == True:

        indexArr = pred_values[0]
        for i in range(len(pred_values[0])):
            result.append(i+1)
    else:
        indexArr = heapq.nlargest(3,pred_values[0])
        for i in range(len(indexArr)):
            for j in range(len(pred_values[0])):
                if indexArr[i] == pred_values[0][j]:
                    result.append(j+1)

    return result

def define_sparseMatix(feature_list, max_Feature):

    tmp = []
    temp = []
    feature_list = dict(feature_list)
    print(feature_list)
    tmp.append(feature_list)
    max_column = max_Feature
    n_feature = max_column
    sparse = []
    for index in range(len(tmp)):
        temp = []
        for counter in range(1, n_feature + 1):
            temp.append(tmp[index].get(counter, 0))
        sparse.append(temp)
    return sparse

def platt_method(decision_value):
    #proba = (1. / (1. + np.exp(-clf.decision_function(np.array(x_test).reshape(1, -1)))))
    proba = (1. / (1. + np.exp(-decision_value.reshape(1,-1))))
    proba /= proba.sum(axis=1).reshape((proba.shape[0], -1))

    return proba

def make_Word_Anlysis(name,proba):
    result = Word_Analysis(name,proba)
    return result

def tone_predict(i, syllables, model_dir, file_path, window_size):
    target_syllable = syllables[i]
    target_syllable_lower = syllables[i].lower()
    target_syllable_lower = "b'" + target_syllable_lower + "'"
    print(target_syllable_lower)
    if target_syllable_lower not in file_path:
        return u'[{}]'.format(target_syllable), None
    model_path = os.path.join(model_dir, '{}.pkl'.format(target_syllable_lower))
    features = get_feature(target_syllable, i, syllables, window_size)
    mapped_feature, max_Feature = feature_mapping(model_dir, target_syllable_lower, features)
    list_train_data = train_data_(model_dir,target_syllable_lower)
    list_label_data = lable_data(model_dir, target_syllable_lower)

    sparseMatrix = define_sparseMatix(mapped_feature, max_Feature=max_Feature)
    # estimate
    # loading a model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("Decision Value:", model.decision_function(sparseMatrix))
    tmp_result = model.decision_function(sparseMatrix)
    proba = platt_method(tmp_result)
    #print("Proba:", proba)
    indexArr = []
    pred_values = []
    if isinstance(tmp_result[0], np.float64):
        pred_values.append(tmp_result)
    else:
        pred_values = tmp_result
    if any(x > 0 for x in pred_values[0].tolist()):
        indexArr = getIndexValue(pred_values)
        #print("Index Arr:", indexArr)
    else:
        if all(x == 0 for x in pred_values[0]):
            indexArr = getIndexValue(pred_values, checkAllZeros=True)
        else:
            for index in range(len(pred_values)):
                for i in range(len(pred_values[index])):
                    indexArr = getIndexValue(pred_values)
        #print("Index Arr:", indexArr)
    print("Index Arr:", indexArr)
    
    result, label = class_mapping(model_dir, target_syllable_lower,  indexArr)
    infor = Word_Analysis.Word_Analysis(target_syllable,proba,result,list_train_data,list_label_data,tmp_result,label)
    return result, infor

def getResult(resultArr):
    result = []
    s = ""
    for i in range(len(resultArr)):

        resultArr[i] = list(resultArr[i])
        for j in range(len(resultArr[i])):

            if resultArr[i][j] == None:
                if i == 0:
                    resultArr[i][j] = resultArr[i+1][j]
                else:
                    resultArr[i][j] = resultArr[i-1][j]
            s = s + resultArr[i][j] + " "
        print(s)
        s = ""
    return



def view_Analysis(result_Object_Word):
    for index in range(len(result_Object_Word)):
        word = result_Object_Word[index].syllable_name
        print("Word:", result_Object_Word[index].syllable_name)
        print("Prediction Value:", result_Object_Word[index].prediction_values)
        print("All Label Word:", result_Object_Word[index].all_label)
        print("probability:", result_Object_Word[index].proba)
        print("Predict List:", result_Object_Word[index].result)

        x_train = result_Object_Word[index].train_data

        y_train = result_Object_Word[index].label_data
       
        tmp = []
        for index in range(len(x_train)):
            tmp.append(max(x_train[index].keys()))
        max_column = max(tmp)
        n_feature = max_column
        sparse = []
        for index in range(len(x_train)):
            temp = []
            for counter in range(1, n_feature + 1):
                temp.append(x_train[index].get(counter, 0))
            sparse.append(temp)

        pca2 = PCA(n_components=2).fit(sparse)
        pca_2d = pca2.transform(sparse)
        #print(pca_2d)
        #print(len(pca_2d[0]))
        if(len(pca_2d) != 1):
            X, y = pca_2d, y_train[0]
            labels = ['SVM Classification']
            if len(y) >=2 and max(y) > 1:
                try:
                    parameters = {'C': [1, 10, 100,200,300]}
                    clf = GridSearchCV(svm.LinearSVC(), parameters)
                    clf.fit(X, y)
                    result_C = clf.best_params_.get('C')
                    if result_C:
                        print("Optimal C:", result_C)
                        print("===================================================================")
                        clf = svm.LinearSVC(C= result_C)
                        X = X[:, [0, 1]]
                        clf.fit(X,y)
                        gs = gridspec.GridSpec(2, 2)
                        fig = plt.figure(figsize=(10, 8))
                except:
                    clf = svm.LinearSVC(C= 1000)
                    X, y = pca_2d, y_train[0]
                    X = X[:, [0, 1]]
                    clf.fit(pca_2d,y)
                    gs = gridspec.GridSpec(2, 2)
                    fig = plt.figure(figsize=(10, 8))
                    clf.fit(X, y)
            elif len(y) >= 2 and max(y) == 1:
                clf = svm.OneClassSVM(kernel='linear', nu=0.06)
                X = X[:, [0, 1]]
                gs = gridspec.GridSpec(2, 2)
                fig = plt.figure(figsize=(10, 8))
                clf.fit(X)
            elif len(y) == 1 and max(y) == 1:
                clf = svm.OneClassSVM(kernel='linear', nu=0.06)
                gs = gridspec.GridSpec(2, 2)
                fig = plt.figure(figsize=(10, 8))
                clf.fit(X)
            for clf, lab, grd in zip([clf], labels, itertools.product([0, 1], repeat=2)):
                # clf.fit(X)
                ax = plt.subplot(gs[grd[0], grd[1]])
                fig = plot_decision_regions(X=X, y=np.array(y).astype(np.integer), clf=clf, legend=2)
                plt.title(lab)
            #save_file = os.path.join("/Users/chinhnguyen/Desktop/TestCapstone/", word +".png")
            #plt.savefig(save_file)
                plt.show()

def main():
    inifile = configparser.SafeConfigParser()
    inifile.read("D:/capstone/FinalCapstone/config.ini")
    model_dir = inifile.get("settings", "model_dir")
    file_path = set([p.split('.')[0] for p in os.listdir(model_dir)])
    window_size = 2
    for line in open('D:/capstone/FinalCapstone/predict/input.txt', 'r'):
        syllables = line.rstrip().split(u' ')
        # predicted_syllables = [tone_predict(i, syllables, model_dir, file_path, window_size) for i in
        #                        range(len(syllables))]
        # print("Result: ", predicted_syllables)
        predicted_syllables = []
        result_Object_Word = []
        for i in range(len(syllables)):
            predicted_syllable, word_Analysis = tone_predict(i,syllables,model_dir,file_path,window_size)
            predicted_syllables.append(predicted_syllable)
            if word_Analysis != None:
                result_Object_Word.append(word_Analysis)

        print()
        print()
        print("=========================View Analysis===========================")

        view_Analysis(result_Object_Word)


        print("===================================================================")
        print()
        print()
        print("==========================View Result==============================")
        for i in range(len(predicted_syllables)):
            if type(predicted_syllables[i]) is list:
                if len(predicted_syllables[i]) == 1:
                    predicted_syllables[i].append(predicted_syllables[i][0])
                    predicted_syllables[i].append(predicted_syllables[i][0])
                if len(predicted_syllables[i]) == 2:
                    predicted_syllables[i].append(predicted_syllables[i][0])
            else:
                tmp = []
                tmp.append(predicted_syllables[i])
                tmp.append(predicted_syllables[i])
                tmp.append(predicted_syllables[i])
                predicted_syllables[i] = tmp
        print("After: ", predicted_syllables)
        tempResult = list(zip_longest(*predicted_syllables))
        getResult(tempResult)
        print("===================================================================")


if __name__ == "__main__":
    main()