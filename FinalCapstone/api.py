# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 11:18:07 2018

@author: HuyBTSE62022a
"""

from flask import Flask, request, Response
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
api = Api(app)
cors = CORS(app, resources={r"/predict/*": {"origins": "*"}})
import os
import sys
import warnings
import configparser
import matplotlib.pyplot as plt
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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import json

print('scipy: %s' % json.__version__)

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def getIndexValue(pred_values, checkAllZeros=None):
    result = []
    if checkAllZeros == True:

        indexArr = pred_values[0]
        for i in range(len(pred_values[0])):
            result.append(i + 1)
    else:
        indexArr = heapq.nlargest(3, pred_values[0])
        for i in range(len(indexArr)):
            for j in range(len(pred_values[0])):
                if indexArr[i] == pred_values[0][j]:
                    result.append(j + 1)

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
    # proba = (1. / (1. + np.exp(-clf.decision_function(np.array(x_test).reshape(1, -1)))))
    proba = (1. / (1. + np.exp(-decision_value.reshape(1, -1))))
    proba /= proba.sum(axis=1).reshape((proba.shape[0], -1))

    return proba


def make_Word_Anlysis(name, proba):
    result = Word_Analysis(name, proba)
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
    sparseMatrix = define_sparseMatix(mapped_feature, max_Feature=max_Feature)
    # estimate
    # loading a model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("Decision Value:", model.decision_function(sparseMatrix))
    tmp_result = model.decision_function(sparseMatrix)
    proba = platt_method(tmp_result)
    # print("Proba:", proba)
    indexArr = []
    pred_values = []
    if isinstance(tmp_result[0], np.float64):
        pred_values.append(tmp_result)
    else:
        pred_values = tmp_result
    if any(x > 0 for x in pred_values[0].tolist()):
        indexArr = getIndexValue(pred_values)
        # print("Index Arr:", indexArr)
    else:
        if all(x == 0 for x in pred_values[0]):
            indexArr = getIndexValue(pred_values, checkAllZeros=True)
        else:
            for index in range(len(pred_values)):
                for i in range(len(pred_values[index])):
                    indexArr = getIndexValue(pred_values)
        # print("Index Arr:", indexArr)
    print("Index Arr:", indexArr)

    result, label = class_mapping(model_dir, target_syllable_lower, indexArr)
    infor = Word_Analysis.Word_Analysis(target_syllable, proba, result, tmp_result, label)
    return result, infor


def getResult(resultArr):
    result = []
    rs = []
    s = ""
    for i in range(len(resultArr)):

        resultArr[i] = list(resultArr[i])
        for j in range(len(resultArr[i])):

            if resultArr[i][j] == None:
                if i == 0:
                    resultArr[i][j] = resultArr[i + 1][j]
                else:
                    resultArr[i][j] = resultArr[i - 1][j]
            s = s + resultArr[i][j] + " "
        rs.append(s)
        s = ""

    return rs


def view_Analysis(result_Object_Word):
    listWord = []
    listAllLabel = []
    listProba = []
    listPredict = []
    listPredictionValue = []

    for index in range(len(result_Object_Word)):
        print("Word:", result_Object_Word[index].syllable_name)
        print("Prediction Value:", result_Object_Word[index].prediction_values)
        print("All Label Word:", result_Object_Word[index].all_label)
        print("probability:", result_Object_Word[index].proba)
        print("Predict List:", result_Object_Word[index].result)
        listWord.append(result_Object_Word[index].syllable_name)
        listAllLabel.append(result_Object_Word[index].all_label)
        listProba.append(result_Object_Word[index].proba)
        listPredict.append(result_Object_Word[index].result)
        listPredictionValue.append(result_Object_Word[index].prediction_values)

    return listWord, listProba, listPredict, listAllLabel, listPredictionValue


class Predict(Resource):
    def get(self, text):

        text = str(text)
        inifile = configparser.SafeConfigParser()
        inifile.read("/Users/nguyenvulong/Documents/CapstonePro/FinalCapstone/config.ini")
        model_dir = inifile.get("settings", "model_dir")
        file_path = set([p.split('.')[0] for p in os.listdir(model_dir)])
        window_size = 2

        syllables = text.rstrip().split(u' ')

        predicted_syllables = []
        result_Object_Word = []
        for i in range(len(syllables)):
            predicted_syllable, word_Analysis = tone_predict(i, syllables, model_dir, file_path, window_size)
            predicted_syllables.append(predicted_syllable)
            if word_Analysis != None:
                result_Object_Word.append(word_Analysis)
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

        rs = getResult(tempResult)

        result = {
            'output1': u''.join(rs[0]),
            'output2': u''.join(rs[1]),
            'output3': u''.join(rs[2])
        }

        return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json')


class Analysis(Resource):
    def get(self, text):

        text = str(text)
        inifile = configparser.SafeConfigParser()
        inifile.read("/Users/nguyenvulong/Documents/CapstonePro/FinalCapstone/config.ini")
        model_dir = inifile.get("settings", "model_dir")
        file_path = set([p.split('.')[0] for p in os.listdir(model_dir)])
        window_size = 2

        syllables = text.rstrip().split(u' ')

        predicted_syllables = []
        result_Object_Word = []
        for i in range(len(syllables)):
            predicted_syllable, word_Analysis = tone_predict(i, syllables, model_dir, file_path, window_size)
            predicted_syllables.append(predicted_syllable)
            if word_Analysis != None:
                result_Object_Word.append(word_Analysis)
        listWord, listProba, listPredict, listAllLabel, listPredictionValue = view_Analysis(result_Object_Word)
        listObject = []
        for i in range(0, len(listWord)):

            try:
                classification = open(model_dir + '/' + listWord[i] + '_classification.txt', 'r', encoding="utf8")
                listClassification = []
                for line in classification:
                    listClassification.append(line)
                result = {
                    'word': u''.join(listWord[i]),
                    'prediction_value': ''.join(str(p) for p in listPredictionValue[i]),
                    'all_Lable': ' '.join(str(p) for p in listAllLabel[i]),
                    'probability': ''.join(str(p) for p in listProba[i]),
                    'predict_List': u' '.join(str(p) for p in listPredict[i]),
                    'classification_report': ''.join(p for p in listClassification),
                }
            except:
                result = {
                    'word': u''.join(listWord[i]),
                    'prediction_value': ''.join(str(p) for p in listPredictionValue[i]),
                    'all_Lable': ' '.join(str(p) for p in listAllLabel[i]),
                    'probability': ''.join(str(p) for p in listProba[i]),
                    'predict_List': u' '.join(str(p) for p in listPredict[i]),
                    'classification_report': ''
                }
            listObject.append(result)

        result1 = {

            'result': listObject

        }

        analysis = json.dumps(result1, ensure_ascii=False)
        return Response(analysis, mimetype='application/json')


api.add_resource(Predict, '/predict/<text>')
api.add_resource(Analysis, '/analysis/<text>')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000)
