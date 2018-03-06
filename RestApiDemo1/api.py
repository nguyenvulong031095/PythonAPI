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
import configparser
from predict.src.liblinear.liblinear import *
from predict.src.liblinear.liblinearutil import *
from predict.src.make_feature import *
from predict.src.train_utils import *
import json

print('scipy: %s' % json.__version__)
def tone_predict(i, syllables, model_dir, file_path, window_size):
    target_syllable = syllables[i]
    target_syllable_lower = syllables[i].lower()

    target_syllable_lower = "b'" + target_syllable_lower + "'"
    print(target_syllable_lower)
    if target_syllable_lower not in file_path:
        return u'[{}]'.format(target_syllable)

    model_path = os.path.join(model_dir, '{}.model'.format(target_syllable_lower))
    #print(model_path)
    # 素性の作成
    features = get_feature(target_syllable, i, syllables, window_size)
    print("feature:", features.get_features())
    mapped_feature = feature_mapping(model_dir, target_syllable_lower, features)
    print("mapped_feature:",mapped_feature)
    # estimate
    # loading a model
    model = load_model(model_path)
    p_label, p_acc, p_val = predict([1], [mapped_feature], model)
    print("p_label:",p_label)
    print("p_acc:", p_acc)
    print("p_val:", p_val)
    print("-------------------------")
    # Restore the estimated class number to a syllable
    try:
        # print("Model_dir: ",model_dir)
        # print("target_syllable_lower:" , target_syllable_lower)
        # print("p_lable[0]", p_label[0])
        # target_syllable_lower = target_syllable_lower.replace("b'", "").replace("'","")
        # print(target_syllable_lower)
        mapped_class = class_mapping(model_dir, target_syllable_lower, p_label[0])
        #print(mapped_class)
    except:
        print("Sai roi ahihi")
    return mapped_class

class Predict(Resource):
    def get(self,text):
        text = str(text)
        inifile = configparser.SafeConfigParser()
        inifile.read("/Users/nguyenvulong/Documents/CapstoneProject/RestApiDemo1/config.ini")
        model_dir = inifile.get("settings", "model_dir")
        file_path = set([p.split('.')[0] for p in os.listdir(model_dir)])
        window_size = 2
        
        syllables = text.rstrip().split(u' ')

        predicted_syllables = [tone_predict(i, syllables, model_dir, file_path, window_size) for i in
                               range(len(syllables))]
        
       
# =============================================================================
#         'input': text,
# =============================================================================
        
        result ={    
                'output': u' '.join(predicted_syllables) }
        
        print(json.dumps(result,ensure_ascii=False))
        return Response(json.dumps(result,ensure_ascii=False),mimetype='application/json')
  

api.add_resource(Predict,'/predict/<text>')

if __name__ == '__main__':
     app.run(host='0.0.0.0',port=3000)
