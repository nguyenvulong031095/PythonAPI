import os
from predict.src.predict_util import *
from collections import defaultdict

def train_data_(model_dir, target_syllable_lower):
    train_data_path = os.path.join(model_dir,'{}.train_data'.format(target_syllable_lower))
    result = load_train_data(train_data_path)
    #print(result)
    return result

def lable_data(model_dir, target_syllable_lower):
    label_data_path = os.path.join(model_dir,"{}.label_data".format(target_syllable_lower))
    result = load_label_data(label_data_path)
    return result

def feature_mapping(model_dir, target_syllable_lower, features):
    feature_map_path = os.path.join(model_dir, '{}.feature_map'.format(target_syllable_lower))
    syllable_feature_map = load_syllable_class_map(feature_map_path)

    maxFeature = len(syllable_feature_map)


    mapped = defaultdict(int)
    for feature in features.get_features():
        ad = syllable_feature_map.get(feature)
        if ad != None:
            mapped[int(ad)] += 1
            # print(mapped)
    return mapped, maxFeature


def class_mapping(model_dir, target_syllable_lower,indexArr):

    syllable_map_path = os.path.join(model_dir, '{}.class_map'.format(target_syllable_lower))
    syllable_class_map = load_syllable_map(syllable_map_path)
    result = []
    label = []
    label = list(syllable_class_map.values())
    #print("Label:", label)
    for i in range(len(indexArr)):

        result.append(syllable_class_map.get(str(int(indexArr[i]))))

    if result.__len__() != 0:
        return result, label
    else:
        return 1
