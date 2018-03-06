import os
from predict.src.predict_util import *
from collections import defaultdict
import pprint

def feature_mapping(model_dir, target_syllable_lower, features):
    feature_map_path = os.path.join(model_dir, '{}.feature_map'.format(target_syllable_lower))

    # 音節に対するクラスマッピングファイルを読み込み
    syllable_feature_map = load_syllable_class_map(feature_map_path)
    #pprint.pprint(syllable_feature_map)
    mapped = defaultdict(int)
    for feature in features.get_features():
        ad = syllable_feature_map.get(feature)
        if ad != None:
            mapped[int(ad)] += 1
            # print(mapped)

    return mapped


def class_mapping(model_dir, target_syllable_lower, p_label):
    syllable_map_path = os.path.join(model_dir, '{}.class_map'.format(target_syllable_lower))
    # 音節に対するマッピングファイルを読み込み

    syllable_class_map = load_syllable_map(syllable_map_path)
    #print(type(syllable_class_map))
    #print(syllable_class_map)
    # s = syllable_class_map.get((int(p_label)))
    #print(p_label)
    s = syllable_class_map.get(str(int(p_label)))
    #print(syllable_class_map.get('1'))
    #print("s:" ,s)
    if s != None:
        return s
    else:
        return 1
