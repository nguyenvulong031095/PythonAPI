from collections import defaultdict
from pickle import dump
import pandas as pd


class PrintFeatures():
    def __init__(self):
        self.class_num = 1
        self.feature_num = 1

        # Hold class number and feature number for syllable
        self.class_dict = {}
        self.feature_dict = {}

        # Id for LIBLINEAR
        self.class_list = []
        self.feature_list = []

    def add_liblinear_format(self, feature):
        class_number = self.get_class_num(feature.cur_syllable)
        features = self.get_feature_nums(feature.get_features())

        features = self.feature_nums2liblinear_format(features)
        self.class_list.append(float(class_number))
        self.feature_list.append(features)

    def get_class_num(self, cur_syllable):

        if cur_syllable not in self.class_dict:
            self.class_dict[cur_syllable] = self.class_num
            self.class_num += 1

        return self.class_dict.get(cur_syllable)

    def get_feature_nums(self, features):
        feature_nums = []
        for feature in features:
            if feature not in self.feature_dict:
                self.feature_dict[feature] = self.feature_num
                self.feature_num += 1

            feature_nums.append(self.feature_dict.get(feature))
        return sorted(feature_nums)

    def feature_nums2liblinear_format(self, feature_nums):
        d = defaultdict(int)
        for feature_num in feature_nums:
            d[feature_num] += 1
        return dict([(int(feature_num), float(freq)) for feature_num, freq in sorted(d.items())])



    def save_confusion_matrix(self,save_path,confusion_matrix_result):
        f = open(save_path,'w',encoding="utf8")
        f.write(str(confusion_matrix_result))
        f.close

        




    def save_classification_report(self,save_path,classification_report_result):
         f = open(save_path,'w',encoding="utf8")
         f.write(str(classification_report_result))
         f.close


    def save_train_data(self,save_path):
        try:
            #print(self.feature_list)

            with open(save_path,'w') as file:
                for index in range(len(self.feature_list)):
                    line = self.feature_list[index]
                    #print(type(line))
                    line = str(line) + u'\n'
                    file.write(str(line))

        except:
            print("Cant write")

    def save_class_data(self,save_path):
        try:
            with open(save_path, 'w') as file:
                line = self.class_list
                line =str(line)
                file.write(line)

        except:
            print("Cant write")

    def save_class_dict(self, save_path):

        # with open(save_path, 'w') as fout:
        #     lines = [u'{}\t{}'.format(k, v) for k, v in sorted(self.class_dict.items(), key=lambda x: x[1])]
        #     fout.write(u'\n'.join(lines))
        try:
            with open(save_path, 'wb') as fout:
                lines = [u'{}\t{}'.format(k, v) for k, v in sorted(self.class_dict.items(), key=lambda x: x[1])]
                #print(lines)
                lines = u'\n'.join(lines)
                #print(lines)
                #print(lines.encode('utf-8'))
                fout.write(lines.encode('utf-8'))
        except:
            print("Cant write")

    def save_feature_dict(self, save_path):

        try:
            with open(save_path, 'wb') as file:
                lines = [u'{}\t{}'.format(k, v) for k, v in sorted(self.feature_dict.items(), key=lambda x: x[1])]
                file.write(u'\n'.join(lines).encode('utf-8'))
        except:
            print("Cant write")


    def save_model(self, save_path, model, name):
        #print("Model:",model)
        #print("save_path:", save_path)

        try:
            dump(model, open(save_path,'wb'))
            # with open(save_path,'wb') as file:
            #     pickle.dump(model, name + '.pkl')
        except:
            print("Cant write")

    def print_features(self):
        for c, f in zip(self.class_list, self.feature_list):
            f = [u'{}:{}'.format(k, v) for k, v in sorted(f.items())]
            print(u'{} {}'.format(c, u' '.join(f)).encode('utf-8'))
