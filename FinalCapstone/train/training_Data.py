import configparser
from itertools import zip_longest
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from train.src.train_utils import *
from train.src.train_format import *
from train.src.make_syllable import *
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.svm import  OneClassSVM

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def define_sparseMatix(feature_list):
    tmp = []
    for index in range(len(feature_list)):
        tmp.append(max(feature_list[index].keys()))

    max_column = max(tmp)

    n_feature = max_column
    sparse = []
    for index in range(len(feature_list)):

        temp = []
        for counter in range(1, n_feature + 1):
            temp.append(feature_list[index].get(counter, 0))
        sparse.append(temp)
    return sparse

def main():
    inifile = configparser.SafeConfigParser()
    inifile.read("/Users/nguyenvulong/Documents/CapstonePro/FinalCapstone/config.ini")
    path1 = inifile.get("settings", "path1")
    path2 = inifile.get("settings", "path2")
    preserve_dir_path = inifile.get("settings", "preserve_dir_path")
    window_size = int(inifile.get("settings", "window_size"))

    # Read the syllable list of the learning object
    print('loading syllable list')
    syllable_list = make_syllable(path1)
    print('pick feature and training')
    cannot_output = 0
    random_state = np.random.RandomState(0)

    parameters = {'C': [1, 10, 100]}
    clf = GridSearchCV(LinearSVC(), parameters)

    for target_syllable in syllable_list:
        pf = PrintFeatures()
        #print("target Syllable:", target_syllable)
        for syllable_indexs, sentence in iter_pick_sentence(target_syllable, path1, path2):
            # print(syllable_indexs)
            # class_id, feature, feature_id Create feature while creating
            for index, syllable in syllable_indexs:
                f = get_feature(syllable, index, sentence, window_size)
                #print("Get_feature:", f.get_features())
                pf.add_liblinear_format(f)



        #print("-------------------------------------")
        if pf.feature_list:
            sparseMatrix = define_sparseMatix(pf.feature_list)
            if len(pf.class_list) >= 2 and max(pf.class_list) > 1:
                try:

                    clf.fit(np.array(sparseMatrix), np.array(pf.class_list))
                    result_C = clf.best_params_.get('C')
                    if result_C:
                        clf = LinearSVC(C= result_C)
                        clf.fit(np.array(sparseMatrix), np.array(pf.class_list))
                        X_train, X_test, Y_train, Y_test = train_test_split(sparseMatrix, np.array(pf.class_list),
                                                                            test_size=.5,
                                                                            random_state=random_state)

                        model_test_split = LinearSVC(C=result_C)
                        model_test_split.fit(X_train, Y_train)
                        Y_predict = model_test_split.predict(X_test)
                        label = list(map(str,list(sorted(set(Y_test)))))
                        result_label = []
                        for i in range(len(label)):
                            tmp = list(pf.class_dict.keys())[int(float(label[i]) - 1.0)]
                            result_label.append(tmp)
                        classification_report_result = classification_report(Y_test, Y_predict,target_names=result_label)
                        # confusion_matrix_result = confusion_matrix(Y_test,Y_predict)
                        pf.save_classification_report("{}/{}_classification.txt".format(preserve_dir_path,target_syllable), classification_report_result)
                        # pf.save_confusion_matrix("{}/{}_matrix.txt".format(preserve_dir_path,target_syllable),confusion_matrix_result)
                            

                except:
                    clf = LinearSVC()
                    clf.fit(np.array(sparseMatrix), np.array(pf.class_list))
            elif len(pf.class_list) >= 2 and max(pf.class_list) == 1:
                clf = OneClassSVM(kernel='linear', nu=0.06)
                clf.fit(sparseMatrix)
            elif len(pf.class_list) == 1 and max(pf.class_list) == 1:
                clf = OneClassSVM(kernel='linear', nu=0.06)
                clf.fit(sparseMatrix)

        else:
            pass
        try:
            if pf.feature_list:

                target_syllable = target_syllable.encode('utf-8')
                #pf.save_train_data("{}/{}.train_data".format(preserve_dir_path,target_syllable))
                #pf.save_class_data("{}/{}.label_data".format(preserve_dir_path, target_syllable))
                pf.save_class_dict("{}/{}.class_map".format(preserve_dir_path, target_syllable))
                pf.save_feature_dict("{}/{}.feature_map".format(preserve_dir_path, target_syllable))
                pf.save_model("{}/{}.pkl".format(preserve_dir_path,target_syllable),model=clf, name=target_syllable)
        except:
            cannot_output += 1
            continue
    print("Can't train:{}".format(cannot_output))



def iter_pick_sentence(keyword, path1, path2):
    """Yield the sentence containing the target syllable"""
    for sentence, no_tonemark_sentence in zip_longest(open(path1, 'r',encoding="utf8"), open(path2, 'r',encoding="utf8")):
        #print("SENTENCE:",sentence)
        no_tonemark_lower_syllables = no_tonemark_sentence.rstrip().split(u' ')
        #print("NO_TONEMARK_LOWER_SYLLABLES:",no_tonemark_lower_syllables)
        if keyword in no_tonemark_lower_syllables:
            sentence = sentence.rstrip().split(u' ')
            # Syllable and its index
            syllable_index = [(i, sentence[i]) for i, w in enumerate(no_tonemark_lower_syllables) if w == keyword]
            #print("SYLLABLE_INDEX:",syllable_index)

            yield syllable_index, no_tonemark_lower_syllables


if __name__ == "__main__":
    main()