class Feature():
    def __init__(self, feature):
        self.cur_syllable = feature[0]
        self.syllables = feature[1]
        self.ngram_syllables = feature[2]
        self.uni_gram_syllable_kinds = feature[3]
        self.bi_gram_syllable_kinds = feature[4]

    def get_cur_syllable(self):
        return self.cur_syllable

    def get_features(self):
        all_features = []
        all_features.extend(self.syllables)
        all_features.extend(self.ngram_syllables)
        all_features.extend(self.uni_gram_syllable_kinds)
        all_features.extend(self.bi_gram_syllable_kinds)
        return all_features


def get_feature(cur_keyword, index, syllables, window_size):
    # print cur_keyword.encode('utf-8'), index, syllables, len(syllables)
    target_syllable = get_window_syllables(syllables, index, window_size)

    bi_gram = list2ngram(target_syllable, 2)

    feature = (
        cur_keyword,  # Target syllable
        target_syllable,  # Syllables for window width
        ngram2feature_form(bi_gram),
        get_syllable_type_feature(target_syllable),
        get_syllable_type_feature(bi_gram),
    )
    return Feature(feature)


def get_window_syllables(syllables, index, window_size):
    if index < window_size:
        return syllables[0:index + window_size + 1]
    else:
        return syllables[index - window_size:index + window_size + 1]


def get_syllable_type_feature(syllable_ngrams):
    syllable_type_stack = []
    for syllable_ngram in syllable_ngrams:
        syllable_type = ''.join([syllable2type(syllable) for syllable in syllable_ngram])
        syllable_type_stack.append(syllable_type)
    

    return list(set(syllable_type_stack))


def syllable2type(syllable):
    if syllable.isdigit():
        return u'N'

    elif syllable[0].isupper():
        return u'U'

    elif syllable[0].islower():
        return u'L'

    else:
        return u'O'


def list2ngram(uni, n=1):
    ngram_result = []
    loop = 0
    while loop + n <= len(uni):
        ngram_result.append(uni[loop:loop + n])
        loop += 1
    return ngram_result


def ngram2feature_form(ngrams):
    return [' '.join(ngram) for ngram in ngrams]