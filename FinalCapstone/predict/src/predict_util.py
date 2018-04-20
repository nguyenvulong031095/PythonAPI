from ast import literal_eval
def load_syllable_map(filename):

    return dict(
        [reversed(line.rstrip().split(u'\t'))
         for line in open(filename, 'r',encoding="utf8")]
    )

def load_syllable_class_map(filename):
    return dict(
        [line.rstrip().split(u'\t')
         for line in open(filename, 'r',encoding="utf8")]
    )

