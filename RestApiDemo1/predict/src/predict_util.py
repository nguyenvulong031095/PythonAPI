def load_syllable_map(filename):

    return dict(
        [reversed(line.rstrip().split(u'\t'))
         for line in open(filename, 'r')]
    )

def load_syllable_class_map(filename):
    return dict(
        [line.rstrip().split(u'\t')
         for line in open(filename, 'r')]
    )