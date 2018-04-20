from collections import defaultdict
from train.src.viet_preprocessing.vietprepro import delete_tonemark
import pprint

def make_syllable(filename):
    # A dictionary for holding unique syllables
    syllable_dic = defaultdict(int)
    # Read from corpus
    for line in open(filename, 'r',encoding="utf8"):
        sentence = line.rstrip()
        syllables = sentence.split(u' ')

        for syllable in syllables:
            # Delete tone mark
            no_tonemark_syllable = delete_tonemark(syllable)

            # Delete the tone mark, add it if it is only alphabet
            if no_tonemark_syllable.isalpha():
                syllable_dic[no_tonemark_syllable.lower()] += 1

            # Perform processing on a character string starting with an alphabet and ending with a symbol
            elif no_tonemark_syllable[0].isalpha() and not no_tonemark_syllable[-1].isalpha():

                # . , ! ?Delete it if it is at the end of the string and make it lowercase.
                syllable = no_tonemark_syllable.rstrip('.').rstrip(',').rstrip('!').rstrip('?').lower()
                # After the above processing, if it consists only of alphabets, add it.
                if syllable.isalpha():
                    syllable_dic[syllable] += 1

    # print(sorted(syllable_dic))
    # print("----------------")
    # pprint.pprint(syllable_dic)
    return [k for k, v in sorted(syllable_dic.items())] #return the key value stored by syllable_dic.
