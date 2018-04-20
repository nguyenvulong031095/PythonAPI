from src.viet_preprocessing.vietprepro import BoDau
import sys


def main():
    for line in sys.stdin:
        sentence = line.rstrip()
        no_tone_mark_sentence = u''.join([BoDau(a) for a in sentence])
        print(no_tone_mark_sentence.lower().encode('utf-8'))


if __name__ == "__main__":
    main()
