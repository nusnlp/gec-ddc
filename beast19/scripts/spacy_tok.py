import spacy
import en_core_web_sm
nlp=en_core_web_sm.load()

def tokenize(s):
    s = s.strip()
    print(' '.join([tok.text for tok in list(nlp(s))]))


import sys

for line in sys.stdin:
    tokenize(line)