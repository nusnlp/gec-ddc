from nltk.tokenize import sent_tokenize

def sent_split(s):
    s = s.strip()
    print('\n'.join(sent_tokenize(s)))


import sys

for line in sys.stdin:
    sent_split(line)

