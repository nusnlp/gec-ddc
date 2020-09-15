from nltk.tokenize.treebank import TreebankWordDetokenizer


def detokenize(s):
    tokens  = s.strip().split()
    print(TreebankWordDetokenizer().detokenize(tokens))

import sys

for line in sys.stdin:
    detokenize(line)

