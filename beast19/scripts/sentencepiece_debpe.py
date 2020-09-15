# -*- coding: utf-8 -*
import sys
import re

def detokenize(line):
	print(line.replace(" ", "").replace("▁", " ").strip())

for line in sys.stdin:
	detokenize(line)