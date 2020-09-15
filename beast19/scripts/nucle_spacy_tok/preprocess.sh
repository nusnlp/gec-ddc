#!/bin/bash

set -x

TRAIN_LOG=train.log.txt
echo "START TIME: `date`" | tee -a TRAIN_LOG

python2.7 preprocess.py -l ../data/nucle3.2.sgml nucle.conll nucle.ann nucle.m2

echo "END TIME: `date`" | tee -a TRAIN_LOG
