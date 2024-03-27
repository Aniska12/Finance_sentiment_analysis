#!/bin/bash
# Prediction command for different datasets.

#provide the index of the data to be tested from the corresponding dataset test file to predict its output

#python predict.py --ind 1584 --data_dir dataset/Finance --vocab_dir dataset/Finance --save_dir ./saved_models/Finance
#python predict.py --data_dir dataset/Semeval2015 --vocab_dir dataset/Semeval2015 --save_dir ./saved_models/Semeval2015

python predict.py --ind 17 --data_dir dataset/FiQA --vocab_dir dataset/FiQA --save_dir ./saved_models/FiQA