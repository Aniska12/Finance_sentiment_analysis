#!/bin/bash
# build vocab for different datasets
python prepare_vocab.py --data_dir dataset/Finance --vocab_dir dataset/Finance
python prepare_vocab.py --data_dir dataset/Finance2 --vocab_dir dataset/Finance2
python prepare_vocab.py --data_dir dataset/Finance3 --vocab_dir dataset/Finance3
