#!/bin/bash
# training command for different datasets.

#python train.py --data_dir dataset/Finance --vocab_dir dataset/Finance --save_dir ./Abeliation/final --img_dir ./Abeliation/final --num_layers 2 --log_step 50 --num_epoch 128 --lr 0.0001 --wd 0.1 --input_dropout 0.1 --lstm_dropout 0.1 --gcn_dropout 0.4 --hidden_dim 30 --optim adamw
#python train.py --data_dir dataset/Semeval2015 --vocab_dir dataset/Semeval2015 --save_dir ./Abeliation/final --img_dir ./Abeliation/final --num_layers 2 --log_step 10 --num_epoch 200 --lr 0.0001 --wd 0.1 --input_dropout 0.2 --lstm_dropout 0.2 --gcn_dropout 0.4 --hidden_dim 30 --optim adamw
python train.py --data_dir dataset/FiQA --vocab_dir dataset/FiQA --save_dir ./Abeliation/final --img_dir ./Abeliation/final --num_layers 2 --log_step 10 --num_epoch 128 --lr 0.001 --wd 0.001 --input_dropout 0.2 --lstm_dropout 0.1 --gcn_dropout 0.7 --hidden_dim 40 --optim adagrad