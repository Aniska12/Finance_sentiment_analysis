import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from vocab import Vocab
from utils import helper
from shutil import copyfile
from draw import draw_curve
from sklearn import metrics
from loader import DataLoader
from trainer import GCNTrainer
from torch.autograd import Variable
from load_w2v import load_pretrained_embedding

import os
import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split


from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Dropout, Conv1D, MaxPool1D, Flatten, concatenate, Dense, \
    LSTM, Bidirectional, Activation, MaxPooling1D, Add, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, RepeatVector, \
    TimeDistributed, Permute, multiply, Lambda, add, Masking, BatchNormalization, Softmax, Reshape, ReLU, \
    ZeroPadding1D, subtract
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import tensorflow.keras.backend as K
import tensorflow as tf
from keras import backend as K, initializers, regularizers, constraints


# Import our dependencies
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
from keras import backend as K
import keras.layers as layers
from keras.models import Model, load_model
from tensorflow.keras.layers import Layer, InputSpec
import numpy as np

from statistics import mode

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tensorflow as tf

# Load Huggingface transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast, TFAutoModel

# Then what you need from tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils.generic_utils import to_list
import tensorflow_hub as hub



import re
import nltk
from nltk.corpus import stopwords

# neural
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential

#Metrics
from sklearn.metrics import balanced_accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/Finance')
parser.add_argument('--vocab_dir', type=str, default='dataset/Finance')
parser.add_argument('--glove_dir', type=str, default='dataset/glove')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=50, help='GCN mem dim.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
parser.add_argument('--num_class', type=int, default=3, help='Num of sentiment class.')

parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.1, help='GCN layer dropout rate.')
parser.add_argument('--lower', default=True, help='Lowercase all words.')
parser.add_argument('--direct', default=False)
parser.add_argument('--loop', default=True)

parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
parser.add_argument('--rnn_hidden', type=int, default=50, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.')

parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--lrlstm', type=float, default=0.00001, help='learning rate.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adamax', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=32, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_dir', type=str, default='./saved_models/Finance1', help='Root dir for saving models.')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()


#LSTM architecture
class Attention(Layer):
    def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None, W_constraint=None,
                 u_constraint=None, b_constraint=None, use_W=True, use_bias=False, return_self_attend=False,
                 return_attend_weight=True, **kwargs):
        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.use_W = use_W
        self.use_bias = use_bias
        self.return_self_attend = return_self_attend    # whether perform self attention and return it
        self.return_attend_weight = return_attend_weight    # whether return attention weight
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        if self.use_W:
            self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),  initializer=self.init,
                                     name='{}_W'.format(self.name), regularizer=self.W_regularizer,
                                     constraint=self.W_constraint)
        if self.use_bias:
            self.b = self.add_weight(shape=(input_shape[1],), initializer='zero', name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer, constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],), initializer=self.init, name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer, constraint=self.u_constraint)

        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        if self.use_W:
            x = K.tanh(K.dot(x, self.W))

        ait = Attention.dot_product(x, self.u)
        if self.use_bias:
            ait += self.b

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        if self.return_self_attend:
            attend_output = K.sum(x * K.expand_dims(a), axis=1)
            if self.return_attend_weight:
                return [attend_output, a]
            else:
                return attend_output
        else:
            return a

    def compute_output_shape(self, input_shape):
        if self.return_self_attend:
            if self.return_attend_weight:
                return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]
            else:
                return input_shape[0], input_shape[-1]
        else:
            return input_shape[0], input_shape[1]

    @staticmethod
    def dot_product(x, kernel):
        if K.backend() == 'tensorflow':
            return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
        else:
            return K.dot(x, kernel)
    
def ae_lstm(lstm_units = 512):
    input_text = Input(shape=(Max_Len,))
    input_aspect = Input(shape=(1,),)

    word_embedding = Embedding(NUM_WORDS, EMBEDDING_SIZE, input_length=Max_Len)
    text_embed = SpatialDropout1D(0.2)(word_embedding(input_text))

    asp_embedding = Embedding(NUM_WORDS, EMBEDDING_SIZE, input_length=Max_Len)
    aspect_embed = asp_embedding(input_aspect)


    aspect_embed = Flatten()(aspect_embed)  # reshape to 2d
    repeat_aspect = RepeatVector(Max_Len)(aspect_embed)  # repeat aspect for every word in sequence

    input_concat = concatenate([text_embed, repeat_aspect], axis=-1)
    hidden = LSTM(lstm_units)(input_concat)
    Dense_layer  = Dense(128, activation='relu')(hidden)
    output_layer = Dense(3, activation='softmax')(Dense_layer)
    return Model([input_text, input_aspect], output_layer)

def at_lstm(lstm_units = 512):
        input_text = Input(shape=(Max_Len,))
        input_aspect = Input(shape=(1,),)

        word_embedding = Embedding(NUM_WORDS, EMBEDDING_SIZE, input_length=Max_Len)
        text_embed = SpatialDropout1D(0.2)(word_embedding(input_text))

        asp_embedding = Embedding(NUM_WORDS, EMBEDDING_SIZE, input_length=Max_Len)
        aspect_embed = asp_embedding(input_aspect)
        aspect_embed = Flatten()(aspect_embed)  # reshape to 2d
        repeat_aspect = RepeatVector(Max_Len)(aspect_embed)  # repeat aspect for every word in sequence
        hidden_vecs = LSTM(lstm_units, return_sequences=True)(text_embed)  # hidden vectors output by lstm
        concat = concatenate([hidden_vecs, repeat_aspect], axis=-1)  # mask after concatenate will be same as hidden_out's mask
        print(concat.shape)
         # apply attention mechanism
        attend_weight = Attention()(concat)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)
        Dense_layer  = Dense(128, activation='relu')(attend_hidden)
        output_layer = Dense(3, activation='softmax')(Dense_layer)
        return Model([input_text, input_aspect], output_layer)

# attention-based lstm with aspect embedding
def atae_lstm(lstm_units = 128):
    input_text = Input(shape=(Max_Len,))
    input_aspect = Input(shape=(1,),)

    word_embedding = Embedding(NUM_WORDS, EMBEDDING_SIZE, input_length=Max_Len)
    text_embed = SpatialDropout1D(0.2)(word_embedding(input_text))

    asp_embedding = Embedding(NUM_WORDS, EMBEDDING_SIZE, input_length=Max_Len)

    aspect_embed = asp_embedding(input_aspect)
    aspect_embed = Flatten()(aspect_embed)  # reshape to 2d
    repeat_aspect = RepeatVector(Max_Len)(aspect_embed)  # repeat aspect for every word in sequence

    input_concat = concatenate([text_embed, repeat_aspect], axis=-1)
    print(input_concat.shape)
    hidden_vecs, state_h, _ = LSTM(lstm_units, return_sequences=True, return_state=True)(input_concat)
    concat = concatenate([hidden_vecs, repeat_aspect], axis=-1)

    # apply attention mechanism
    attend_weight = Attention()(concat)
    attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
    attend_hidden = multiply([hidden_vecs, attend_weight_expand])
    attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

    attend_hidden_dense = Dense(lstm_units)(attend_hidden)
    last_hidden_dense = Dense(lstm_units)(state_h)
    final_output = Activation('tanh')(add([attend_hidden_dense, last_hidden_dense]))
    output_layer = Dense(3, activation='softmax')(final_output)
    return Model([input_text, input_aspect], output_layer)

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtracting the max for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


#process GCN data
# set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
helper.print_arguments(args)

# load vocab 
print("Loading vocab...")
token_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_tok.vocab')    # token
post_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_post.vocab')    # position
pos_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_pos.vocab')      # POS
dep_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_dep.vocab')      # deprel
pol_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_pol.vocab')      # polarity
vocab = (token_vocab, post_vocab, pos_vocab, dep_vocab, pol_vocab)
print("token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(len(token_vocab), len(post_vocab), len(pos_vocab), len(dep_vocab), len(pol_vocab)))
args.tok_size = len(token_vocab)
args.post_size = len(post_vocab)
args.pos_size = len(pos_vocab)

# load pretrained word emb
print("Loading pretrained word emb...")
word_emb = load_pretrained_embedding(glove_dir=args.glove_dir, word_list=token_vocab.itos)
assert len(word_emb) == len(token_vocab)
assert len(word_emb[0]) == args.emb_dim
word_emb = torch.FloatTensor(word_emb)                                 # convert to tensor

# load data
print("Loading data from {} with batch size {}...".format(args.data_dir, args.batch_size))
train_batch = DataLoader(args.data_dir + '/train.json', args.batch_size, args, vocab)

# check saved_models director
model_save_dir = args.save_dir
helper.ensure_dir(model_save_dir, verbose=True)
# log
file_logger = helper.FileLogger(model_save_dir + '/' + args.log, header="# epoch\ttrain_loss\ttest_loss\ttrain_acc\ttest_acc\ttest_f1")


#process LSTM data
df= pd.read_csv(args.data_dir + '/train.csv')
df = df.drop(['Unnamed: 0'],axis=1)
print(df['polarity'].isna().sum())
df['polarity'][df["polarity"] == 'negative'] = 0
df['polarity'][df["polarity"] == 'positive'] = 1
df['polarity'][df["polarity"] == 'neutral'] = 2
Aspects_term = df['polarity'].unique()

df["polarity"] = pd.to_numeric(df["polarity"],errors='coerce')
df.dropna(subset = ['polarity'], inplace = True)

NUM_WORDS = 100000 ## MAx of words to keep, based on word frequency.
EMBEDDING_SIZE = 128 ## the length of the Vector the will

X_train=df
y_train=df['polarity']
y_valid=list(y_train)

tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True, )
## Fit_on_texts : Updates internal vocabulary based on a list of texts.
tokenizer.fit_on_texts(list(X_train.Sentence))
text_X_train_tokenized = tokenizer.texts_to_sequences(X_train.Sentence) # list of tokenized sentences
Aspect_X_train_tokenized = tokenizer.texts_to_sequences(X_train['Aspect Term']) # list of tokenized sentences

Max_Len =  max([len(one_title) for one_title in text_X_train_tokenized])

text_X_train_padded = pad_sequences(text_X_train_tokenized, maxlen=Max_Len)
aspect_X_train_padded = pad_sequences(Aspect_X_train_tokenized, maxlen=1)


# build model
gcn_model = GCNTrainer(args, emb_matrix=word_emb)

lstm_model =  atae_lstm()
lstm_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=args.lrlstm))

# start training
train_acc_history, train_loss_history= [0.], []
bat=args.batch_size
for epoch in range(1, args.num_epoch+1):
    train_loss = []
    ans=0
    for i, batch in enumerate(train_batch):
        bat=min(bat,len(batch[0]))
        train_y_data= y_train[i:i+bat]
        train_data =  [text_X_train_padded[i:i+bat].reshape(bat,Max_Len),aspect_X_train_padded[i:i+bat]]
        
        lstm_out = lstm_model(train_data)
        gcn_out = gcn_model.update(batch)[2]
        lstm_model .fit(x = train_data, y = train_y_data, batch_size=32, epochs=1,verbose=0)
        
        alpha_i = gcn_out.cpu().detach().numpy()
        beta_i = lstm_out.numpy()
        
        
        # Define the fusion gate parameters
        Wg = np.random.rand(2, 1)  # Assuming you have 2 attention weights to be fused

        # Fusion gate calculation
        g_input = np.concatenate([alpha_i.reshape(-1, 1), beta_i.reshape(-1, 1)], axis=1)

        g = 1 / (1 + np.exp(-(g_input.dot(Wg))))  # Sigmoid function
        g = g.reshape(alpha_i.shape)
        
        # Attention fusion
        gamma_i = g * alpha_i + (1 - g) * beta_i

        # Normalization
        gamma_i_normalized = np.exp(gamma_i) / np.sum(np.exp(gamma_i))
        
        #softmax output
        softmax_output = softmax(gamma_i_normalized)
        
        
        abcd=[]
        for j in range(bat):
            abcd.append(np.argmax(softmax_output[j]))
            if(abcd[-1]==y_valid[i+j]):ans+=1
        
        if(not(i%10)):
            print("Accuracy is:",(ans/((i+1)*bat))*100,"%")
        
        abcd=tf.constant(abcd,dtype=tf.float32)
        efgh=tf.constant(y_valid[i:i+bat],dtype=tf.float32)
        mse = tf.reduce_mean(tf.square(abcd - efgh)).numpy()
        train_loss.append(mse)
        
    print("")
    print("Epoch no",epoch)
    train_acc_history.append((ans/len(y_valid))*100)
    train_loss_history.append(sum(train_loss)/(len(train_loss)))
    
    print(f"Accuracy is {train_acc_history[-1]}%")
    print(f"Loss is {train_loss_history[-1]}")
    print("___________________________________________________________")

print("Training ended with {} epochs.".format(epoch))

#save model weights
gcn_model.save(model_save_dir + '/gcn_model.pt')
lstm_model.save_weights(model_save_dir + '/lstm_model.h5')

draw_curve(train_acc_history, train_loss_history, args.num_epoch)