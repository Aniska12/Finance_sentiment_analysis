import torch
import random
import argparse
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from vocab import Vocab
from utils import helper
from draw import draw_curve
from loader_test import DataLoader
from gcn import GCNClassifier
import torch.nn.functional as F
from load_w2v import load_pretrained_embedding
from torch.utils.data import TensorDataset, DataLoader as DataLoader1

# neural
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utils import torch_utils
from sklearn.metrics import *

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

parser.add_argument('--batch_size', type=int, default=1, help='Training batch size.')
parser.add_argument('--save_dir', type=str, default='./saved_models/Finance1', help='Root dir for saving models.')
parser.add_argument('--ind', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
warnings.filterwarnings("ignore")

class Attention(nn.Module):
    def __init__(self, hidden_dim, use_W=True, use_bias=False, return_self_attend=False, return_attend_weight=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_W = use_W
        self.use_bias = use_bias
        self.return_self_attend = return_self_attend
        self.return_attend_weight = return_attend_weight

        if self.use_W:
            self.W = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
            nn.init.xavier_uniform_(self.W)

        if self.use_bias:
            self.b = nn.Parameter(torch.zeros(hidden_dim))

        self.u = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.uniform_(self.u)

    def forward(self, x, mask=None):
        if self.use_W:
            #print(x.shape)
            x = torch.tanh(torch.matmul(x, self.W))

        ait = torch.matmul(x, self.u.t())  # Transpose u for correct matrix multiplication
        if self.use_bias:
            ait = ait+self.b

        a = torch.exp(ait)

        if mask is not None:
            a = a*mask

        a =a/ torch.sum(a, dim=1, keepdim=True) + 1e-10  # Add epsilon to avoid numerical instability

        if self.return_self_attend:
            attend_output = torch.sum(x * a.unsqueeze(-1), dim=1)
            if self.return_attend_weight:
                return attend_output, a
            else:
                return attend_output
        else:
            return a


class ATAE_LSTM(nn.Module):
    def __init__(self, num_words, embedding_size, max_len, lstm_units):
        super(ATAE_LSTM, self).__init__()
        self.embedding = nn.Embedding(num_words, embedding_size)
        self.dropout = nn.Dropout(args.input_dropout)
        self.lstm = nn.LSTM(embedding_size * 2, lstm_units, batch_first=True, dropout=args.gcn_dropout)
        self.attention = Attention(2*lstm_units)
        self.dense1 = nn.Linear(lstm_units, lstm_units)
        self.dense2 = nn.Linear(lstm_units, lstm_units)
        self.output_layer = nn.Linear(lstm_units, args.num_class)
        self.max_len = max_len

    def forward(self, inp):
        input_text=inp[0]
        input_aspect=inp[1]
        text_embed = self.dropout(self.embedding(input_text))
        aspect_embed = self.embedding(input_aspect)
        repeat_aspect = aspect_embed.repeat(1, self.max_len, 1)

        input_concat = torch.cat([text_embed, repeat_aspect], dim=-1)
        hidden_vecs, (state_h, rev) = self.lstm(input_concat)
    
        concat = torch.cat([hidden_vecs, repeat_aspect], dim=-1)

        attend_weight = self.attention(concat)
        return attend_weight,hidden_vecs,state_h[0]
    

class CombinedModel(nn.Module):
    def __init__(self, args,word_emb,NUM_WORDS,EMBEDDING_SIZE,Max_Len):
        super(CombinedModel, self).__init__()
        self.gcn1 = GCNClassifier(args, emb_matrix=word_emb)
        self.lstm = ATAE_LSTM(NUM_WORDS,EMBEDDING_SIZE,Max_Len,args.hidden_dim)
        self.args = args
        self.maxlen=Max_Len
        self.output_layer = nn.Linear(args.hidden_dim, args.num_class)
    def forward(self, input_a, input_b):
        # Run input_a through GCN1
        # Run input_b through LSTM
        attention,out,logits,h = self.gcn1(input_a)
        att_s_1,hidden_s,features = self.lstm(input_b)

        max_len = self.maxlen
        batch_size, sequence_length = attention.shape
        k=max_len - sequence_length
        
        padding = torch.zeros((batch_size, k), dtype=attention.dtype,device='cuda:0')
        attention = torch.cat((padding, attention, torch.zeros((batch_size, max_len - sequence_length - k), dtype=attention.dtype,device='cuda:0')), dim=1)
        w_att = nn.Parameter(torch.zeros(max_len, max_len, requires_grad=True)).to('cuda:0')
        
        gate = torch.sigmoid(torch.matmul(att_s_1, w_att) + torch.matmul(attention, w_att))
        att = gate * att_s_1 + (1 - gate) * attention
        
        att_s_1_expanded = att.unsqueeze(-1) 
        mul=hidden_s * att_s_1_expanded
        outputs_s_1 = torch.sum(mul, dim=1)
        #outputs_s_1=self.dense1(outputs_s_1)
        
        pool_t_1 = out+features
        #pool_t_1=self.dense2(pool_t_1)
        
        pool_t_2 = pool_t_1 + outputs_s_1
        #pool_t_2 = self.tan(pool_t_2)
        # Softmax layer to get final probability distribution
        prob = F.softmax(self.output_layer(pool_t_2),dim=1)
        return prob, att

# check saved_models director
model_save_dir = args.save_dir
filename=model_save_dir + '/best_model.pt'
try:
    checkpoint = torch.load(filename)

except BaseException:
    print("Cannot load model from {}".format(filename))
    exit()

dets = checkpoint['config']

#process GCN data
# set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

# load vocab 
token_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_tok.vocab')    # token
post_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_post.vocab')    # position
pos_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_pos.vocab')      # POS
dep_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_dep.vocab')      # deprel
pol_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_pol.vocab')      # polarity
vocab = (token_vocab, post_vocab, pos_vocab, dep_vocab, pol_vocab)

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
print("Loading data from {}...".format(args.data_dir))

train_batch = DataLoader(args.data_dir + '/train.json', 1, args, vocab)
test_batch= DataLoader(args.data_dir + '/test.json', 1, args, vocab)


#process LSTM data
df1= pd.read_csv(args.data_dir + '/train.csv')
df2= pd.read_csv(args.data_dir + '/test.csv')
df1 = df1.drop(['Unnamed: 0'],axis=1)
df2 = df2.drop(['Unnamed: 0'],axis=1)


df1['polarity'][df1["polarity"] == 'negative'] = 0
df1['polarity'][df1["polarity"] == 'positive'] = 2
df1['polarity'][df1["polarity"] == 'neutral'] = 1
Aspects_term = df1['polarity'].unique()

df2['polarity'][df2["polarity"] == 'negative'] = 0
df2['polarity'][df2["polarity"] == 'positive'] = 2
df2['polarity'][df2["polarity"] == 'neutral'] = 1
Aspects_term = df1['polarity'].unique()

df1["polarity"] = pd.to_numeric(df1["polarity"],errors='coerce')
df1.dropna(subset = ['polarity'], inplace = True)

df2["polarity"] = pd.to_numeric(df2["polarity"],errors='coerce')
df2.dropna(subset = ['polarity'], inplace = True)

NUM_WORDS = 100000 ## MAx of words to keep, based on word frequency.
EMBEDDING_SIZE = dets.hidden_dim ## the length of the Vector the will

X_train=df1
y_train=df1['polarity']

X_test=df2
y_test=df2['polarity']

tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True, )
## Fit_on_texts : Updates internal vocabulary based on a list of texts.
tokenizer.fit_on_texts(list(X_train.Sentence))
text_X_train_tokenized = tokenizer.texts_to_sequences(X_train.Sentence) # list of tokenized sentences
Aspect_X_train_tokenized = tokenizer.texts_to_sequences(X_train['Aspect Term']) # list of tokenized sentences
text_X_test_tokenized = tokenizer.texts_to_sequences(X_test.Sentence) # list of tokenized sentences
Aspect_X_test_tokenized = tokenizer.texts_to_sequences(X_test['Aspect Term']) # list of tokenized sentences

Max_Len =  max([len(one_title) for one_title in text_X_train_tokenized])

text_X_train_padded = pad_sequences(text_X_train_tokenized, maxlen=Max_Len)
aspect_X_train_padded = pad_sequences(Aspect_X_train_tokenized, maxlen=1)
text_X_test_padded = pad_sequences(text_X_test_tokenized, maxlen=Max_Len)
aspect_X_test_padded = pad_sequences(Aspect_X_test_tokenized, maxlen=1)

text_X_train_padded = torch.tensor(text_X_train_padded, dtype=torch.long)
aspect_X_train_padded = torch.tensor(aspect_X_train_padded, dtype=torch.long)
text_X_test_padded = torch.tensor(text_X_test_padded, dtype=torch.long)
aspect_X_test_padded = torch.tensor(aspect_X_test_padded, dtype=torch.long)

# Create a DataLoader
train_dataset = TensorDataset(text_X_train_padded[args.ind:args.ind+1], aspect_X_train_padded[args.ind:args.ind+1])
test_dataset = TensorDataset(text_X_test_padded[args.ind:args.ind+1], aspect_X_test_padded[args.ind:args.ind+1])

train_loader = DataLoader1(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader1(test_dataset, batch_size=1, shuffle=False)


model=CombinedModel(dets,word_emb,NUM_WORDS,EMBEDDING_SIZE,Max_Len).to("cuda:0")
model.load_state_dict(checkpoint['model'])


context={0:"Negative",1:"Neutral",2:"Positive"}
for train_data, batch in zip(test_loader,test_batch):
    batch = [b.cuda() for b in batch]
    train_data =  [t.cuda() for t in train_data]

    inputs = batch[0:8]
    label = batch[-1]
    model.eval()
    logits,weight = model(inputs,train_data)
    out=torch.argmax(logits, -1)

    print("\nSentence is:",X_test.Sentence[args.ind],"\nAspect is:",X_test['Aspect Term'][args.ind])
    print("\nProbability output is: ",list(logits))
    print("\nThe given sentence has \"{}\" polarity with respect to given sentence".format(context[out.item()]))