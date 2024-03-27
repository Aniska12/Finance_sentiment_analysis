import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from vocab import Vocab
from utils import helper
from draw import draw_curve
from loader import DataLoader
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
parser.add_argument('--hidden_dim', type=int, default=128, help='GCN mem dim.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
parser.add_argument('--num_class', type=int, default=3, help='Num of sentiment class.')

parser.add_argument('--input_dropout', type=float, default=0.1, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.1, help='GCN layer dropout rate.')
parser.add_argument('--lstm_dropout', type=float, default=0.1, help='LSTM dropout rate.')
parser.add_argument('--lower', default=True, help='Lowercase all words.')
parser.add_argument('--direct', default=False)
parser.add_argument('--loop', default=True)

parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
parser.add_argument('--rnn_hidden', type=int, default=50, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.')

parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd', type=float, default=0.01, help='weight decay.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax','adamw','adadelta'], default='adam', help='Optimizer: sgd, adagrad, adadelta ,adam, adamax, adamw')
parser.add_argument('--num_epoch', type=int, default=32, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_dir', type=str, default='./saved_models/Finance1', help='Root dir for saving models.')
parser.add_argument('--img_dir', type=str, default='./images/Finance1', help='Root dir for saving plots.')
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()


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
        self.lstm = nn.LSTM(embedding_size * 2, lstm_units, batch_first=True, dropout=args.lstm_dropout)
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
test_batch= DataLoader(args.data_dir + '/test.json', args.batch_size, args, vocab)

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

NUM_WORDS = 200000 ## MAx of words to keep, based on word frequency.
EMBEDDING_SIZE = args.hidden_dim ## the length of the Vector the will

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

Max_Len =  max([len(one_title) for one_title in text_X_train_tokenized])+5

text_X_train_padded = pad_sequences(text_X_train_tokenized, maxlen=Max_Len)
aspect_X_train_padded = pad_sequences(Aspect_X_train_tokenized, maxlen=1)
text_X_test_padded = pad_sequences(text_X_test_tokenized, maxlen=Max_Len)
aspect_X_test_padded = pad_sequences(Aspect_X_test_tokenized, maxlen=1)

text_X_train_padded = torch.tensor(text_X_train_padded, dtype=torch.long)
aspect_X_train_padded = torch.tensor(aspect_X_train_padded, dtype=torch.long)
text_X_test_padded = torch.tensor(text_X_test_padded, dtype=torch.long)
aspect_X_test_padded = torch.tensor(aspect_X_test_padded, dtype=torch.long)

# Create a DataLoader
train_dataset = TensorDataset(text_X_train_padded, aspect_X_train_padded)
test_dataset = TensorDataset(text_X_test_padded, aspect_X_test_padded)

train_loader = DataLoader1(train_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader1(test_dataset, batch_size=args.batch_size, shuffle=False)

# build model
model=CombinedModel(args,word_emb,NUM_WORDS,EMBEDDING_SIZE,Max_Len).to("cuda:0")

_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch_utils.get_optimizer(args.optim, _params, args.lr,args.wd)

# start training
train_acc_history, train_loss_history, train_precision_history, train_recall_history, train_f1_history= [0.], [1.8],[0.],[0.],[0.]
test_acc_history, test_loss_history, test_precision_history, test_recall_history, test_f1_history= [0.], [1.8],[0.],[0.],[0.]
epsilon = 1e-7  # small value to avoid division by zero

for epoch in range(1, args.num_epoch+1):
    print("Epoch no",epoch)
    print("\n")
    i,log_cnt=0,0
    train_final_acc,train_final_prec,train_final_rec,train_final_f1,train_final_loss=0,0,0,0,0
    test_final_acc,test_final_prec,test_final_rec,test_final_f1,test_final_loss=0,0,0,0,0
    n_correct=0
    n_total=0
    n_correct_test=0
    n_total_test=0
    #train
    for train_data, batch in zip(train_loader,train_batch):
        torch.autograd.set_detect_anomaly(True)
        bat=min(args.batch_size,len(batch[0]))
        #print(len(batch[0]),len(train_data[0]))
        
        batch = [b.cuda() for b in batch]
        train_data =  [t.cuda() for t in train_data]
        inputs = batch[0:8]
        label = batch[-1]
        
        target=F.one_hot(label, num_classes=args.num_class).float()
        #print(target)
        
        # step forward
        model.train()
        optimizer.zero_grad()
        logits,weight=model(inputs,train_data)
        #print(logits.shape)
        loss = F.cross_entropy(logits, label, reduction='mean')
        
        # backward
        loss.backward()
        optimizer.step()
        
        #print(logits)
        if(not(i%args.log_step)):
            log_cnt+=1
            out=torch.argmax(logits, -1)
            
            n_correct += (out == label).sum().item()
            n_total += bat
            acc = (n_correct / n_total)
            
            micro_precision = precision_score(out.cpu().detach().numpy(), label.cpu().detach().numpy(), average='micro')
            micro_recall = recall_score(out.cpu().detach().numpy(), label.cpu().detach().numpy(), average='micro')

            
            #micro_precision = tp.sum() / (tp.sum() + fp.sum() + epsilon)
            #micro_recall = tp.sum() / (tp.sum() + fn.sum() + epsilon)
            micro_f1 = f1_score(out.cpu().detach().numpy(), label.cpu().detach().numpy(), average='micro')
            # Calculate accuracy

            train_final_acc,train_final_loss,train_final_prec,train_final_rec,train_final_f1=train_final_acc+acc,train_final_loss+loss.item(),train_final_prec+micro_precision,train_final_rec+micro_recall,train_final_f1+micro_f1
            print("Accuracy is {:.4f}%  Loss is {:.4f} F1 is {:.4f}".format(acc*100,loss.item(),micro_f1))
            print()        
        i+=1
    
    train_final_acc,train_final_loss,train_final_prec,train_final_rec,train_final_f1=train_final_acc/log_cnt,train_final_loss/log_cnt,train_final_prec/log_cnt,train_final_rec/log_cnt,train_final_f1/log_cnt
    print("Accuracy is {:.4f}%  Loss is {:.4f} F1 is {:.4f}".format(train_final_acc*100,train_final_loss,train_final_f1))
    
    train_acc_history.append(train_final_acc*100)
    #train_loss_history.append(1-train_final_acc)
    train_loss_history.append(train_final_loss)

    train_precision_history.append(train_final_prec)
    train_recall_history.append(train_final_rec)
    train_f1_history.append(train_final_f1)
    
    #test
    print("\nValidating on test data...")
    j=0
    for test_data, batch_test in zip(test_loader,test_batch):
        bat_test=min(args.batch_size,len(batch_test[0]))
        batch_test = [b.cuda() for b in batch_test]
        # unpack inputs and label
        inputs_test = batch_test[0:8]
        label_test = batch_test[-1]

        test_data =  [t.cuda() for t in test_data]
        model.eval()
        logits_test,weight_test= model(inputs_test,test_data)
        loss_test = F.cross_entropy(logits_test, label_test, reduction='mean')
        out_test=torch.argmax(logits_test, -1)
        
        n_correct_test += (out_test == label_test).sum().item()
        n_total_test += bat_test
        acc_test = (n_correct_test / n_total_test)
        
        conf_matrix_test=torch.zeros(args.num_class,args.num_class)
        for t, p in zip(label_test.view(-1), out_test.view(-1)):
            conf_matrix_test[t.long(), p.long()] += 1

        # Calculate precision, recall, and F1 score
        tp_test = conf_matrix_test.diag()
        fp_test = conf_matrix_test.sum(dim=0) - tp_test
        fn_test = conf_matrix_test.sum(dim=1) - tp_test

        micro_precision_test = tp_test.sum() / (tp_test.sum() + fp_test.sum() + epsilon)
        micro_recall_test = tp_test.sum() / (tp_test.sum() + fn_test.sum() + epsilon)
        micro_f1_test = 2 * (micro_precision_test * micro_recall_test) / (micro_precision_test + micro_recall_test + epsilon)
        # Calculate accuracy
        
        test_final_acc,test_final_prec,test_final_rec,test_final_f1=test_final_acc+acc_test,test_final_prec+micro_precision_test.item(),test_final_rec+micro_recall_test.item(),test_final_f1+micro_f1_test.item()
        test_final_loss+=loss_test.item()
        j+=1
    
    test_final_acc,test_final_prec,test_final_rec,test_final_f1=(test_final_acc/j),test_final_prec/j,test_final_rec/j,test_final_f1/j
    test_final_loss/=j
    print("Accuracy is {:.4f}%  Loss is {:.4f} F1 is {:.4f}".format(test_final_acc*100,test_final_loss, test_final_f1))
    
    test_acc_history.append(test_final_acc*100)
    test_precision_history.append(test_final_prec)
    test_recall_history.append(test_final_rec)
    test_f1_history.append(test_final_f1) 
    #test_loss_history.append(1-test_final_acc)
    test_loss_history.append(test_final_loss)
    print("\n")
    print("___________________________________________________________")

print("Training ended with {} epochs.\n".format(epoch))

print("Maximum Train Accuracy: ", max(train_acc_history))
print("Maximum Test Accuracy: ", max(test_acc_history))
print("Minimum Train Loss: ", min(train_loss_history))
print("Minimum Test Loss: ", min(test_loss_history))
print("Maximum Train F1-Score: ", max(train_f1_history))
print("Maximum Test F1-Score: ", max(test_f1_history))

# check saved_models director
model_save_dir = args.save_dir
params = {
                'model': model.state_dict(),
                'config': args,
        }
try:
    filename=model_save_dir + '/best_model.pt'
    torch.save(params, filename)
    print("\nmodel saved to {}".format(filename))
except BaseException:
    print("\n[Warning: Saving failed... continuing anyway.]")

draw_curve(train_acc_history, test_acc_history,train_precision_history,train_recall_history, train_f1_history,test_precision_history,test_recall_history, test_f1_history,train_loss_history,test_loss_history, args)