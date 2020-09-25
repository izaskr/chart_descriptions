# tony-2, conda env torch14py36
# base tutorial https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, Iterator
from torchtext.datasets import TranslationDataset
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import random
import math
import time
import argparse
from model_components import *

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()

parser.add_argument("-debug", action='store_true', help="if used, pdb will be used in breakpoints")


parser.add_argument("-src-emb", required=False, help="embedding size of source inputs", default=128, type=int)
# parser.add_argument("-tg-emb", required=False, help="embedding size of target inputs", default=128, type=int)
# parser.add_argument("-hidden-dim", required=False, help="dimension of hidden layer", default=128, type=int)
# parser.add_argument("-device", required=False, help="index of GPU", default=1, type=int)
# parser.add_argument("-max-len", required=False, help="maximum length of output", default=40, type=int)
# parser.add_argument("-epoch", required=False, help="number of epochs for training", default=50, type=int)
# parser.add_argument("-beam", required=False, help="size of the beam for beam search decoding", default=3, type=int)
# parser.add_argument("-dropout", required=False, help="dropout probability", default=0.2, type=float)
# parser.add_argument("-num-layers", required=False, help="number of layers of RNN", default=2, type=int)
# parser.add_argument("-lr", required=False, help="learning rate", default=0.05, type=float)
# parser.add_argument("-encoder", required=False, help="encoder type: LSTM or Transformer", default="Transformer", type=str)
# parser.add_argument("-attention", required=False, help="attention type: dot, bilinear, linear", default="dot", type=str)
# parser.add_argument("-itype", required=False, help="type of input: copy, set, exhaustive", default="set", type=str)
# parser.add_argument("-otype", required=False, help="type of output: lex or delex", default="lex", type=str)

#parser.add_argument("-out", required=False, help="name of output file", default="corpora_v02/b01_delex")
args = vars(parser.parse_args())
"""
### TODO below
SRC_EMBEDDING_DIM = args["src_emb"]
TG_EMBEDDING_DIM = args["tg_emb"]
HIDDEN_DIM = args["hidden_dim"]
CUDA_DEVICE = args["device"]
max_decoding_steps = args["max_len"]
n_epoch = args["epoch"]
beam = args["beam"]
dropout = args["dropout"]
num_layers = args["num_layers"]
lr = args["lr"]
# SRC_EMBEDDING_DIM = 128 # source
# TG_EMBEDDING_DIM = 128 # target
# HIDDEN_DIM = 128
# CUDA_DEVICE = 0

enc = args["encoder"]
if enc.lower() not in {"lstm", "transformer"}:
    sys.exit("The -encoder argument must be either LSTM or Transformer")

if args["attention"] not in {"dot", "linear", "bilinear"}:
    sys.exit("The -attenton argument must be either dot, bilinear, or linear")

if args["itype"] not in {"copy", "set", "exhaustive"}:
    sys.exit("The -itype argument must be either copy, set, or exhaustive")

if args["otype"] not in {"lex", "delex"}:
    sys.exit("The -otype argument must be either lex or delex")

# data map: itype - otype
map = {"copy":{"lex":"a", "delex":"b", "p":"copy_tgt"}, "set": {"lex":"c", "delex":"d","p":"copy_tgt_set"},
       "exhaustive":{"lex":"e", "delex":"f", "p":"exhautive"}}

out_general = args["otype"] # lex or delex
outtype = map[args["itype"]][out_general] # a, b ... f

# data dir path + name
data_dir = "/home/CE/skrjanec/chart_descriptions/corpora_v02/keyvalue/complete/" + map[args["itype"]]["p"] + "/map/" + "tab_"
# train: + "train_" + outtype + ".txt"
# val: + "val_" + outtype + ".txt"
# test: + "test_" + outtype + ".txt"

### TODO above

"""

def tokenize_src(text):
    """
    Tokenizes source text from a string of key-value pairs into a list of strings
    'key[value], key2[value2]' into ['key[value]', 'key2[value2]']
    """
    s = text.split(", ")
    return s

def tokenize_tg(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return text.split()


# test_b.src  train_a.src  train_b.src  val_a.src
MAX_LEN = 100
SRC = Field(tokenize = tokenize_src,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True, use_vocab=True, fix_length=MAX_LEN)

TRG = Field(tokenize = tokenize_tg,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True, sequential=True, use_vocab=True, fix_length=MAX_LEN)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu") # TODO
pth = "/home/CE/skrjanec/chart_descriptions/corpora_v02/keyvalue/complete/copy_tgt/mt/"

mt_train = TranslationDataset(
     path=pth + "train_a", exts=('.src', '.tgt'),
    fields=(SRC, TRG))

mt_dev = TranslationDataset(
     path=pth + "val_a", exts=('.src', '.tgt'),
    fields=(SRC, TRG))

mt_test = TranslationDataset(
     path=pth + "test_a", exts=('.src', '.tgt'),
    fields=(SRC, TRG))

SRC.build_vocab(mt_train, min_freq=2)
TRG.build_vocab(mt_train, min_freq=2)

train_iter = BucketIterator(dataset=mt_train, batch_size=32,
    sort_key=lambda x: torchtext.data.interleave_keys(len(x.src), len(x.trg)), device=device)

dev_iter = BucketIterator(dataset=mt_dev, batch_size=32,
    sort_key=lambda x: torchtext.data.interleave_keys(len(x.src), len(x.trg)), device=device)

test_iter = Iterator(dataset=mt_test, batch_size=32, shuffle=False, device=device)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

src_VOCAB_SIZE = len(SRC.vocab)
tgt_VOCAB_SIZE = len(TRG.vocab)

print("size of source and target vocabs", src_VOCAB_SIZE, tgt_VOCAB_SIZE)

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device)

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

if args["debug"]:
    import pdb; pdb.set_trace()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


model.apply(initialize_weights)

LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        #print("--- types and shapes of source an target batch", type(src), type(trg), src.shape, trg.shape)
        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])
        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output, _ = model(src, trg[:, :-1])
            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, dev_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

"""

RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`

This error message is not very informative

"""