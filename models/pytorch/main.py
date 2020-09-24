# tony-2, conda env pytorch14py36

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import random
import math
import time
import argparse

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("-src-emb", required=False, help="embedding size of source inputs", default=128, type=int)
parser.add_argument("-tg-emb", required=False, help="embedding size of target inputs", default=128, type=int)
parser.add_argument("-hidden-dim", required=False, help="dimension of hidden layer", default=128, type=int)
parser.add_argument("-device", required=False, help="index of GPU", default=1, type=int)
parser.add_argument("-max-len", required=False, help="maximum length of output", default=40, type=int)
parser.add_argument("-epoch", required=False, help="number of epochs for training", default=50, type=int)
parser.add_argument("-beam", required=False, help="size of the beam for beam search decoding", default=3, type=int)
parser.add_argument("-dropout", required=False, help="dropout probability", default=0.2, type=float)
parser.add_argument("-num-layers", required=False, help="number of layers of RNN", default=2, type=int)
parser.add_argument("-lr", required=False, help="learning rate", default=0.05, type=float)
parser.add_argument("-encoder", required=False, help="encoder type: LSTM or Transformer", default="Transformer", type=str)
parser.add_argument("-attention", required=False, help="attention type: dot, bilinear, linear", default="dot", type=str)
parser.add_argument("-itype", required=False, help="type of input: copy, set, exhaustive", default="set", type=str)
parser.add_argument("-otype", required=False, help="type of output: lex or delex", default="lex", type=str)

#parser.add_argument("-out", required=False, help="name of output file", default="corpora_v02/b01_delex")
args = vars(parser.parse_args())

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

SRC = Field(tokenize = tokenize_src,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)

TRG = Field(tokenize = tokenize_tg,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)


pth = "/home/CE/skrjanec/chart_descriptions/corpora_v02/keyvalue/complete/copy_tgt/mt/"

mt_train = TranslationDataset(
     path=pth + "train_a", exts=('.src', '.tgt'),
    fields=(SRC, TRG))

mt_dev = TranslationDataset(
     path=pth + "val_a", exts=('.src', '.tgt'),
    fields=(SRC, TRG))

SRC.build_vocab(mt_train, min_freq=2)
TRG.build_vocab(mt_train, min_freq=2)

train_iter = BucketIterator(dataset=mt_train, batch_size=32,
    sort_key=lambda x: torchtext.data.interleave_keys(len(x.src), len(x.trg)))