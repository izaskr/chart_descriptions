"""
Seq2seq for the generation of bar chart summaries

Allennlp 1.0

Conda env allennlp_env  on tony-2
"""


# BELOW: allannlp 1.0 imports

import argparse
import sys
import spacy
from torchtext.data.metrics import bleu_score
import torch.optim as optim
from allennlp_models.generation import Seq2SeqDatasetReader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.tokenizers import WhitespaceTokenizer, SpacyTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, SpacyTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp_models.rc.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp_models.generation.models import SimpleSeq2Seq
from allennlp.data import DataLoader, PyTorchDataLoader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, AllennlpDataset
from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention
from allennlp.training.trainer import Trainer
from allennlp.training.trainer import GradientDescentTrainer
from allennlp_models.generation.predictors import Seq2SeqPredictor


parser = argparse.ArgumentParser()
parser.add_argument("-src-emb", required=False, help="embedding size of source inputs", default=128, type=int)
parser.add_argument("-tg-emb", required=False, help="embedding size of target inputs", default=128, type=int)
parser.add_argument("-hidden-dim", required=False, help="dimension of hidden layer", default=128, type=int)
parser.add_argument("-device", required=False, help="index of GPU", default=1, type=int)
parser.add_argument("-max-len", required=False, help="maximum length of generated output", default=40, type=int)
parser.add_argument("-epoch", required=False, help="number of epochs for training", default=50, type=int)
parser.add_argument("-beam", required=False, help="size of the beam for beam search decoding", default=3, type=int)
parser.add_argument("-dropout", required=False, help="dropout probability", default=0.2, type=float)
parser.add_argument("-drop-enc", required=False, help="encoder dropout rate, default 0.1", default=0.1, type=float)
#parser.add_argument("-num-layers", required=False, help="number of layers of RNN", default=2, type=int)
parser.add_argument("-enc-layers", required=False, help="number of layer in the encoder, default 3", default=3, type=int)
parser.add_argument("-dec-layers", required=False, help="number of layer in the decoder, default 3", default=3, type=int)
parser.add_argument("-enc-heads", required=False, help="number of attention heads in the encoder, default 8", default=8, type=int)
parser.add_argument("-enc-pf", required=False, help="size of the hidden dim of the positional FF for the encoder, default 512", default=512, type=int)
parser.add_argument("-proj-dim", required=False, help="dimension of the linear projections for the self-attention layers, default 128", default=128, type=int)
parser.add_argument("-lr", required=False, help="learning rate, default 0.0005", default=0.0005, type=float)
parser.add_argument("-encoder", required=False, help="encoder type: LSTM or Transformer", default="Transformer", type=str)
parser.add_argument("-attention", required=False, help="attention type: dot, bilinear, linear", default="dot", type=str)
parser.add_argument("-itype", required=False, help="type of input: copy, set, exhaustive", default="set", type=str)
parser.add_argument("-otype", required=False, help="type of output: lex or delex", default="lex", type=str)

#parser.add_argument("-out", required=False, help="name of output file", default="corpora_v02/b01_delex")
args = vars(parser.parse_args())

if args["itype"] not in {"copy", "set", "exhaustive"}:
    sys.exit("The -itype argument must be either copy, set, or exhaustive")

if args["otype"] not in {"lex", "delex"}:
    sys.exit("The -otype argument must be either lex or delex")

# data map: itype - otype
map = {"copy":{"lex":"a", "delex":"b", "p":"copy_tgt"}, "set": {"lex":"c", "delex":"d","p":"copy_tgt_set"},
       "exhaustive":{"lex":"e", "delex":"f", "p":"exhautive"}}

in_type, out_type = args["itype"], args["otype"]
pth = "/home/CE/skrjanec/chart_descriptions/corpora_v02/keyvalue/complete/"
folder_pth = pth + map[in_type]["p"] + "/"

# tab_train_b.txt, tab_val_b.txt, tab_test_b.txt
# map[in_type][out_type] # a ... f
train_path = folder_pth + "tab_train_" + map[in_type][out_type] + ".txt"
val_path = folder_pth + "tab_val_" + map[in_type][out_type] + ".txt"
test_path = folder_pth + "tab_test_" + map[in_type][out_type] + ".txt"



reader = Seq2SeqDatasetReader(source_tokenizer=WhitespaceTokenizer(),target_tokenizer=WhitespaceTokenizer(),source_token_indexers={'tokens': SingleIdTokenIndexer()},target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})
#reader = Seq2SeqDatasetReader(source_tokenizer=SpacyTokenizer(), target_tokenizer=SpacyTokenizer(),source_token_indexers={'tokens': SpacyTokenIndexer()},target_token_indexers={'tokens': SpacyTokenIndexer(namespace='target_tokens')})
# home_dir = "/home/CE/skrjanec/chart_descriptions/corpora_v02/keyvalue/"
# train_dataset = reader.read(home_dir+"keyvalue_train.txt")
# validation_dataset = reader.read(home_dir+"keyvalue_val.txt")
train_dataset = reader.read(train_path)
validation_dataset = reader.read(val_path)
test_dataset = reader.read(test_path)
vocab = Vocabulary.from_instances(train_dataset + validation_dataset, min_count={'tokens': 1, 'target_tokens': 1})

train_dataset.index_with(vocab)
validation_dataset.index_with(vocab)


SRC_EMBEDDING_DIM = args["src_emb"]
HIDDEN_DIM = args["hidden_dim"]

src_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             embedding_dim=SRC_EMBEDDING_DIM)
source_embedder = BasicTextFieldEmbedder({"tokens": src_embedding})

# class StackedSelfAttentionEncoder(Seq2SeqEncoder):
#  | def __init__(
#  |     self,
#  |     input_dim: int,
#  |     hidden_dim: int,
#  |     projection_dim: int,
#  |     feedforward_hidden_dim: int,
#  |     num_layers: int,
#  |     num_attention_heads: int,
#  |     use_positional_encoding: bool = True,
#  |     dropout_prob: float = 0.1,
#  |     residual_dropout_prob: float = 0.2,
#  |     attention_dropout_prob: float = 0.1
#  | ) -> None

# projection_dim : int
# The dimension of the linear projections for the self-attention layers.

enc_layers = args["enc_layers"]
dec_layers = args["dec_layers"]
enc_heads = args["enc_heads"]
ff_dim = args["enc_pf"]
proj_dim = args["proj_dim"]
enc_dropout = args["drop_enc"]

encoder = StackedSelfAttentionEncoder(input_dim=SRC_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                                      projection_dim=proj_dim, feedforward_hidden_dim=ff_dim, num_layers=enc_layers,
                                      num_attention_heads=enc_heads, dropout_prob=enc_dropout)

attention = DotProductAttention()
max_decoding_steps = args["max_len"]
TG_EMBEDDING_DIM = args["tg_emb"]
beam = args["beam"]
CUDA_DEVICE = 0

# class SimpleSeq2Seq(Model):
#  | def __init__(
#  |     self,
#  |     vocab: Vocabulary,
#  |     source_embedder: TextFieldEmbedder,
#  |     encoder: Seq2SeqEncoder,
#  |     max_decoding_steps: int,
#  |     attention: Attention = None,
#  |     beam_size: int = None,
#  |     target_namespace: str = "tokens",
#  |     target_embedding_dim: int = None,
#  |     scheduled_sampling_ratio: float = 0.0,
#  |     use_bleu: bool = True,
#  |     bleu_ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25),
#  |     target_pretrain_file: str = None,
#  |     target_decoder_layers: int = 1
#  | ) -> None


model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=TG_EMBEDDING_DIM,
                          target_namespace='target_tokens',
                          attention=attention,
                          beam_size=beam,
                          use_bleu=True, target_decoder_layers=dec_layers)

model = model.cuda(CUDA_DEVICE)
LR = args["lr"]
optimizer = optim.Adam(model.parameters(), lr=LR)

#ITERATOR = BucketBatchSampler(data_source=train_dataset, batch_size=32)
#train_it=BucketBatchSampler(train_dataset,batch_size=16)

train_data_loader = PyTorchDataLoader(train_dataset,batch_sampler=BucketBatchSampler(train_dataset,batch_size=16))
dev_data_loader = PyTorchDataLoader(validation_dataset,batch_sampler=BucketBatchSampler(validation_dataset,batch_size=16))
trainer = GradientDescentTrainer(model=model, optimizer=optimizer,data_loader=train_data_loader,
                                 validation_data_loader=dev_data_loader,num_epochs=3)

# run the training
trainer.train()
# define the predictor
predictor = Seq2SeqPredictor(model, reader)

trgs, pred_trgs = [], []

for instance in test_dataset:
    results = predictor.predict_instance(instance)
    # ['predictions', 'loss', 'class_log_probabilities', 'predicted_tokens']
    #import pdb; pdb.set_trace()

    print(instance.fields["source_tokens"].tokens)
    print(results["predicted_tokens"])
    print("\n"*3)
    # gold target tokens: cut off the @start@ and @end@ symbol
    trg_tokens = instance.fields['target_tokens'].tokens[1:-1]
    # predicted tokens: take the first one: ordered by logloss, ascending
    pred_trg_tokens = results["predicted_tokens"][0]
    pred_trgs.append(pred_trg_tokens)
    trgs.append([trg_tokens])

test_bleu = bleu_score(pred_trgs, trgs)
print(f'BLEU score (bleu-4 detokenized) on test data = {bleu_score*100:.2f}')



# instance.fields["source_tokens"].tokens
# [@start@, YLABEL[percentage, of, women, representation],, XHIGHEST[insurance],, YHIGHESTAPPROX[63],,
# YUNIT[%],, XLEAST[law, firm],, YLEAST[35], @end@]

 # instance.fields['target_tokens'].tokens
# [@start@, This, chart, shows, a, percentage, of, women, representation, in, different, sectors, in, Benoni, .,
# \\n, From, the, chart, we, see, that, the, highest, percentage, of, women, representation, is, in, insurance, at,
# 63, %, ., We, can, see, that, the, lowest, percentage, of, women, representation, is, in, law, firm,
# at, 35, %, @end@]

#bleu_score(pred_trgs, trgs)
# pred_trgs - a list, where wach prediction is a list of tokens; pred_trgs = [ ["A", "c"], ["c", "b"] ]
# trgs - a list, where each golden target is a list of tokens - in a list; trgs = [ [["a", "b"]], [["c", "d"]] ]


# TODO: defualt setting raise an error
# RuntimeError: rnn: hx is not contiguous
# these don't
# python seq2seq_main.py -epoch 3 -dec-layers 1 -enc-heads 3 -enc-layers 3 -enc-pf 128 -proj-dim 300

# useful https://colab.research.google.com/github/mhagiwara/realworldnlp/blob/master/examples/ner/ner.ipynb#scrollTo=SaRW0qPIypDm


"""
Useful
https://docs.allennlp.org/models/master/models/rc/modules/seq2seq_encoders/stacked_self_attention/
http://docs.allennlp.org/master/api/predictors/predictor/
https://guide.allennlp.org/representing-text-as-features#6


"""