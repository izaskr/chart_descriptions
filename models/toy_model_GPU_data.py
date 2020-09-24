"""
Code for preparing data and training a sequence-to-sequence model

Input sequence: content plan
Output sequence: chart summary

"""


import itertools
import argparse
import sys
from comet_ml import Experiment
import torch
import torch.optim as optim
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.activations import Activation
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SimpleSeq2SeqPredictor
from allennlp.training.trainer import Trainer
from tensorboardX import SummaryWriter


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
data_dir = "/home/CE/skrjanec/chart_descriptions/corpora_v02/keyvalue/complete/" + map[args["itype"]]["p"] + "/tab_"
# train: + "train_" + outtype + ".txt"
# val: + "val_" + outtype + ".txt"
# test: + "test_" + outtype + ".txt"

# TODO check the model arch below

### COMET ML CONFIGURATION ###


hyperparameters = {"source_emb_size":SRC_EMBEDDING_DIM, "target_emb_size":TG_EMBEDDING_DIM,
                   "hidden_layer_RNN_size":HIDDEN_DIM, "num_layers_RNN":num_layers,
                   "max_length":max_decoding_steps, "epochs":n_epoch, "beam":beam, "dropout":dropout,
                   "optimizer":"adam", "model_type":"dot_attention_seq2seq_LSTM", "LR":lr} # NOTE change the model type

#SRC_EMBEDDING_DIM = 256 # source
#TG_EMBEDDING_DIM = 256 # target
#HIDDEN_DIM = 128
#CUDA_DEVICE = 0

def main(topicID):

    print("\t Current topic", topicID)

    ## TRACKING EXPERIMENTS WITH COMET ML ##
    experiment = Experiment(api_key="Vnua3GA829lW6sM60FNYOPStH",
                            project_name="data", workspace="izaskr")

    #hyperparameters = {"source_emb_size":SRC_EMBEDDING_DIM, "target_emb_size":TG_EMBEDDING_DIM,
    #               "hidden_layer_RNN_size":HIDDEN_DIM, "num_layers_RNN":num_layers,
    #               "max_length":max_decoding_steps, "epochs":n_epoch, "beam":beam, "dropout":dropout,
    #               "optimizer":"adam", "model_type":"vanilla_seq2seq_LSTM"}
    experiment.log_parameters(hyperparameters)
    #experiment.add_tags(["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14"])
    #experiment.add_tag(topicID)

    # use the AllenNLP parallel data reader
    reader = Seq2SeqDatasetReader(
        source_tokenizer=WordTokenizer(),
        target_tokenizer=WordTokenizer(),
        source_token_indexers={'tokens': SingleIdTokenIndexer()},
        target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})
    home_dir = "/home/CE/skrjanec/chart_descriptions/corpora_v02/keyvalue/"
    train_dataset = reader.read(home_dir+"keyvalue_train.txt")
    validation_dataset = reader.read(home_dir+"keyvalue_val.txt")

    vocab = Vocabulary.from_instances(train_dataset + validation_dataset,
                                      min_count={'tokens': 1, 'target_tokens': 1})

    src_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             embedding_dim=SRC_EMBEDDING_DIM)
    #encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=SRC_EMBEDDING_DIM, hidden_size=HIDDEN_DIM, num_layers=num_layers, batch_first=True, dropout=dropout))
    encoder = StackedSelfAttentionEncoder(input_dim=SRC_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, projection_dim=128, feedforward_hidden_dim=128, num_layers=1, num_attention_heads=8)

    source_embedder = BasicTextFieldEmbedder({"tokens": src_embedding})

    #attention = LinearAttention(HIDDEN_DIM, HIDDEN_DIM, activation=Activation.by_name('tanh')())
    #attention = BilinearAttention(HIDDEN_DIM, HIDDEN_DIM, activation=Activation.by_name('tanh')()) # default: no activation
    attention = DotProductAttention()

    #max_decoding_steps = 40   # DONE: make this variable # Maximum length of decoded sequences
    # model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
    #                       target_embedding_dim=TG_EMBEDDING_DIM,
    #                       target_namespace='target_tokens',
    #                       beam_size=beam,
    #                       use_bleu=True)
    model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=TG_EMBEDDING_DIM,
                          target_namespace='target_tokens',
                          attention=attention,
                          beam_size=beam,
                          use_bleu=True) # has attention

    model = model.cuda(CUDA_DEVICE) # NOTE else error: why? we put the Trainer onto CUDA already
    optimizer = optim.Adam(model.parameters(), lr=lr)
    iterator = BucketIterator(batch_size=32, sorting_keys=[("source_tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      num_epochs=n_epoch,
                      cuda_device=CUDA_DEVICE)
                      #,serialization_dir="logging")
    """
    trainer.train()
    predictor = SimpleSeq2SeqPredictor(model, reader)
    for instance in itertools.islice(validation_dataset, 1):
        print('SOURCE:', instance.fields['source_tokens'].tokens)
        print('GOLD:', instance.fields['target_tokens'].tokens)
        print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])
    """
    # Tensorboard logger
    #writer = SummaryWriter('runs/exp-1')
    for i in range(1): # DONE make a variable
        print('Epoch: {}'.format(i))
        metrics = trainer.train()
        #for x,v in metrics.items(): print("******* METRICS",x,v)

        print("METRICS")
        print("\t", "Train_loss", metrics["training_loss"])
        print("\t", "Validation_loss", metrics["validation_loss"])
        print("\t", "Validation_BLEU", metrics["validation_BLEU"])
        #print("Logging onto comet ml")
        #experiment.log_metric("Train_loss", metrics["training_loss"], step=i)
        #experiment.log_metric("Validation_loss", metrics["validation_loss"], step=i)
        #experiment.log_metric("Validation_BLEU",metrics["validation_BLEU"], step=i)
        # writer.add_scalar('Training loss', metrics["training_loss"], i)
        # writer.add_scalar("Validation loss",metrics["validation_loss"], i)
        # writer.add_scalar("Validation BLEU",metrics["validation_BLEU"], i)
        #print("*"*10, "PRINTING METRICS",model.get_metrics())
        predictor = SimpleSeq2SeqPredictor(model, reader)

        for instance in itertools.islice(validation_dataset, 10):
            print('SOURCE:', instance.fields['source_tokens'].tokens)
            print('GOLD:', instance.fields['target_tokens'].tokens)
            print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])
    
    return None

if __name__ == '__main__':

    #all_topicIDs = ["01", "02", "03"] #, "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14"]
    #all_topicIDs = ["01","02","03","04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14"]
    #all_topicIDs = ["01","02"]
    main("0")
    # for t in all_topicIDs:
    #
    #     # run the training
    #     main(t)
