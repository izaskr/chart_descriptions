"""
Code for preparing data and training a sequence-to-sequence model

Input sequence: content plan
Output sequence: chart summary

"""


import itertools
import argparse
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
parser.add_argument("-device", required=False, help="index of GPU", default=0, type=int)
parser.add_argument("-max-len", required=False, help="maximum length of output", default=40, type=int)
parser.add_argument("-epoch", required=False, help="number of epochs for training", default=50, type=int)
parser.add_argument("-beam", required=False, help="size of the beam for beam search decoding", default=5, type=int)
parser.add_argument("-dropout", required=False, help="dropout probability", default=0.2, type=float)
parser.add_argument("-num-layers", required=False, help="number of layers of RNN", default=2, type=int)
parser.add_argument("-lr", required=False, help="learning rate", default=0.01, type=float)

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



### COMET ML CONFIGURATION ###
experiment = Experiment(api_key="Vnua3GA829lW6sM60FNYOPStH",
                            project_name="charts_seq2seq", workspace="izaskr")

hyperparameters = {"source_emb_size":SRC_EMBEDDING_DIM, "target_emb_size":TG_EMBEDDING_DIM,
                   "hidden_layer_RNN_size":HIDDEN_DIM, "num_layers_RNN":num_layers,
                   "max_length":max_decoding_steps, "epochs":n_epoch, "beam":beam, "dropout":dropout,
                   "optimizer":"adam", "model_type":"vanilla_seq2seq_LSTM"}
experiment.log_parameters(hyperparameters)
experiment.add_tag("03_01")


def main():
    reader = Seq2SeqDatasetReader(
        source_tokenizer=WordTokenizer(),
        target_tokenizer=WordTokenizer(),
        source_token_indexers={'tokens': SingleIdTokenIndexer()},
        target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})
    home_dir = "/home/CE/skrjanec/"
    train_dataset = reader.read(home_dir+'chart_descriptions/corpora_v02/delexicalized/delex_03_01_train.txt')
    validation_dataset = reader.read(home_dir+'chart_descriptions/corpora_v02/delexicalized/delex_03_01_val.txt')

    vocab = Vocabulary.from_instances(train_dataset + validation_dataset,
                                      min_count={'tokens': 1, 'target_tokens': 1})

    src_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             embedding_dim=SRC_EMBEDDING_DIM)
    encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=SRC_EMBEDDING_DIM, hidden_size=HIDDEN_DIM, num_layers=num_layers, batch_first=True, dropout=dropout))
    #encoder = StackedSelfAttentionEncoder(input_dim=SRC_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, projection_dim=128, feedforward_hidden_dim=128, num_layers=1, num_attention_heads=8)

    source_embedder = BasicTextFieldEmbedder({"tokens": src_embedding})

    # attention = LinearAttention(HIDDEN_DIM, HIDDEN_DIM, activation=Activation.by_name('tanh')())
    # attention = BilinearAttention(HIDDEN_DIM, HIDDEN_DIM)
    attention = DotProductAttention()

    #max_decoding_steps = 40   # DONE: make this variable # Maximum length of decoded sequences
    model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=TG_EMBEDDING_DIM,
                          target_namespace='target_tokens',
                          beam_size=beam,
                          use_bleu=True)
    # model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
    #                       target_embedding_dim=TG_EMBEDDING_DIM,
    #                       target_namespace='target_tokens',
    #                       attention=attention,
    #                       beam_size=8,
    #                       use_bleu=True) # has attention

    model = model.cuda(CUDA_DEVICE) # NOTE else error? 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    iterator = BucketIterator(batch_size=2, sorting_keys=[("source_tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      num_epochs=1,
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
    for i in range(n_epoch): # DONE make a variable
        print('Epoch: {}'.format(i))
        metrics = trainer.train()
        for x,v in metrics.items(): print("******* METRICS",x,v)

        print("Logging onto comet ml")
        experiment.log_metric("Train_loss", metrics["training_loss"], step=i)
        experiment.log_metric("Validation_loss", metrics["validation_loss"], step=i)
        experiment.log_metric("Validation_BLEU",metrics["validation_BLEU"], step=i)
        # writer.add_scalar('Training loss', metrics["training_loss"], i)
        # writer.add_scalar("Validation loss",metrics["validation_loss"], i)
        # writer.add_scalar("Validation BLEU",metrics["validation_BLEU"], i)
        #print("*"*10, "PRINTING METRICS",model.get_metrics())
        predictor = SimpleSeq2SeqPredictor(model, reader)

        #for instance in itertools.islice(validation_dataset, 1):
        #    print('SOURCE:', instance.fields['source_tokens'].tokens)
        #    print('GOLD:', instance.fields['target_tokens'].tokens)
        #    print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])
    
    return None

if __name__ == '__main__':
    main()
