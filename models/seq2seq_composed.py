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
from allennlp.modules.seq2seq_encoders import PytorchTransformer
from allennlp_models.rc.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp_models.generation.models import SimpleSeq2Seq, ComposedSeq2Seq
from allennlp_models.generation.modules import AutoRegressiveSeqDecoder, SeqDecoder
from allennlp_models.generation.modules import StackedSelfAttentionDecoderNet
from allennlp.data import DataLoader, PyTorchDataLoader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, AllennlpDataset
from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention
from allennlp.training.trainer import Trainer
from allennlp.training.trainer import GradientDescentTrainer
from allennlp_models.generation.predictors import Seq2SeqPredictor


parser = argparse.ArgumentParser()
parser.add_argument("-debug", action='store_true', help="if used, pdb will be used in breakpoints")
parser.add_argument("-pytorch-transformer", action='store_true', help="if used, PytorchTransformer will be used for the encoder")
parser.add_argument("-composed-seq2seq", action='store_true', help="if used, the seq2seq will be built with the ComposedSeq2Seq class")

parser.add_argument("-src-emb", required=False, help="embedding size of source inputs", default=128, type=int)
parser.add_argument("-tg-emb", required=False, help="embedding size of target inputs", default=128, type=int)
parser.add_argument("-hidden-dim", required=False, help="dimension of hidden layer", default=128, type=int)
parser.add_argument("-device", required=False, help="index of GPU", default=1, type=int)
parser.add_argument("-max-len", required=False, help="maximum length of generated output", default=40, type=int)
parser.add_argument("-epoch", required=False, help="number of epochs for training", default=50, type=int)
parser.add_argument("-beam", required=False, help="size of the beam for beam search decoding", default=3, type=int)
parser.add_argument("-dropout", required=False, help="dropout probability", default=0.2, type=float)
parser.add_argument("-drop-enc", required=False, help="encoder dropout rate, default 0.1", default=0.1, type=float)
parser.add_argument("-drop-dec", required=False, help="encoder dropout rate, default 0.1", default=0.1, type=float)

#parser.add_argument("-num-layers", required=False, help="number of layers of RNN", default=2, type=int)
parser.add_argument("-enc-layers", required=False, help="number of layer in the encoder, default 3", default=3, type=int)
parser.add_argument("-dec-layers", required=False, help="number of layer in the decoder, default 3", default=3, type=int)
parser.add_argument("-enc-heads", required=False, help="number of attention heads in the encoder, default 2", default=2, type=int)
parser.add_argument("-dec-heads", required=False, help="number of attention heads in the decoder, default 2", default=2, type=int)

parser.add_argument("-enc-pf", required=False, help="size of the hidden dim of the positional FF for the encoder, default 512", default=512, type=int)
parser.add_argument("-proj-dim", required=False, help="dimension of the linear projections for the self-attention layers, default 128", default=128, type=int)
parser.add_argument("-lr", required=False, help="learning rate, default 0.0005", default=0.0005, type=float)
parser.add_argument("-encoder", required=False, help="encoder type: LSTM or Transformer", default="Transformer", type=str)
parser.add_argument("-attention", required=False, help="attention type: dot, bilinear, linear", default="dot", type=str)
parser.add_argument("-itype", required=False, help="type of input: copy, set, exhaustive", default="set", type=str)
parser.add_argument("-otype", required=False, help="type of output: lex or delex", default="lex", type=str)

args = vars(parser.parse_args())

if args["attention"] not in {"dot", "linear", "bilinear"}:
    sys.exit("The -attenton argument must be either dot, bilinear, or linear")

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


reader = Seq2SeqDatasetReader(source_tokenizer=WhitespaceTokenizer(),target_tokenizer=WhitespaceTokenizer(),
                              source_token_indexers={'tokens': SingleIdTokenIndexer()},
                              target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})

train_dataset = reader.read(train_path)
validation_dataset = reader.read(val_path)
test_dataset = reader.read(test_path)
vocab = Vocabulary.from_instances(train_dataset + validation_dataset, min_count={'tokens': 1, 'target_tokens': 1})

train_dataset.index_with(vocab)
validation_dataset.index_with(vocab)


SRC_EMBEDDING_DIM = args["src_emb"]
TG_EMBEDDING_DIM = args["tg_emb"]
HIDDEN_DIM = args["hidden_dim"]

src_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             embedding_dim=SRC_EMBEDDING_DIM)
source_embedder = BasicTextFieldEmbedder({"tokens": src_embedding})

trg_embedding = Embedding(num_embeddings=vocab.get_vocab_size('target_tokens'),
                             embedding_dim=TG_EMBEDDING_DIM) # TODO
target_embedder = BasicTextFieldEmbedder({"target_tokens": trg_embedding})



enc_layers = args["enc_layers"]
dec_layers = args["dec_layers"]
enc_heads = args["enc_heads"]
dec_heads = args["dec_heads"]
ff_dim = args["enc_pf"]
proj_dim = args["proj_dim"]
enc_dropout = args["drop_enc"]
dec_dropout = args["drop_dec"]

encoder = StackedSelfAttentionEncoder(input_dim=SRC_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                                      projection_dim=proj_dim, feedforward_hidden_dim=ff_dim, num_layers=enc_layers,
                                      num_attention_heads=enc_heads, dropout_prob=enc_dropout, use_positional_encoding=True)

# if args["pytorch_transformer"]:
#     encoder = PytorchTransformer(input_dim=SRC_EMBEDDING_DIM, num_layers=enc_layers)


#attention = DotProductAttention()
max_decoding_steps = args["max_len"]
beam = args["beam"]
CUDA_DEVICE = 0



#encoder = PytorchTransformer(input_dim=SRC_EMBEDDING_DIM, num_layers=enc_layers)
print("input dimension %d , source embeding dimensionality %d" % (vocab.get_vocab_size('tokens'), SRC_EMBEDDING_DIM))
print("decoding dimension %d , target embeding dimensionality %d" % (vocab.get_vocab_size('target_tokens'), TG_EMBEDDING_DIM))
print("number of heads in the encoder %d , in the decoder %d" % (enc_heads, dec_heads))

decoder_net = StackedSelfAttentionDecoderNet(decoding_dim=TG_EMBEDDING_DIM, target_embedding_dim=TG_EMBEDDING_DIM,
                                             feedforward_hidden_dim=128, num_layers=dec_layers,
                                             num_attention_heads=dec_heads, use_positional_encoding=True, dropout_prob=dec_dropout)

decoder = AutoRegressiveSeqDecoder(vocab=vocab, decoder_net=decoder_net, max_decoding_steps=max_decoding_steps,
                                target_embedder=target_embedder, target_namespace='target_tokens', beam_size=beam)
#decoder = None
model = ComposedSeq2Seq(vocab=vocab, source_text_embedder=source_embedder,
                        encoder=encoder, decoder=decoder)

model = model.cuda(CUDA_DEVICE)


if args["debug"]:
    import pdb; pdb.set_trace()

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
    if args["debug"]:
        import pdb; pdb.set_trace()

    print(instance.fields["source_tokens"].tokens)
    print(results["predicted_tokens"])
    print("\n"*3)
    # gold target tokens: cut off the @start@ and @end@ symbol
    trg_tokens = instance.fields['target_tokens'].tokens[1:-1]
    # predicted tokens: take the first one: ordered by logloss, ascending
    pred_trg_tokens = [x for x in results["predicted_tokens"][0]]
    pred_trgs.append(pred_trg_tokens)
    tmp = [z.text for z in trg_tokens]
    trgs.append([tmp])

test_bleu = bleu_score(pred_trgs, trgs)
print(f'BLEU score (bleu-4 detokenized) on test data = {test_bleu*100:.2f}')


