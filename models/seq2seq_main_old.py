"""
Seq2seq for the generation of bar chart summaries

Allennlp 1.0

Conda env allennlp_env  on tony-2
"""


# BELOW: allannlp 1.0 imports


import spacy
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

#
reader = Seq2SeqDatasetReader(source_tokenizer=WhitespaceTokenizer(),target_tokenizer=WhitespaceTokenizer(),source_token_indexers={'tokens': SingleIdTokenIndexer()},target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})
#reader = Seq2SeqDatasetReader(source_tokenizer=SpacyTokenizer(), target_tokenizer=SpacyTokenizer(),source_token_indexers={'tokens': SpacyTokenIndexer()},target_token_indexers={'tokens': SpacyTokenIndexer(namespace='target_tokens')})
home_dir = "/home/CE/skrjanec/chart_descriptions/corpora_v02/keyvalue/"
train_dataset = reader.read(home_dir+"keyvalue_train.txt")
validation_dataset = reader.read(home_dir+"keyvalue_val.txt")
vocab = Vocabulary.from_instances(train_dataset + validation_dataset, min_count={'tokens': 1, 'target_tokens': 1})

train_dataset.index_with(vocab)
validation_dataset.index_with(vocab)


SRC_EMBEDDING_DIM = 256
HIDDEN_DIM = 128
src_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             embedding_dim=SRC_EMBEDDING_DIM)
source_embedder = BasicTextFieldEmbedder({"tokens": src_embedding})

encoder = StackedSelfAttentionEncoder(input_dim=SRC_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                                      projection_dim=128, feedforward_hidden_dim=128, num_layers=1, num_attention_heads=8)

attention = DotProductAttention()
max_decoding_steps = 10
TG_EMBEDDING_DIM = 128
beam = 3
CUDA_DEVICE = 0

model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=TG_EMBEDDING_DIM,
                          target_namespace='target_tokens',
                          attention=attention,
                          beam_size=beam,
                          use_bleu=True)
model = model.cuda(CUDA_DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.06)

#ITERATOR = BucketBatchSampler(data_source=train_dataset, batch_size=32)
#train_it=BucketBatchSampler(train_dataset,batch_size=16)

train_data_loader = PyTorchDataLoader(train_dataset,batch_sampler=BucketBatchSampler(train_dataset,batch_size=16))
dev_data_loader = PyTorchDataLoader(validation_dataset,batch_sampler=BucketBatchSampler(validation_dataset,batch_size=16))
trainer = GradientDescentTrainer(model=model, optimizer=optimizer,data_loader=train_data_loader,
                                 validation_data_loader=dev_data_loader,num_epochs=3)




# useful https://colab.research.google.com/github/mhagiwara/realworldnlp/blob/master/examples/ner/ner.ipynb#scrollTo=SaRW0qPIypDm


# >>> trainer = GradientDescentTrainer(model=model, optimizer=optimizer,data_loader=ITERATOR)
# >>>
# >>> trainer.train()
#   0%|          | 0/12 [00:00<?, ?it/s]
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/home/CE/skrjanec/anaconda3/lib/python3.6/site-packages/allennlp/training/trainer.py", line 867, in train
#     train_metrics = self._train_epoch(epoch)
#   File "/home/CE/skrjanec/anaconda3/lib/python3.6/site-packages/allennlp/training/trainer.py", line 560, in _train_epoch
#     for batch_group in batch_group_generator_tqdm:
#   File "/home/CE/skrjanec/anaconda3/lib/python3.6/site-packages/tqdm/std.py", line 1133, in __iter__
#     for obj in iterable:
#   File "/home/CE/skrjanec/anaconda3/lib/python3.6/site-packages/allennlp/common/util.py", line 135, in lazy_groups_of
#     s = list(islice(iterator, group_size))
#   File "/home/CE/skrjanec/anaconda3/lib/python3.6/site-packages/allennlp/data/samplers/bucket_batch_sampler.py", line 119, in __iter__
#     indices, _ = self._argsort_by_padding(self.data_source)
#   File "/home/CE/skrjanec/anaconda3/lib/python3.6/site-packages/allennlp/data/samplers/bucket_batch_sampler.py", line 94, in _argsort_by_padding
#     self._guess_sorting_keys(instances)
#   File "/home/CE/skrjanec/anaconda3/lib/python3.6/site-packages/allennlp/data/samplers/bucket_batch_sampler.py", line 147, in _guess_sorting_keys
#     instance.index_fields(self.vocab)
#   File "/home/CE/skrjanec/anaconda3/lib/python3.6/site-packages/allennlp/data/instance.py", line 75, in index_fields
#     field.index(vocab)
#   File "/home/CE/skrjanec/anaconda3/lib/python3.6/site-packages/allennlp/data/fields/text_field.py", line 68, in index
#     self._indexed_tokens[indexer_name] = indexer.tokens_to_indices(self.tokens, vocab)
#   File "/home/CE/skrjanec/anaconda3/lib/python3.6/site-packages/allennlp/data/token_indexers/single_id_token_indexer.py", line 92, in tokens_to_indices
#     indices.append(vocabulary.get_token_index(text, self.namespace))
# AttributeError: 'NoneType' object has no attribute 'get_token_index

# fix trainer https://docs.allennlp.org/master/api/training/trainer/

"""
Useful
https://docs.allennlp.org/models/master/models/rc/modules/seq2seq_encoders/stacked_self_attention/
http://docs.allennlp.org/master/api/predictors/predictor/
https://guide.allennlp.org/representing-text-as-features#6


"""