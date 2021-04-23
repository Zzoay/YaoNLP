

train_data_file = r"data/TTNewsCorpus_NLPCC2017/toutiao4nlpcc/train_with_summ.txt"
val_data_file = r"data/TTNewsCorpus_NLPCC2017/toutiao4nlpcc_eval/evaluation_with_ground_truth.txt"
vocab_file = r"data/TTNewsCorpus_NLPCC2017/vocab.txt"
# train_data_file = r"/home/jgy/YaoNLP/data/TTNews_Processed/train.txt"
# val_data_file = r"/home/jgy/YaoNLP/data/TTNews_Processed/val.txt"
# vocab_file = r"data/TTNews_Processed/vocab.txt"
model_file = r"examples/text_summarization/models/model.pt"
log_root = r"examples/text_summarization/log"

random_seed = 1314

val_ratio = 0.2
batch_size = 16
shuffle = True

max_summ_len = 100
max_article_len = 900

epochs = 20
prt_erery_step = 25
eval_every_step = 200
use_cuda = True

max_grad_norm = 5
emb_dim = 300
hidden_size = 200
dropout = 0
use_coverage = True
coverage_loss_weight = 1.0
use_pgen = True

min_dec_steps = 35
beam_size = 4

max_dec_steps = 100

use_syn_enhanced = True
parser_vocab_file = '/home/jgy/YaoNLP/data/ctb8.0/vocab/word_vocab.txt'

use_bert_enhanced = False
bert_tokenizer_file = "/home/jgy/YaoNLP/pretrained_model/bert-base-uncased"