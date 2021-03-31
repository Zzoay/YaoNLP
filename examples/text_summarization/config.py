

train_data_file = r"data/TTNewsCorpus_NLPCC2017/toutiao4nlpcc/train_with_summ.txt"
val_data_file = r"data/TTNewsCorpus_NLPCC2017/toutiao4nlpcc_eval/evaluation_with_ground_truth.txt"
vocab_file = r"data/TTNewsCorpus_NLPCC2017/vocab.txt"
model_file = r"examples/text_summarization/models/model.pt"
log_root = r"examples/text_summarization/log"

val_ratio = 0.2
batch_size = 64
shuffle = True

epochs = 2
prt_erery_step = 25
eval_every_step = 100
use_cuda = True

input_size = 0 
emb_dim = 300
hidden_size = 100
dropout = 0
use_coverage = True
coverage_loss_weight = 1.0
use_pgen = True

min_dec_steps = 35
beam_size = 4

max_dec_steps = 100