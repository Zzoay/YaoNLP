

train_data_file = r"data\TTNewsCorpus_NLPCC2017\toutiao4nlpcc\train_with_summ.txt"
val_data_file = r"data\TTNewsCorpus_NLPCC2017\toutiao4nlpcc_eval\evaluation_with_ground_truth.txt"
vocab_file = r"data\TTNewsCorpus_NLPCC2017\vocab.txt"
model_file = r"examples\text_summarization\model.pt"

val_ratio = 0.2
batch_size = 8
shuffle = True

epochs = 1
prt_erery_step = 25
eval_every_step = 100
use_cuda = True

input_size = 0 
emb_dim = 300
hidden_size = 100
dropout = 0
coverage_loss_weight = 1.0