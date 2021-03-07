
import sys
sys.path.append(r".")   # add the YAONLP path

import argparse

from transformers import BertTokenizer

from data_helper import MyDataSet, TokenizedCollator, MyDataLoader
from model import AOPE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="2014Lap")
    parser.add_argument('--trick', type=str, default="Non")

    dataset_default = parser.get_default("dataset")
    trick_default = parser.get_default("trick")

    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])

    ## if test
    parser.add_argument('--test_model', type=str, default=f"./model/{dataset_default}/{trick_default}-final.model")
    parser.add_argument('--test_eval_dir', type=str, default=f"./test_eval/{dataset_default}")

    ## if train
    parser.add_argument('--model_dir', type=str, default=f"./model/{dataset_default}")
    parser.add_argument('--eval_dir', type=str, default=f"./eval/{dataset_default}")

    parser.add_argument('--bert_path', type=str, default=r"pretrained_model\bert-base-uncased")
    parser.add_argument('--bert_json_dir', 
                        type=str,
                        default=r"pretrained_model\bert-base-uncased\config.json")
    parser.add_argument('--bert_checkpoint_dir', 
                        type=str,
                        default=r"pretrained_model\bert-base-uncased\pytorch_model.bin")

    parser.add_argument('--tagScheme', type=str, default="BIO")
    parser.add_argument('--ifgpu', type=bool, default=False)

    parser.add_argument('--target_hidden_dim', type=int, default=250)
    parser.add_argument('--relation_hidden_dim', type=int, default=250)
    parser.add_argument('--relation_attention_dim', type=int, default=250)
    parser.add_argument('--relation_threds', type=float, default=0.1)
    parser.add_argument('--inference_threds', type=float, default=0.5)
    parser.add_argument('--iteration', type=int, default=70)
    parser.add_argument('--batchSize', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr_rate', type=float, default=0.001)
    parser.add_argument('--R_lr_rate', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.98)
    parser.add_argument('--step', type=int, default=2)  # step 2 in paper purposed
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--shuffle', type=bool, default=True)

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(r"pretrained_model\bert-base-uncased")
    train_dataset = MyDataSet(args=args, filename=r"data\SDRN_DATA\original_data\2014LapStandard.train")

    collator = TokenizedCollator(tokenizer, token_idx=0, label_idx=1, sort_key=lambda x:x[3])
    dataloader = MyDataLoader(train_dataset, args=args, collate_fn=collator)

    model = AOPE(args)
    for batch in dataloader:
        input_ids, segment_ids, input_masks, label_ids, relations, seq_len = batch

        # print(model(input_ids, segment_ids, input_masks))

        break