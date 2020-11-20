#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yaozz
# Date: 2020-11-10
# Remark:


"""
data format:
    data/:
        corpus_name/:
            train_data/:
                data:
                    "word1 word2 word3 ..."
                    ...
                labels:
                    "label1"
                    ...
            test_data/:
                data:
                    "word1 word2 word3 ..."
                    ...
                labels:
                    "label1"
                    ...
            vocab:
                "word1 freq1"
                "word2 freq2"
            ...

procedure:
    definition:
        hyper-parameter input
        model definition

    training procedure:
        input data -> feed into model and train -> save results

hyper-parameter:
    data:
        data_path

        train_dev_ratio

    model:
        optimizer:
            learning_rate

        loss_func

    metrics:
        accuracy
        f1

    train:
        epochs
        save_steps

pseudo-code:



"""


