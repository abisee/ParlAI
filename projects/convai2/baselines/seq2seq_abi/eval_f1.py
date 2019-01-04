#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Evaluate pre-trained model trained for f1 metric.
This seq2seq model was trained on convai2:self.
"""

from parlai.core.build_data import download_models
from projects.convai2.eval_f1 import setup_args, eval_f1


if __name__ == '__main__':
    parser = setup_args()
    parser.set_params(
        model='parlai_internal.projects.controllable_dialog.controllable_seq2seq.controllable_seq2seq:ControllableSeq2seqAgent',
        model_file='/private/home/abisee/models/baseline',
        dict_file="/checkpoint/abisee/ahm/dict_twit30k_train_split",
        dict_lower=True,
        batchsize=32,
    )
    opt = parser.parse_args(print_args=False)
    eval_f1(opt, print_parser=parser)
