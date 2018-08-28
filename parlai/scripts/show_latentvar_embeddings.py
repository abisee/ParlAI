# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Basic example which iterates through the tasks specified and
evaluates the given model on them.

For example:
`python eval_model.py -t "babi:Task1k:2" -m "repeat_label"`
or
`python eval_model.py -t "#CornellMovie" -m "ir_baseline" -mp "-lp 0.5"`
"""
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.logs import TensorboardLogger
from parlai.core.worlds import create_task
from parlai.core.utils import TimeLogger

import random
import os
import torch

def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True)
    TensorboardLogger.add_cmdline_args(parser)
    parser.set_defaults(datatype='valid')
    parser.set_defaults(task='fromfile:parlaiformat')
    parser.set_defaults(fromfile_datapath='/private/home/abisee/ParlAI/data/ConvAI2_abi/valid_self_original_abi.txt')
    return parser

def proj_file_remove_dups(writer):
    """look at projector config file and delete duplicate listings"""
    proj_config_fname = os.path.join(writer.log_dir, 'projector_config.pbtxt')
    lines = []
    with open(proj_config_fname, "r") as f:
        for line in f:
            lines.append(line.strip())
    assert len(lines) % 5 == 0
    num_records = int(len(lines)/5)
    records = [lines[5*i:5*(i+1)] for i in range(num_records)]
    new_records = []
    for idx, r in enumerate(records):
        if r[1] not in [rec[1] for rec in new_records]:
            new_records.append(r)
    with open(proj_config_fname, 'w') as f:
        for record in new_records:
            for line in record:
                f.write(line + "\n")


def eval_model(opt, printargs=None, print_parser=None):
    """Evaluates a model.

    Arguments:
    opt -- tells the evaluation function how to run
    print_parser -- if provided, prints the options that are set within the
        model after loading the model
    """
    if printargs is not None:
        print('[ Deprecated Warning: eval_model no longer uses `printargs` ]')
        print_parser = printargs
    if print_parser is not None:
        if print_parser is True and isinstance(opt, ParlaiParser):
            print_parser = opt
        elif print_parser is False:
            print_parser = None
    if isinstance(opt, ParlaiParser):
        print('[ Deprecated Warning: eval_model should be passed opt not Parser ]')
        opt = opt.parse_args()

    random.seed(42)

    # Load model from file
    agent = create_agent(opt, requireModelExists=True)

    if print_parser:
        # Show arguments after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()


    writer = TensorboardLogger(agent.opt)
    num_clusters = agent.opt['num_clusters']
    cluster_embeddings = agent.model.cluster_id_embeddings.weight # tensor shape (num_clusters, emb_size)
    assert cluster_embeddings.size(0) == num_clusters
    cluster_names = [str(i) for i in range(num_clusters)]
    writer.add_embedding(cluster_embeddings, cluster_names, tag="cluster_embeddings")
    proj_file_remove_dups(writer) # remove any duplicates from projector config file



if __name__ == '__main__':
    parser = setup_args()
    eval_model(parser.parse_args(print_args=False), print_parser=parser)
