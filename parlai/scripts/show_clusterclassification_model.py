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

import parlai.scripts.interactive as interactive
from parlai.scripts.make_cluster_datasets import read_clusterfile
from parlai.scripts.tfidf import get_tfidfs_wrtclusters, get_tfidfs_wrtsents
from parlai.scripts.show_clustergen_model import show_cluster_examples, show_cluster_keywords

import random
import os


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True)
    # Get command line arguments
    parser.add_argument('-ne', '--num-examples', type=int, default=-1)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.add_argument('--metrics', type=str, default="all",
                        help="list of metrics to show/compute, e.g. ppl,f1,accuracy,hits@1."
                        "If 'all' is specified [default] all are shown.")
    TensorboardLogger.add_cmdline_args(parser)
    parser = interactive.setup_args(parser)
    parser.set_defaults(datatype='valid')
    return parser


def eval_model(opt, printargs=None, print_parser=None):
    """Evaluates a model.

    Arguments:
    opt -- tells the evaluation function how to run
    print_parser -- if provided, prints the options that are set within the
        model after loading the model
    """
    assert opt.get('batchsize')==1
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

    cluster_fname = "/private/home/abisee/ParlAI/data/ConvAI2_clusters/clusters/input_target_clusters_200.txt"
    text2clusterid, clusterid2lst = read_clusterfile(cluster_fname)

    print("getting tfidfs...")
    clusterid2tfidfs = get_tfidfs_wrtclusters(clusterid2lst)
    print("done")

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    world = create_task(opt, agent)

    if print_parser:
        # Show arguments after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()
    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()

    # Show some example dialogs:
    cnt = 0
    while not world.epoch_done():
        cnt += opt.get('batchsize', 1)
        world.parley()

        text = world.acts[0]['text']
        for line in text.split('\n'):
            if "your persona" in line:
                print(line)
            else:
                print("input: %s" % line)
        print("")
        target_clusterid = int(world.acts[0]['target_clusterid'])
        target_str = "target cluster %i (%s)" % (target_clusterid, show_cluster_keywords(clusterid2tfidfs, target_clusterid, num_samples=10))
        print(target_str)
        print("target:      %s" % world.acts[0]['eval_labels'][0])

        pred_clusterid = world.acts[1]['pred_clusterid'] # int
        pred_str = "predicted cluster %i (%s)" % (pred_clusterid, show_cluster_keywords(clusterid2tfidfs, pred_clusterid, num_samples=10))
        print(pred_str)

        top_clusters = world.acts[1]['class_ranking'] # list of ints
        num_clusters = len(top_clusters)
        print("top clusters: ", ", ".join(["%i (%s)" % (i, show_cluster_keywords(clusterid2tfidfs, i, num_samples=3)) for i in top_clusters[:10]]))

        gold_cluster_rank = top_clusters.index(target_clusterid)+1 # int in range [1,num_clusters]
        print("target cluster rank: %i/%i" % (gold_cluster_rank, num_clusters))

        # write attnvis to file
        if opt.get('attn_vis'):
            world.agents[1].model.attn_vis.add_target([target_str])
            world.agents[1].model.attn_vis.output_tokens = [pred_str]
            world.agents[1].model.attn_vis.write()

        print("")
        if world.acts[0].get('episode_done', False):
            print("==========")
            print("")

        if opt['num_examples'] > 0 and cnt >= opt['num_examples']:
            break

    if world.epoch_done():
        print("EPOCH DONE")
    print('finished evaluating task {} using datatype {}'.format(
          opt['task'], opt.get('datatype', 'N/A')))
    report = world.report()
    print(report)
    return report


if __name__ == '__main__':
    parser = setup_args()
    eval_model(parser.parse_args(print_args=False), print_parser=parser)