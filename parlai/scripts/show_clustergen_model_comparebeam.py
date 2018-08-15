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
from parlai.core.worlds import create_task, validate
from parlai.core.utils import TimeLogger

import parlai.scripts.interactive as interactive
from parlai.scripts.make_cluster_datasets import read_clusterfile
from parlai.scripts.tfidf import get_tfidfs_wrtclusters, get_tfidfs_wrtsents

import random
import os

import nltk
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))


def show_cluster_examples(clusterid2lst, clusterid, num_samples=5):
    """Returns a list of strings"""
    examples = clusterid2lst[clusterid]
    num_samples = min(num_samples, len(examples))
    return random.sample(examples, num_samples)


def show_cluster_keywords(cluster2tfidfs, clusterid, num_samples=10, remove_stopwords=False):
    """Returns string"""
    keywords = []
    for word,_ in cluster2tfidfs[clusterid].most_common():
        if remove_stopwords and word in stopwords:
            continue
        keywords.append(word)
        if len(keywords)==num_samples:
            break
    return ", ".join(keywords)


def fprint(outfile, txt):
    print(txt)
    outfile.write(txt + '\n')


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True)
    # Get command line arguments
    parser.add_argument('-ne', '--num-examples', type=int, default=-1)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.add_argument('-of', '--outfile', type=str, default='/tmp/comparebeam.txt')
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
    beam_sizes = range(1,21)
    cnt = 0
    outfile = opt['outfile']
    # clear outfile
    with open(outfile, "w") as f:
        pass
    while not world.epoch_done():
        f = open(outfile, "a")

        cnt += opt.get('batchsize', 1)
        # world.parley()

        act_0 = world.agents[0].act()

        text = act_0['text']
        for line in text.split('\n'):
            if "your persona" in line:
                fprint(f, line)
            else:
                fprint(f, "input: %s" % line)

        if agent.opt.get('cluster_conditional'):
            fprint(f, "")
            target_clusterid = int(act_0['target_clusterid'])
            fprint(f, "target cluster %02i (%s)" % (target_clusterid, show_cluster_keywords(clusterid2tfidfs, target_clusterid, num_samples=10)))

        fprint(f, "target: %s" % act_0['eval_labels'][0])

        world.agents[1].observe(validate(act_0)) # observe just once
        for beam_size in beam_sizes:
            world.agents[1].beam_size = beam_size
            act_1 = world.agents[1].act() # generate

            if opt.get('classifier_model_file') and beam_size==beam_sizes[0]:
                used_clusterid = act_1['used_clusterid']
                fprint(f, "classifier predicted cluster %i (%s)" % (used_clusterid, show_cluster_keywords(clusterid2tfidfs, used_clusterid, num_samples=10)))

            fprint(f, "generated (beamsize %02i): %s" % (beam_size, act_1['text']))

        world.update_counters()

        fprint(f, "")
        if act_0.get('episode_done', False):
            fprint(f, "==========")
            fprint(f, "")

        f.close() # write to file at end of each example

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
