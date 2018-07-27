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

from collections import Counter

import random
import math
import os

def tokenize(str):
    return str.split(' ')

def get_idf(word, clusterid2hashmap, num_clusters):
    num_clusters_containing_w = len([1 for hashmap in clusterid2hashmap.values() if word in hashmap])
    return math.log(num_clusters / num_clusters_containing_w)


def get_tfidfs(clusterid2lst):

    num_clusters = len(clusterid2lst)

    clusterid2multiset = {clusterid:sorted([w for sent in lst for w in tokenize(sent)]) for clusterid, lst in clusterid2lst.items()} # int to sorted list of words incl repetition

    clusterid2set = {clusterid:sorted(list(set(lst))) for clusterid, lst in clusterid2multiset.items()} # int to sorted list with no repetition

    all_words = [w for set in clusterid2set.values() for w in set]
    all_words = sorted(list(set(all_words)))

    clusterid2hashmap = {clusterid:{w:True for w in lst} for clusterid,lst in clusterid2set.items()} # int to a dict which maps word to True

    # calculate idf for every word in all_words
    print("calculating idf for all words...")
    word2idf = Counter({word: get_idf(word, clusterid2hashmap, num_clusters) for word in all_words})

    # for each cluster, calculate tfidf for every word in the cluster
    print("calculating tfidfs for each cluster...")
    clusterid2tfidfs = {} # int to Counter mapping word to tfidf
    for clusterid, set_lst in clusterid2set.items():
        # print("clusterid %i of %i" % (clusterid, num_clusters))
        word2tfidf = Counter()
        multiset = clusterid2multiset[clusterid]
        num_words_in_cluster = len(multiset)


        # === SLOW VERSION ===
        # for word in set_lst:
        #     tf = len([1 for w in multiset if w==word]) / num_words_in_cluster
        #     tfidf = tf * word2idf[word]
        #     word2tfidf[word] = tfidf
        # ====================


        # === FAST VERSION ===

        setpointer = 0
        multisetpointer = 0

        while setpointer < len(set_lst):
            word = set_lst[setpointer]
            assert multiset[multisetpointer] == word
            old_multisetpointer = multisetpointer

            while multiset[multisetpointer] == word:
                multisetpointer += 1
                if multisetpointer == len(multiset):
                    assert setpointer == len(set_lst)-1
                    break

            num_occ = multisetpointer - old_multisetpointer

            tf = num_occ / num_words_in_cluster

            tfidf = tf * word2idf[word]
            word2tfidf[word] = tfidf

            setpointer += 1

        # ====================

        clusterid2tfidfs[clusterid] = word2tfidf

    return clusterid2tfidfs


def show_cluster_examples(clusterid2lst, clusterid, num_samples=5):
    examples = clusterid2lst[clusterid]
    num_samples = min(num_samples, len(examples))
    print("CLUSTER %i:" % clusterid)
    for text in random.sample(examples, num_samples):
        print(text)


def show_cluster_keywords(cluster2tfidfs, clusterid, num_samples=10):
    # returns string
    return ", ".join([word for word,_ in cluster2tfidfs[clusterid].most_common(num_samples)])



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
    clusterid2tfidfs = get_tfidfs(clusterid2lst)
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
        print("target cluster %i (%s)" % (target_clusterid, show_cluster_keywords(clusterid2tfidfs, target_clusterid, num_samples=10)))
        print("target:      %s" % world.acts[0]['eval_labels'][0])

        pred_clusterid = world.acts[1]['text'] # text
        assert pred_clusterid[:8]=="cluster "
        pred_clusterid = int(pred_clusterid[8:])
        print("predicted cluster %i (%s)" % (pred_clusterid, show_cluster_keywords(clusterid2tfidfs, pred_clusterid, num_samples=10)))

        top_clusters = [int(i) for i in world.acts[1]['class_ranking'].split(',')] # list of ints
        print("top clusters: ", ", ".join(["%i (%s)" % (i, show_cluster_keywords(clusterid2tfidfs, i, num_samples=3)) for i in top_clusters[:10]]))

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
