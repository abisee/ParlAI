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
from parlai.scripts.show_clustergen_model import show_cluster_examples, show_cluster_keywords

import random
import os

def get_cluster_info_fn(opt, num_clusters):
    if "ConvAI2_clusters" in opt['fromfile_datapath']:
        cluster_fname = "/private/home/abisee/ParlAI/data/ConvAI2_clusters/clusters/input_target_clusters_200.txt"
        text2clusterid, clusterid2lst = read_clusterfile(cluster_fname)

        print("getting tfidfs...")
        clusterid2tfidfs = get_tfidfs_wrtclusters(clusterid2lst)
        print("done")

        def get_cluster_info(clusterid):
            return show_cluster_keywords(clusterid2tfidfs, clusterid, num_samples=10)

    elif "ConvAI2_specificity" in opt['fromfile_datapath']:
        def get_cluster_info(clusterid):
            return "specificity of %i" % num_clusters

    else:
        raise Exception()

    return get_cluster_info


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True)
    # Get command line arguments
    parser.add_argument('-ne', '--num-examples', type=int, default=-1)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.add_argument('-of', '--outfile', type=str, default='/tmp/clustercond_alternatives.txt')
    parser.add_argument('--metrics', type=str, default="all",
                        help="list of metrics to show/compute, e.g. ppl,f1,accuracy,hits@1."
                        "If 'all' is specified [default] all are shown.")
    TensorboardLogger.add_cmdline_args(parser)
    parser = interactive.setup_args(parser)
    parser.set_defaults(datatype='valid')
    return parser

def fprint(outfile, txt, end='\n'):
    print(txt, end=end)
    outfile.write(txt + end)

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

    num_clusters = agent.opt['num_clusters']

    # This function displays information about the cluster
    get_cluster_info = get_cluster_info_fn(opt, num_clusters)

    # Show some example dialogs:
    cnt = 0

    f = open(opt['outfile'], "w")

    while not world.epoch_done():
        cnt += opt.get('batchsize', 1)

        act_0 = world.agents[0].act()

        text = act_0['text']
        for line in text.split('\n'):
            if "your persona" in line:
                fprint(f, line)
            else:
                fprint(f, "input: %s" % line)
        fprint(f, "")
        target_clusterid = int(act_0['target_clusterid'])
        use_newline = "ConvAI2_clusters" in opt['fromfile_datapath'] # bool
        fprint(f, "target cluster %i (%s): " % (target_clusterid, get_cluster_info(target_clusterid)), end='\n' if use_newline else '')
        fprint(f, "target: %s" % act_0['eval_labels'][0])
        fprint(f, "")

        world.agents[1].observe(validate(act_0)) # observe just once

        # If we have a classification model, run the generation module forward so we can get cluster_ranking
        if agent.opt.get('classifier_model_file'):
            act_1 = world.agents[1].act() # run forward
            ranking = world.agents[1].cluster_ranking # np array shape (1, k), containing ints
            assert ranking.shape == (1, num_clusters)
            ranking = ranking.squeeze(0).tolist() # len k
            clusters_to_report = ranking[:10]
            fprint(f, "Top clusters predicted by classification model:")
        else:
            clusters_to_report = range(num_clusters)

        # For each of the alternative clusterids, show generated output
        for cluster_id in clusters_to_report:
            world.agents[1].observation['target_clusterid'] = cluster_id # change target clusterid
            world.agents[1].observation['dont_override_target_clusterid'] = True # don't use the clusterid given by the classifier model
            act_1 = world.agents[1].act() # generate

            if use_newline:
                fprint(f, "")
            fprint(f, "target cluster %i (%s): " % (cluster_id, get_cluster_info(cluster_id)), end='\n' if use_newline else '')
            fprint(f, "generated: %s" % act_1['text'])
        fprint(f, "")

        fprint(f, "==============================")

        world.update_counters()

        if act_0.get('episode_done', False):
            fprint(f, "==============================")
            fprint(f, "END OF EPISDOE")
            fprint(f, "==============================")
            fprint(f, "")

        if opt['num_examples'] > 0 and cnt >= opt['num_examples']:
            break

    if world.epoch_done():
        fprint(f, "EPOCH DONE")
    print('finished evaluating task {} using datatype {}'.format(
          opt['task'], opt.get('datatype', 'N/A')))
    report = world.report()
    print(report)
    return report


if __name__ == '__main__':
    parser = setup_args()
    eval_model(parser.parse_args(print_args=False), print_parser=parser)
