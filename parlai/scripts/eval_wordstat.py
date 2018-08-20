#!/usr/bin/env python
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""
This helper script can be used alone with modelfile and task: the output will contain the
word statistics of the model outputs.
One can also use the function defined here in other places in order to get such statistic
for any agent given the agent object (with corr. dict) and a sequence.

Example:
    python eval_wordstat.py -mf data/model -t convai2:self

One can specify bins boundaries with argument -fb | --freq-bins 10,100,1000 or so

Also function get_word_stats can be used in other parts of runtime code since it depends only on
the agent object. To use it - firstly do the import:

    from parlai.scripts.eval_wordstat import get_word_stats

then you can call this function like this:

    reqs, cnt = get_word_stats(predictions.tolist(), self.dict)
"""

from parlai.core.params import ParlaiParser
from parlai.core.dict import DictionaryAgent
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.utils import TimeLogger
from parlai.core.metrics import normalize_answer
from parlai.core.logs import TensorboardLogger
from collections import Counter
from parlai.scripts.closeness_metrics import init_closeness_metrics, update_closeness_metrics, show_closeness_metrics, add_to_dialoghist

from parlai_internal.projects.nlg_plan.cluster_classifier import load_model, embed_sentences, load_centroids_directly, compute_metrics
from parlai_internal.projects.nlg_plan.eval_starspace_classifier import classify

import torch
import copy
import numpy
import random


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True)
    DictionaryAgent.add_cmdline_args(parser)
    # Get command line arguments
    parser.add_argument('-ne', '--num-examples', type=int, default=-1)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.add_argument('-ed', '--external-dict', type=str, default=None,
                        help='External dictionary for stat computation')
    parser.add_argument('-fb', '--freq-bins', type=str, default='0,100,1000,10000',
                        help='Bins boundaries for rare words stat')
    parser.add_argument('-dup', '--dump-predictions-path', type=str, default=None,
                        help='Dump predictions into file')
    parser.add_argument('-cun', '--compute-unique', type=bool, default=True,
                        help='Compute % of unique responses from the model')
    # parser.set_defaults(datatype='valid', model='repeat_label')
    parser.set_defaults(datatype='valid')
    TensorboardLogger.add_cmdline_args(parser)
    return parser


def get_word_stats(text, agent_dict, bins=[0, 100, 1000, 100000]):
    """
    Function which takes text sequence and dict, returns word freq and length statistics
    :param sequence: text sequence
    :param agent_dict: can be external dict or dict from the model
    :param bins: list with range boundaries
    :return: freqs dictionary, num words, avg word length, avg char length
    """
    pred_list = agent_dict.tokenize(text)
    pred_freq = [agent_dict.freq[word] for word in pred_list]
    freqs = {i: 0 for i in bins}
    for f in pred_freq:
        for b in bins:
            if f <= b:
                freqs[b] += 1
                break

    wlength = len(pred_list)
    clength = len(text)  # including spaces
    return freqs, len(pred_freq), wlength, clength


def eval_wordstat(opt, print_parser=None):
    """Evaluates a model.

    Arguments:
    opt -- tells the evaluation function how to run
    print_parser -- if provided, prints the options that are set within the
        model after loading the model
    """
    random.seed(42)

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    world = create_task(opt, agent)

    if opt.get('external_dict'):
        print('[ Using external dictionary from: {} ]'.format(
            opt['external_dict']))
        dict_opt = copy.deepcopy(opt)
        dict_opt['dict_file'] = opt['external_dict']
        dictionary = DictionaryAgent(dict_opt)
    else:
        print('[ Using model bundled dictionary ]')
        dictionary = agent.dict

    batch_size = opt['batchsize']

    if print_parser:
        # Show arguments after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()
    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()

    cnt = 0
    word_statistics = {'mean_wlength': [], 'mean_clength': [], 'freqs_cnt': Counter(), 'word_cnt': 0, 'pred_list': [], 'pure_pred_list': [], 'context_list': []}
    bins = [int(i) for i in opt['freq_bins'].split(',')]

    def process_prediction(prediction, word_statistics):
        word_statistics['pred_list'].append(normalize_answer(prediction))
        freqs, _cnt, wlength, clength = get_word_stats(prediction, dictionary, bins=bins)
        word_statistics['word_cnt'] += _cnt
        word_statistics['mean_wlength'].append(wlength)
        word_statistics['mean_clength'].append(clength)
        word_statistics['freqs_cnt'] += Counter(freqs)
        return word_statistics

    generated = [] # list of strings
    used_clusterids = [] # list of ints

    starspace_model = load_model()

    # init closeness metrics
    closeness_metrics_eucl = init_closeness_metrics()
    closeness_metrics_cos = init_closeness_metrics()

    while not world.epoch_done():
        world.parley()
        if batch_size == 1:
            cnt += 1
            prediction = world.acts[-1]['text']
            word_statistics['context_list'].append(world.acts[0]['text'])
            word_statistics['pure_pred_list'].append(prediction)
            word_statistics = process_prediction(prediction, word_statistics)
        else:
            for w in world.worlds:
                try:
                    prediction = w.acts[-1]['text']

                    # ===== to measure wordstats on gold responses: ======
                    # prediction = w.acts[0]['eval_labels']
                    # assert len(prediction)==1
                    # prediction = prediction[0]
                    # =======================================
                except KeyError:
                    continue

                # Update closeness metrics
                add_to_dialoghist(w.agents[1], w.acts[0])
                closeness_metrics_eucl, choice_eucl = update_closeness_metrics(prediction, w.agents[1], closeness_metrics_eucl, starspace_model, dist_metric="eucl")
                closeness_metrics_cos, choice_cos = update_closeness_metrics(prediction, w.agents[1], closeness_metrics_cos, starspace_model, dist_metric="cosine")
                label = w.acts[0]['eval_labels'][0]
                w.agents[1].dialoghist_convo.append(label)

                word_statistics['context_list'].append(w.acts[0]['text'])
                word_statistics['pure_pred_list'].append(prediction)
                cnt += 1
                word_statistics = process_prediction(prediction, word_statistics)

                if 'used_clusterid' in w.acts[-1]:
                    generated.append(prediction)
                    used_clusterids.append(w.acts[-1]['used_clusterid'])

        if log_time.time() > log_every_n_secs:
            report = world.report()
            text, report = log_time.log(report['exs'], world.num_examples(), report)
            print(text)
            stat_str = 'total_words: {}, '.format(word_statistics['word_cnt']) + ', '.join(
                ['<{}:{} ({:.{prec}f}%)'.format(b, word_statistics['freqs_cnt'].get(b, 0), (word_statistics['freqs_cnt'].get(b, 0) / word_statistics['word_cnt']) * 100, prec=2)
                 for b in bins])
            stat_str = "Word statistics: {}, avg_word_length: {:.{prec}f}, avg_char_length: {:.{prec}f}".format(
                stat_str, numpy.array(word_statistics['mean_wlength']).mean(), numpy.array(word_statistics['mean_clength']).mean(), prec=2)
            print(stat_str)
        if opt['num_examples'] > 0 and cnt >= opt['num_examples']:
            break
    if world.epoch_done():
        print("EPOCH DONE")

    if opt['compute_unique'] is True:
        unique_list = []
        cntr = Counter(word_statistics['pred_list'])
        for k,v in cntr.items():
            if v == 1:
                unique_list.append(k)
        unique_stat_str = "Unique responses: {:.{prec}f}%".format(len(unique_list) / len(word_statistics['pred_list']) * 100, prec=2)
        print(unique_stat_str)

    if opt['dump_predictions_path'] is not None:
        with open(opt['dump_predictions_path'], 'w') as f:
            f.writelines(['CONTEXT: {}\nPREDICTION:{}\n\n'.format(c,p) for c,p in zip(word_statistics['context_list'],word_statistics['pure_pred_list'])])
        if opt['compute_unique'] is True:
            with open(opt['dump_predictions_path']+'_unique', 'w') as f:
                f.writelines(['{}\n'.format(i) for i in unique_list])

    stat_str = 'total_words: {}, '.format(word_statistics['word_cnt']) + ', '.join(
        ['<{}:{} ({:.{prec}f}%)'.format(b, word_statistics['freqs_cnt'].get(b, 0), (word_statistics['freqs_cnt'].get(b, 0) / word_statistics['word_cnt']) * 100, prec=2)
         for b in bins])
    stat_str = "Word statistics: {}, avg_word_length: {:.{prec}f}, avg_char_length: {:.{prec}f}".format(
        stat_str, numpy.array(word_statistics['mean_wlength']).mean(), numpy.array(word_statistics['mean_clength']).mean(), prec=2)
    print(stat_str)

    report = world.report()
    print(report)

    # Show closeness metrics
    show_closeness_metrics(closeness_metrics_eucl)
    show_closeness_metrics(closeness_metrics_cos)

    # Compute faithfulness
    if len(used_clusterids) > 0:
        starspace_model = load_model()
        cluster_centers = torch.Tensor(load_centroids_directly())
        scores, ranking = classify(generated, starspace_model, "output", cluster_centers, "euclidean") # scores and ranking both shape (num_exs, num_classes)
        hits_at_n, mrr = compute_metrics(scores, used_clusterids)
        print("Faithfulness stats from %i examples:" % len(generated))
        print(", ".join(["hits_at/%i: %.2f%%" % (n, perc) for n, perc in hits_at_n.items()]) + ", " + "mrr: %4f"%mrr)

    # write results to file
    outfile = "%s.%s.%s" % (opt.get('model_file'), opt.get('datatype'), "wordstats")
    print("writing to %s" % outfile)
    with open(outfile, 'w') as f:
        f.write("%s: %s\n" % (opt.get('datatype'), str(report)))
        f.write(stat_str + "\n")
        f.write(unique_stat_str + "\n")

    return report


if __name__ == '__main__':
    parser = setup_args()
    eval_wordstat(parser.parse_args(print_args=False), print_parser=parser)
