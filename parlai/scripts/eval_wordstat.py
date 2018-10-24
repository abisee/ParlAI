#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""
This helper script can be used alone with modelfile and task: the output will
contain the word statistics of the model outputs.
One can also use the function defined here in other places in order to get such
statistic for any agent given the agent object (with corr. dict) and a
sequence.


Additionally provides function get_word_stats that can be used in other parts
of runtime code since it depends only on the agent object. For example:

::

  from parlai.scripts.eval_wordstat import get_word_stats
  reqs, cnt = get_word_stats(predictions.tolist(), self.dict)


Examples
--------

.. code-block:: shell

  eval_wordstat.py -mf data/model -t convai2:self --freq-bins 10,100,1000

"""

from parlai.core.params import ParlaiParser
from parlai.core.dict import DictionaryAgent
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.utils import TimeLogger
from parlai.core.metrics import normalize_answer
from parlai.core.logs import TensorboardLogger
from collections import Counter

from parlai_internal.projects.seq2plan2seq.controlled_seq2seq.control_vars import eval_control, CONTROL2CONTINUOUS

import numpy as np

import copy
import numpy
import random
import json


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'compute statistics from model predictions')
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
                        help='Compute %% of unique responses from the model')
    parser.set_defaults(datatype='valid', model='repeat_label')
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

def update_faithfulness_stats(faithfulness_stats, act0, prediction, controls):
    for control in controls.keys():
        # Intended control value
        target_controlval_bucket, target_controlval = eval_control(act0, act0['eval_labels'][0], control)
        assert target_controlval_bucket == int(act0[control])

        # Get control value for prediction
        reply_controlval_bucket, reply_controlval = eval_control(act0, prediction, control)

        # Record in confusion matrix
        faithfulness_stats['confusion_matrices'][control][target_controlval_bucket][reply_controlval_bucket] += 1

        # For continuous control variables, record continuous value
        if CONTROL2CONTINUOUS[control]:
            faithfulness_stats['continuous_values'][control]['target'][target_controlval_bucket].append(target_controlval)
            faithfulness_stats['continuous_values'][control]['model'][target_controlval_bucket].append(reply_controlval)

    return faithfulness_stats



def eval_wordstat(opt, print_parser=None):
    """Evaluates a model.

    :param opt: tells the evaluation function how to run
    :param print_parser: if provided, prints the options that are set within the
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

    data = {} # to write to json
    data['opt'] = opt
    if opt['model']=="repeat_label":
        outfile = "/private/home/abisee/models/goldresponse.wordstats.json"
    else:
        outfile = "%s.%s.%s.%s" % (
            opt.get('model_file'),
            opt.get('datatype'),
            "beam%i" % agent.opt['beam_size'],
            "beamminnbest%i" % agent.opt['beam_min_n_best']
            )
        if len(agent.opt['controls']) > 0:
            outfile += "setcontrols:" + "_".join(["%s%s" % (c, str(d['set_value'])) for c,d in agent.opt['controls'].items()])
        outfile += ".wordstats.json"
    print("writing to outfile: %s" % outfile)

    cnt = 0
    word_statistics = {
        'mean_wlength': [],
        'mean_clength': [],
        'freqs_cnt': Counter(),
        'word_cnt': 0,
        'pred_list': [],
        'pure_pred_list': [],
        'context_list': []
    }
    bins = [int(i) for i in opt['freq_bins'].split(',')]

    # Init this to keep track of faithfulness metrics
    faithfulness_stats = {}

    # Keep a confusion matrix for each control var
    faithfulness_stats['confusion_matrices'] = {
        c: np.zeros((d['num_buckets'], d['num_buckets']))
            for c,d in agent.opt['controls'].items()
        }

    # For the continuous control values, for each target bucket keep a list of all the target and model's control values
    faithfulness_stats['continuous_values'] = {
        c: {
            'target': {b: [] for b in range(d['num_buckets'])},
            'model': {b: [] for b in range(d['num_buckets'])},
            }
        for c,d in agent.opt['controls'].items() if CONTROL2CONTINUOUS[c]
    }


    def process_prediction(prediction, word_statistics):
        word_statistics['pred_list'].append(normalize_answer(prediction))
        freqs, _cnt, wlength, clength = get_word_stats(
            prediction, dictionary, bins=bins
        )
        word_statistics['word_cnt'] += _cnt
        word_statistics['mean_wlength'].append(wlength)
        word_statistics['mean_clength'].append(clength)
        word_statistics['freqs_cnt'] += Counter(freqs)
        return word_statistics

    while not world.epoch_done():
        world.parley()
        if batch_size == 1:
            # raise Exception("some things aren't implemented for batchsize 1")
            cnt += 1
            prediction = world.acts[-1]['text']
            word_statistics['context_list'].append(world.acts[0]['text'])
            word_statistics['pure_pred_list'].append(prediction)
            word_statistics = process_prediction(prediction, word_statistics)
            faithfulness_stats = update_faithfulness_stats(faithfulness_stats, world.acts[0], prediction, agent.opt['controls'])
        else:
            for world_idx,w in enumerate(world.worlds):
                try:
                    try:
                        prediction = w.acts[-1]['text']
                    except KeyError:
                        continue
                    if opt['model']=="repeat_label":
                        prediction = w.acts[0]['eval_labels'][0]
                    word_statistics['context_list'].append(w.acts[0]['text'])
                    word_statistics['pure_pred_list'].append(prediction)
                except IndexError:
                    continue
                cnt += 1
                word_statistics = process_prediction(prediction, word_statistics)

                faithfulness_stats = update_faithfulness_stats(faithfulness_stats, w.acts[0], prediction, agent.opt['controls'])

        if log_time.time() > log_every_n_secs:
            report = world.report()
            text, report = log_time.log(report['exs'], world.num_examples(), report)
            print(text)
            stat_str = 'total_words: {}, '.format(word_statistics['word_cnt'])
            stat_str += ', '.join([
                '<{}:{} ({:.{prec}f}%)'.format(
                    b,
                    word_statistics['freqs_cnt'].get(b, 0),
                    (word_statistics['freqs_cnt'].get(b, 0) /
                        word_statistics['word_cnt']) * 100,
                    prec=2
                )
                for b in bins
            ])
            print(
                "Word statistics: {}, avg_word_length: {:.{prec}f}, "
                "avg_char_length: {:.{prec}f}"
                .format(
                    stat_str,
                    numpy.array(word_statistics['mean_wlength']).mean(),
                    numpy.array(word_statistics['mean_clength']).mean(),
                    prec=2
                )
            )
        if opt['num_examples'] > 0 and cnt >= opt['num_examples']:
            break
    if world.epoch_done():
        print("EPOCH DONE")

    if opt['compute_unique'] is True:
        unique_list = []
        cntr = Counter(word_statistics['pred_list'])
        for k, v in cntr.items():
            if v == 1:
                unique_list.append(k)
        unique_percent = len(unique_list) / len(word_statistics['pred_list']) * 100
        print(
            "Unique responses: {:.{prec}f}%"
            .format(
                unique_percent,
                prec=2
            )
        )

    if opt['dump_predictions_path'] is not None:
        with open(opt['dump_predictions_path'], 'w') as f:
            f.writelines([
                'CONTEXT: {}\nPREDICTION:{}\n\n'.format(c, p)
                for c, p in zip(
                    word_statistics['context_list'],
                    word_statistics['pure_pred_list']
                )
            ])
        if opt['compute_unique'] is True:
            with open(opt['dump_predictions_path'] + '_unique', 'w') as f:
                f.writelines(['{}\n'.format(i) for i in unique_list])

    stat_str = 'total_words: {}, '.format(word_statistics['word_cnt'])
    stat_str += ', '.join([
        '<{}:{} ({:.{prec}f}%)'.format(
            b,
            word_statistics['freqs_cnt'].get(b, 0),
            (word_statistics['freqs_cnt'].get(b, 0) /
                word_statistics['word_cnt']) * 100,
            prec=2
        )
        for b in bins
    ])
    print(
        "Word statistics: {}, avg_word_length: {:.{prec}f}, "
        "avg_char_length: {:.{prec}f}"
        .format(
            stat_str,
            numpy.array(word_statistics['mean_wlength']).mean(),
            numpy.array(word_statistics['mean_clength']).mean(),
            prec=2
        )
    )
    report = world.report()
    if opt['model']=='repeat_label':
        report['ppl'] = 0.0
    print(report)

    data['unique_percent'] = unique_percent
    data['word_statistics'] = word_statistics
    data['predictions'] = word_statistics['pure_pred_list']
    data['report'] = report
    faithfulness_stats['confusion_matrices'] = {c:a.tolist() for c,a in faithfulness_stats['confusion_matrices'].items()} # need to convert to lists because can't write numpy arrays to json
    data['faithfulness_stats'] = faithfulness_stats

    # write results to file
    print("writing to %s" % outfile)
    with open(outfile, 'w') as f:
        json.dump(data, f)

    return report


if __name__ == '__main__':
    parser = setup_args()
    eval_wordstat(parser.parse_args(print_args=False), print_parser=parser)
