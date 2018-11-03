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
from parlai_internal.projects.seq2plan2seq.controlled_seq2seq.beamsearch_features import eval_attr, ATTR2SENTSCOREFN
from parlai_internal.projects.seq2plan2seq.controlled_seq2seq.util import get_history

import numpy as np

import copy
import numpy
import random
import json
import time
import hashlib
import os


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
    parser.add_argument('-gr', '--gold-response', type=bool, default=False,
                        help='Compute stats for gold response')
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


def update_output_attr_stats(output_attribute_stats, prediction, history):
    for attr, l in output_attribute_stats.items():
        attr_score = eval_attr(prediction, history, attr)
        l.append(attr_score)
    return output_attribute_stats


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
    if opt['gold_response']:
        outfile = "/private/home/abisee/models/goldresponse"
    else:
        outfile = "%s.%s.%s.%s" % (
            opt.get('model_file'),
            opt.get('datatype'),
            "use%sreply" % agent.opt['use_reply'],
            "beam%i" % agent.opt['beam_size'],
            )
        if agent.opt['beam_size'] > 1:
            outfile += ".beamminnbest%i" % agent.opt['beam_min_n_best']
        if len(agent.opt['controls']) > 0:
            outfile += ".setcontrols:" + "_".join(["%s%s" % (c, str(d['set_value'])) for c,d in agent.opt['controls'].items()])
        if len(agent.beam_features) > 0:
            outfile += ".beamfeatures:" + "_".join(["%s%s" % (f, str(w)) for f,w in zip(agent.beam_features, agent.beam_feature_wts)])
    if opt['num_examples'] != -1:
        outfile += ".numex%i" % opt['num_examples']
    outfile += ".wordstats.json"
    print("\nWriting to outfile: %s\n" % outfile)

    data['outfile'] = outfile

    # check if outfile is too long. if so, replace with hash
    if len(os.path.basename(outfile))>255:
        print("Outfile name is too long. hashing instead.")
        hash_object = hashlib.md5(outfile.encode())
        outfile = "%s.%s.wordstats.json" % (opt.get('model_file'), hash_object.hexdigest())
        print("\nNew outfile: %s\n" % outfile)

    # check you can write to it
    with open(outfile, 'w') as f:
        json.dump({}, f)
    os.remove(outfile)

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

    # Init this to keep track of various output attributes
    # For each attribute, we have a list of all the values
    output_attribute_stats = {attr: [] for attr in ATTR2SENTSCOREFN.keys()} # string to list of floats

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

    t0 = time.time()
    while not world.epoch_done():
        # print("parlaying...")
        world.parley()
        if batch_size == 1:
            # raise Exception("some things aren't implemented for batchsize 1")
            cnt += 1
            # print('updating stats...')
            prediction = world.acts[-1]['text']
            if opt['gold_response']:
                prediction = w.acts[0]['eval_labels'][0]
            word_statistics['context_list'].append(world.acts[0]['text'])
            word_statistics['pure_pred_list'].append(prediction)
            word_statistics = process_prediction(prediction, word_statistics)
            faithfulness_stats = update_faithfulness_stats(faithfulness_stats, world.acts[0], prediction, agent.opt['controls'])
            history = get_history([world.acts[0]])[0] # triple
            output_attribute_stats = update_output_attr_stats(output_attribute_stats, prediction, history)
        else:
            for world_idx,w in enumerate(world.worlds):
                try:
                    try:
                        prediction = w.acts[-1]['text']
                    except KeyError:
                        continue
                    if opt['gold_response']:
                        prediction = w.acts[0]['eval_labels'][0]
                    word_statistics['context_list'].append(w.acts[0]['text'])
                    word_statistics['pure_pred_list'].append(prediction)
                except IndexError:
                    continue
                cnt += 1
                # print('updating stats...')
                word_statistics = process_prediction(prediction, word_statistics)
                faithfulness_stats = update_faithfulness_stats(faithfulness_stats, w.acts[0], prediction, agent.opt['controls'])
                history = get_history([w.acts[0]])[0] # triple
                output_attribute_stats = update_output_attr_stats(output_attribute_stats, prediction, history)

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
            # print(
            #     "Word statistics: {}, avg_word_length: {:.{prec}f}, "
            #     "avg_char_length: {:.{prec}f}"
            #     .format(
            #         stat_str,
            #         numpy.array(word_statistics['mean_wlength']).mean(),
            #         numpy.array(word_statistics['mean_clength']).mean(),
            #         prec=2
            #     )
            # )
        if opt['num_examples'] > 0 and cnt >= opt['num_examples']:
            break
    if world.epoch_done():
        print("EPOCH DONE")
    print("time to process %i examples: %f" % (cnt, time.time()-t0))

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
    if opt['gold_response']:
        report['ppl'] = 0.0
    print(report)

    data['unique_percent'] = unique_percent
    data['word_statistics'] = word_statistics
    data['predictions'] = word_statistics['pure_pred_list']
    data['report'] = report
    faithfulness_stats['confusion_matrices'] = {c:a.tolist() for c,a in faithfulness_stats['confusion_matrices'].items()} # need to convert to lists because can't write numpy arrays to json
    data['faithfulness_stats'] = faithfulness_stats
    data['output_attribute_stats'] = output_attribute_stats

    # write results to file
    print("writing to %s" % outfile)
    with open(outfile, 'w') as f:
        json.dump(data, f)

    return report


if __name__ == '__main__':
    parser = setup_args()
    eval_wordstat(parser.parse_args(print_args=False), print_parser=parser)
