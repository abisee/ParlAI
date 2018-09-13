#!/usr/bin/env python
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

Example:
    python eval_wordstat.py -mf data/model -t convai2:self

One can specify bins boundaries with argument -fb | --freq-bins 10,100,1000

Also function get_word_stats can be used in other parts of runtime code since
it depends only on the agent object. To use it - firstly do the import:

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
from parlai.scripts.niwf import load_niwf_buckets, load_niwf_fn, niwf_to_clusterid

from parlai_internal.projects.nlg_plan.cluster_classifier import load_model, embed_sentences, load_centroids_directly, compute_metrics
from parlai_internal.projects.nlg_plan.eval_starspace_classifier import classify

import torch
import copy
import numpy
import random
import json


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
    parser.add_argument('--eval-gold', type=bool, default=False,
                        help='Just evaluate gold responses')
    # parser.set_defaults(datatype='valid', model='repeat_label')
    parser.set_defaults(datatype='valid')
    TensorboardLogger.add_cmdline_args(parser)
    return parser

def fprint(outfile, txt, end='\n'):
    print(txt, end=end)
    # outfile.write(txt + end)

def get_word_stats(text, agent_dict, bins=[0, 100, 1000, 100000]):
    """
    Function which takes text sequence and dict, returns word freq and length statistics
    :param sequence: text sequence
    :param agent_dict: can be external dict or dict from the model
    :param bins: list with range boundaries
    :return: freqs dictionary, num words, avg word length, avg char length
    """
    assert bins == sorted(bins) # bins must be increasing
    pred_list = agent_dict.tokenize(text)
    pred_freq = [agent_dict.freq[word] for word in pred_list]
    freqs = {i: 0 for i in bins}
    for f in pred_freq:
        for b in bins:
            if f <= b:
                freqs[b] += 1
                # break

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

    data = {} # to write to json
    data['opt'] = opt
    if opt['eval_gold']:
        outfile = "/private/home/abisee/models/goldresponse.wordstats.json"
    else:
        outfile = "%s.%s.%s" % (opt.get('model_file'), opt.get('datatype'), "beam%i" % opt['beam_size'])
        if opt['fixed_clusterid'] != -1:
            outfile += ".fixed_clusterid%i" % (opt['fixed_clusterid'])
        outfile += ".wordstats.json"
    print("writing to outfile: %s" % outfile)
    # f = open(outfile, "w")
    f = None

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

    generated = [] # list of strings
    used_clusterids = [] # list of ints

    starspace_model = load_model()

    # init closeness metrics
    closeness_metrics_eucl = init_closeness_metrics()
    closeness_metrics_cos = init_closeness_metrics()

    # init distinctness metrics
    distinct_n = [1, 2, 3, 4]
    ngram_counters = {n:Counter() for n in distinct_n}

    # init niwf metrics
    niwfs = [] # list of floats
    sent2niwf_fn = load_niwf_fn()
    niwf_oov_cnt = 0

    if "specificity" in opt["fromfile_datapath"]:
        niwf_bucket_lbs = load_niwf_buckets(agent.opt['num_clusters'])
    else:
        niwf_bucket_lbs = load_niwf_buckets(10)

    while not world.epoch_done():
        print("running world.parley...")
        world.parley()
        print("finished parley. updating counts...")

        if batch_size == 1:
            raise Exception("some things aren't implemented for batchsize 1")
            cnt += 1
            prediction = world.acts[-1]['text']
            word_statistics['context_list'].append(world.acts[0]['text'])
            word_statistics['pure_pred_list'].append(prediction)
            word_statistics = process_prediction(prediction, word_statistics)
        else:
            for w in world.worlds:
                try:
                    if opt['eval_gold']:
                        # ===== to measure wordstats on gold responses: ======
                        prediction = w.acts[0]['eval_labels']
                        assert len(prediction)==1
                        prediction = prediction[0]
                        # =======================================
                    else:
                        prediction = w.acts[-1]['text']
                except KeyError:
                    continue

                # Update closeness metrics
                add_to_dialoghist(w.agents[1], w.acts[0])
                closeness_metrics_eucl, choice_eucl = update_closeness_metrics(prediction, w.agents[1], closeness_metrics_eucl, starspace_model, dist_metric="eucl")
                closeness_metrics_cos, choice_cos = update_closeness_metrics(prediction, w.agents[1], closeness_metrics_cos, starspace_model, dist_metric="cosine")
                label = w.acts[0]['eval_labels'][0]
                w.agents[1].dialoghist_convo.append(label)

                # Update distinct-n metrics
                pred_words = prediction.split()
                for n in distinct_n:
                    pred_ngrams = [" ".join(pred_words[i:i+n]) for i in range(len(pred_words)-n+1)] # n-grams in prediction
                    ngram_counters[n].update(pred_ngrams)

                # Update wordstats
                word_statistics['context_list'].append(w.acts[0]['text'])
                word_statistics['pure_pred_list'].append(prediction)
                cnt += 1
                word_statistics = process_prediction(prediction, word_statistics)

                # Record NIWF
                pred_niwf, problem_words = sent2niwf_fn(prediction)
                if len(problem_words) > 0:
                    niwf_oov_cnt += 1
                niwfs.append(pred_niwf)
                if 'target_niwf' in w.acts[0]:
                    label_niwf, label_pw = sent2niwf_fn(label)
                    assert len(label_pw)==0
                    assert label_niwf == float(w.acts[0]['target_niwf']) # check sent2niwf_fn matches the labels in the datafile
                    assert niwf_to_clusterid(label_niwf, niwf_bucket_lbs) == int(w.acts[0]['target_clusterid']) # check bucket matches label in datafile

                # For clustercond models, record generated text and the clusterid used
                if 'used_clusterid' in w.acts[-1]:
                    generated.append(prediction)
                    used_clusterids.append(w.acts[-1]['used_clusterid'])


        if log_time.time() > log_every_n_secs:
            print('reporting...')
            report = world.report()
            text, report = log_time.log(report['exs'], world.num_examples(), report)
            print(text)
            stat_str = 'total_words: {}, '.format(word_statistics['word_cnt']) + ', '.join(
                ['<{}:{} ({:.{prec}f}%)'.format(b, word_statistics['freqs_cnt'].get(b, 0), (word_statistics['freqs_cnt'].get(b, 0) / word_statistics['word_cnt']) * 100, prec=2)
                 for b in bins])
            stat_str = "Word statistics (cumulative rare word counts): {}, avg_word_length: {:.{prec}f}, avg_char_length: {:.{prec}f}".format(
                stat_str, numpy.array(word_statistics['mean_wlength']).mean(), numpy.array(word_statistics['mean_clength']).mean(), prec=2)
            print(stat_str)
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
        unique_stat_str = "Unique responses: {:.{prec}f}%".format(unique_percent, prec=2)
        fprint(f, unique_stat_str)

    data['unique_percent'] = unique_percent

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

    stat_str = 'total_words: {}, '.format(word_statistics['word_cnt']) + ', '.join(
        ['<{}:{} ({:.{prec}f}%)'.format(b, word_statistics['freqs_cnt'].get(b, 0), (word_statistics['freqs_cnt'].get(b, 0) / word_statistics['word_cnt']) * 100, prec=2)
         for b in bins])
    stat_str = "Word statistics (cumulative rare word counts): {}, avg_word_length: {:.{prec}f}, avg_char_length: {:.{prec}f}".format(
        stat_str, numpy.array(word_statistics['mean_wlength']).mean(), numpy.array(word_statistics['mean_clength']).mean(), prec=2)
    fprint(f, stat_str)

    data['word_statistics'] = {
        'mean_wlength': numpy.array(word_statistics['mean_wlength']).mean(),
        'mean_clength': numpy.array(word_statistics['mean_clength']).mean(),
        'freqs_perc': {b : (word_statistics['freqs_cnt'].get(b,0) * 100 / word_statistics['word_cnt']) for b in bins},
    }

    data['predictions'] = word_statistics['pure_pred_list']

    report = world.report()
    fprint(f, str(report))

    data['report'] = report

    # Show distinct-n metrics
    num_unigrams = sum(ngram_counters[1].values())
    fprint(f, "\nDiversity metrics: ", end='')
    fprint(f, ", ".join([
        "distinct-%i: %.4f (%i/%i)" % (n, len(ngram_counter)/num_unigrams, len(ngram_counter), num_unigrams)
        for n, ngram_counter in ngram_counters.items()
        ]))
    fprint(f, "")

    data['distinct-n'] = {n: len(ngram_counter)/num_unigrams for n, ngram_counter in ngram_counters.items()}

    # Print niwf metrics
    avg_niwf = sum(niwfs)/len(niwfs)
    fprint(f, "Average NIWF over %i examples: %.6f" % (len(niwfs), avg_niwf))
    fprint(f, "NIWF bucket lowerbounds: %s" % (", ".join(["%.4f" % lb for lb in niwf_bucket_lbs])))
    pred_niwf_bucketids = [niwf_to_clusterid(niwf, niwf_bucket_lbs) for niwf in niwfs]
    bucket2count = Counter()
    bucket2count.update(pred_niwf_bucketids)
    assert sum(bucket2count.values())==len(niwfs)
    fprint(f, "Percent generated in each bucket: %s" % (", ".join(["%.2f%% (%i/%i)" % (bucket2count[bucketid]*100/len(niwfs), bucket2count[bucketid], len(niwfs)) for bucketid in sorted(bucket2count.keys())])))
    fprint(f, "Number of responses containing words that are OOV for PersonaChat (affecting NIWF measure): %i/%i (%.2f%%)\n" % (niwf_oov_cnt, len(niwfs), niwf_oov_cnt*100/len(niwfs)))

    data['niwf'] = {
        'avg_niwf': avg_niwf,
        'niwf_bucket_lbs': niwf_bucket_lbs,
        'niwf_bucket_dist': [bucket2count[bucketid]*100/len(niwfs) for bucketid in sorted(bucket2count.keys())], # list of percentages
        'niwf_oov_cnt': niwf_oov_cnt,
    }

    # Compute faithfulness
    if len(used_clusterids) > 0:
        if "specificity" in opt["fromfile_datapath"]:
            acc = len([1 for (pred, gold) in zip(pred_niwf_bucketids, used_clusterids) if pred==gold])
            acc /= len(pred_niwf_bucketids)
            fprint(f, "Faithfulness stats from %i examples: acc=%.2f%%" % (len(generated), acc*100))
            data['faithfulness'] = acc*100
        else:
            starspace_model = load_model()
            cluster_centers = torch.Tensor(load_centroids_directly())
            scores, ranking = classify(generated, starspace_model, "output", cluster_centers, "euclidean") # scores and ranking both shape (num_exs, num_classes)
            hits_at_n, mrr = compute_metrics(scores, used_clusterids)
            fprint(f, "Faithfulness stats from %i examples:" % len(generated))
            fprint(f, ", ".join(["hits_at/%i: %.2f%%" % (n, perc) for n, perc in hits_at_n.items()]) + ", " + "mrr: %4f"%mrr)
        fprint(f, "")

    # Show closeness metrics
    fprint(f, show_closeness_metrics(closeness_metrics_eucl))
    # show_closeness_metrics(closeness_metrics_cos)

    # write results to file
    print("writing to %s" % outfile)
    with open(outfile, 'w') as f:
        json.dump(data, f)

    return report


if __name__ == '__main__':
    parser = setup_args()
    eval_wordstat(parser.parse_args(print_args=False), print_parser=parser)
