from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from collections import Counter
import math
import pickle
import nltk
from rouge import Rouge


def get_rouge(sent1, sent2):
    # print("")
    # print("comparing '%s' and '%s'" % (sent1, sent2))
    rouge = Rouge()
    scores = rouge.get_scores(sent1, sent2)
    return scores[0]


def get_bleu(sent1, sent2):
    """Return similarity for two strings"""
    print("comparing '%s' and '%s'" % (sent1, sent2))
    tokens1 = sent1.split()
    tokens2 = sent2.split()
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([tokens1], tokens2, weights = [1])
    # print(BLEUscore)
    # BLEUscore = nltk.translate.bleu_score.sentence_bleu([tokens2], tokens1, weights = [1])
    # print(BLEUscore)

    return BLEUscore


def get_extrep(reply, prev_utts):
    scores = [get_rouge(reply, utt) for utt in prev_utts]

    sims_r1 = [score['rouge-1']['p'] for score in scores]
    sims_r2 = [score['rouge-2']['p'] for score in scores]
    sims_rl = [score['rouge-l']['p'] for score in scores]

    # utts_sims = [(utt, sim) for utt, sim in zip(prev_utts, sims)]
    # utts_sims = sorted(utts_sims, key = lambda x: x[1], reverse=True)

    # print("")
    # print("REPLY: %s" %  reply)
    # for (utt, sim) in utts_sims:
    #     print("sim %.4f  PREVUTT: %s" % (sim, utt))

    max_r1 = max(sims_r1)
    max_r2 = max(sims_r2)
    max_rl = max(sims_rl)

    return max_r1, max_r2, max_rl
