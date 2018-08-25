from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.core.utils import msg_to_str
from collections import Counter
import math

def train_opts(opt, num_buckets=None):
    opt['datatype'] = 'train:ordered'
    opt['fromfile_datapath'] = "/private/home/abisee/ParlAI/data/ConvAI2_parlaiformat/train_self_original.txt"
    opt['outfile'] = '/tmp/dump_train'
    if num_buckets:
        opt['outfile'] = '/private/home/abisee/ParlAI/data/ConvAI2_specificity/train_self_original_classification_%ibuckets.txt' % (num_buckets)
    return opt

def valid_opts(opt, num_buckets=None):
    opt['datatype'] = 'valid'
    opt['fromfile_datapath'] = "/private/home/abisee/ParlAI/data/ConvAI2_parlaiformat/valid_self_original.txt"
    opt['outfile'] = '/tmp/dump_valid'
    if num_buckets:
        opt['outfile'] = '/private/home/abisee/ParlAI/data/ConvAI2_specificity/valid_self_original_classification_%ibuckets.txt' % (num_buckets)
    return opt

def get_word_counts(opt):
    # Create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    ignorefields = opt.get('ignore_fields', '')

    # Count word frequency for all words in training and validation set
    word_counter = Counter()
    num_sents = 0
    print("Getting NIWF labels from %s..." % opt['datatype'])
    while True:
        world.parley()
        text = world.acts[0]['text']
        text = text.split('\n')[-1] # string
        reply = world.acts[0].get('labels', world.acts[0].get('eval_labels'))[0] # string

        words = reply.split()
        words = list(set(words))
        word_counter.update(words)
        num_sents += 1

        if world.epoch_done():
        # if num_sents > 100:
            print('EPOCH DONE')
            break

    print("num_sents: ", num_sents)
    print("most common: ", word_counter.most_common(10))

    return num_sents, word_counter


def get_iwf_for_sents(opt, word2iwf):
    # Create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    ignorefields = opt.get('ignore_fields', '')

    # Count word frequency for all words in training and validation set
    sent2iwf = {}
    print("Calculating IWF for sentences in %s..." % opt['datatype'])
    num_sents = 0
    while True:
        world.parley()
        text = world.acts[0]['text']
        text = text.split('\n')[-1] # string
        reply = world.acts[0].get('labels', world.acts[0].get('eval_labels'))[0] # string

        words = reply.split()
        words = list(set(words))
        iwf = max([word2iwf[w] for w in words]) # iwf for the sent
        sent2iwf[reply] = iwf
        num_sents += 1

        if world.epoch_done():
        # if num_sents > 100:
            print('EPOCH DONE')
            break

    return sent2iwf


def learn_niwf(opt):
    # get word counts from train and val sets
    opt = train_opts(opt)
    num_sents_train, word_counter_train = get_word_counts(opt)
    # num_sents_train, word_counter_train = 0, Counter()
    opt = valid_opts(opt)
    num_sents_valid, word_counter_valid = get_word_counts(opt)

    # merge word counts
    num_sents = num_sents_train + num_sents_valid
    word_counter = word_counter_train
    for word,count in word_counter_valid.items():
        word_counter[word] += count

    # Compute IWF for every word
    print("computing IWF for all words...")
    word2iwf = {}
    nom = math.log(1 + num_sents)
    for word, count in word_counter.items():
        word2iwf[word] = nom/count

    # Compute IWF for every sentence in train + val
    opt = train_opts(opt)
    sent2iwf_train = get_iwf_for_sents(opt, word2iwf)
    # sent2iwf_train = {}
    opt = valid_opts(opt)
    sent2iwf_valid = get_iwf_for_sents(opt, word2iwf)

    # Merge
    sent2iwf = sent2iwf_train
    for sent,iwf in sent2iwf_valid.items():
        if sent in sent2iwf:
            assert sent2iwf[sent]==iwf
        else:
            sent2iwf[sent] = iwf

    # Get min and max sent iwf
    min_iwf = min(sent2iwf.values())
    max_iwf = max(sent2iwf.values())
    print("min_iwf: %.3f, max_iwf: %.3f" % (min_iwf, max_iwf))

    # Compute niwf for each sent
    sent2niwf = {sent:niwf(iwf, min_iwf, max_iwf) for sent, iwf in sent2iwf.items()}

    return sent2niwf


def niwf(iwf, min_iwf, max_iwf):
    return (iwf - min_iwf)/(max_iwf - min_iwf)


def get_niwf_from_file(opt):
    # Create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    ignorefields = opt.get('ignore_fields', '')

    # Count word frequency for all words in training and validation set
    sent2niwf = {}
    print("Loading sentence niwf from %s..." % opt['datatype'])
    while True:
        world.parley()
        reply = world.acts[0].get('labels', world.acts[0].get('eval_labels'))[0] # string
        niwf = float(world.acts[0]['target_niwf'])
        sent2niwf[reply] = niwf

        if world.epoch_done():
            print('EPOCH DONE')
            break

    return sent2niwf


def load_niwf(opt):
    # load sent2niwf from file
    opt['datatype'] = 'train:ordered'
    opt['fromfile_datapath'] = "/tmp/dump_train"
    sent2niwf_train = get_niwf_from_file(opt)
    opt['datatype'] = 'valid'
    opt['fromfile_datapath'] = "/tmp/dump_valid"
    sent2niwf_valid = get_niwf_from_file(opt)
    sent2niwf = sent2niwf_train
    for sent, niwf in sent2niwf_valid.items():
        sent2niwf[sent] = niwf
    return sent2niwf


def get_niwf_buckets(opt, sent2niwf, num_buckets=2):
    sent2niwf_pairs = [(sent,niwf) for sent,niwf in sent2niwf.items()]
    sent2niwf_pairs = sorted(sent2niwf_pairs, key=lambda x: x[1])
    num_sents = len(sent2niwf_pairs)
    bucket_boundaries = [] # lower niwf boundary for each bucket
    bucket_boundaries_int = [int(i*num_sents/num_buckets) for i in range(num_buckets)]
    for cnt in bucket_boundaries_int:
        (sent, niwf) = sent2niwf_pairs[cnt]
        bucket_boundaries.append(niwf)
        print("(%i/%i): niwf=%.3f, sent=%s" % (cnt, num_sents, niwf, sent))
    return bucket_boundaries



def niwf_to_clusterid(niwf, bucket_boundaries):
    num_buckets = len(bucket_boundaries)
    for clusterid in range(num_buckets-1, -1, -1):
        lower_bound = bucket_boundaries[clusterid]
        if niwf >= lower_bound:
            return clusterid
    raise Exception()
