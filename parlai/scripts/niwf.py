from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from collections import Counter
import math
import pickle

def train_opts(num_buckets=None):
    """Opts for reading the orig train file"""
    opt = {}
    opt['task'] = 'fromfile:parlaiformat'
    opt['datatype'] = 'train:ordered'
    opt['fromfile_datapath'] = "/private/home/abisee/ParlAI/data/ConvAI2_parlaiformat/train_self_original.txt"
    return opt

def valid_opts(num_buckets=None):
    """Opts for reading the orig valid file"""
    opt = {}
    opt['task'] = 'fromfile:parlaiformat'
    opt['datatype'] = 'valid'
    opt['fromfile_datapath'] = "/private/home/abisee/ParlAI/data/ConvAI2_parlaiformat/valid_self_original.txt"
    return opt

def get_word_counts(opt):
    # Create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    ignorefields = opt.get('ignore_fields', '')

    # Count word frequency for all words in dataset
    # Also collect list of sentences
    word_counter = Counter()
    sents = []
    print("Getting word counts from %s..." % opt['datatype'])
    while True:
        world.parley()
        text = world.acts[0]['text']
        text = text.split('\n')[-1] # string
        reply = world.acts[0].get('labels', world.acts[0].get('eval_labels'))[0] # string

        words = reply.split()
        words = list(set(words))
        word_counter.update(words)
        sents.append(reply)

        if world.epoch_done():
        # if num_sents > 100:
            print('EPOCH DONE')
            break

    print("num_sents: ", len(sents))
    print("most common: ", word_counter.most_common(10))

    return sents, word_counter


def get_iwf(sent, word2iwf):
    """Returns:
        iwf: iwf of sentence, using all words that are in the dictionary
        problem_words: words in the sentence that aren't in the dictionary
    """
    words = sent.split()
    words = list(set(words))
    problem_words = [w for w in words if w not in word2iwf]
    ok_words = [w for w in words if w in word2iwf]
    iwf = max([word2iwf[w] for w in ok_words])
    return iwf, problem_words


def learn_niwf():
    """Go through data and calculate NIWF for every sentence; also return sent2niwf function"""
    # get word counts from train and val sets
    opt = train_opts()
    sents_train, word_counter_train = get_word_counts(opt)
    opt = valid_opts()
    sents_valid, word_counter_valid = get_word_counts(opt)

    # merge word counts and sent list
    num_sents = len(sents_train) + len(sents_valid)
    word_counter = word_counter_train
    for word,count in word_counter_valid.items():
        word_counter[word] += count
    sents = sents_valid + sents_train

    # Compute IWF for every word
    print("Computing IWF for all words...")
    word2iwf = {}
    nom = math.log(1 + num_sents)
    for word, count in word_counter.items():
        word2iwf[word] = nom/count

    # Compute IWF for every sentence in train + val
    print("Computing IWF for all sentences in train and val sets...")
    sent2iwf = {}
    for sent in sents:
        iwf, problem_words = get_iwf(sent, word2iwf)
        assert len(problem_words)==0
        sent2iwf[sent] = iwf

    # Get min and max sent iwf
    min_iwf = min(sent2iwf.values())
    max_iwf = max(sent2iwf.values())
    print("min_iwf: %.3f, max_iwf: %.3f" % (min_iwf, max_iwf))

    # Compute niwf for each sent
    print("Computing NIWF for all sentences in train and val sets...")
    sent2niwf_dict = {sent:get_niwf(iwf, min_iwf, max_iwf) for sent, iwf in sent2iwf.items()}

    # Write word2iwf, min_iwf, max_iwf to pickle file
    outfile = "/private/home/abisee/ParlAI/data/ConvAI2_specificity/word2iwf.pkl"
    data = {
        "word2iwf": word2iwf,
        "min_iwf": min_iwf,
        "max_iwf": max_iwf,
    }
    with open(outfile, "wb") as f:
        pickle.dump(data, f)

    # This fn uses word2iwf, min_iwf, max_iwf
    def sent2niwf_fn(sent):
        sent_iwf, problem_words = get_iwf(sent, word2iwf)
        sent_niwf = get_niwf(sent_iwf, min_iwf, max_iwf)
        return sent_niwf, problem_words

    return sent2niwf_dict, sent2niwf_fn


def get_niwf(iwf, min_iwf, max_iwf):
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


def load_niwf_fn():
    """Load niwf function from file"""
    outfile = "/private/home/abisee/ParlAI/data/ConvAI2_specificity/word2iwf.pkl"
    print("Loading NIWF function from %s..." % outfile)
    with open(outfile, "rb") as f:
        data = pickle.load(f)
    word2iwf = data['word2iwf']
    min_iwf = data['min_iwf']
    max_iwf = data['max_iwf']

    # This fn uses word2iwf, min_iwf, max_iwf
    def sent2niwf_fn(sent):
        sent_iwf, problem_words = get_iwf(sent, word2iwf)
        sent_niwf = get_niwf(sent_iwf, min_iwf, max_iwf)
        return sent_niwf, problem_words

    return sent2niwf_fn


def load_niwf_dict():
    """Load sent2niwf dict from file"""
    opt = {}
    opt['task'] = 'fromfile:parlaiformat'

    # load train niwf
    opt['datatype'] = 'train:ordered'
    opt['fromfile_datapath'] = "/private/home/abisee/ParlAI/data/ConvAI2_specificity/train_self_original_classification_2buckets.txt"
    sent2niwf_train = get_niwf_from_file(opt)

    # load valid niwf
    opt['datatype'] = 'valid'
    opt['fromfile_datapath'] = "/private/home/abisee/ParlAI/data/ConvAI2_specificity/valid_self_original_classification_2buckets.txt"
    sent2niwf_valid = get_niwf_from_file(opt)

    # merge
    sent2niwf = sent2niwf_train
    for sent, niwf in sent2niwf_valid.items():
        sent2niwf[sent] = niwf
    return sent2niwf


def load_niwf_buckets(num_buckets):
    """Load the NIWF buckets from file"""
    bucket_outfile = '/private/home/abisee/ParlAI/data/ConvAI2_specificity/%ibucket_boundaries.txt' % (num_buckets)
    print("Loading NIWF bucket boundaries from %s..." % bucket_outfile)
    with open(bucket_outfile, "rb") as f:
        bucket_boundaries = pickle.load(f)
    assert len(bucket_boundaries)==num_buckets
    return bucket_boundaries


def get_niwf_buckets(sent2niwf, num_buckets=2):
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
