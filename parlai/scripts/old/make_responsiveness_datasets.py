# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Convert a dataset into the ParlAI text format.
E.g.:
`python convert_data_to_parlai_format.py -t babi:task1k:1 --outfile /tmp/dump `
"""

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.core.utils import msg_to_str
import pickle
import random


import nltk
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english') + [".", "?", "!", ","]


def contains_user_word(reply, last_utt):
    reply = [t for t in reply.split() if t not in STOPWORDS]
    last_utt = [t for t in last_utt.split() if t not in STOPWORDS]
    for t in reply:
        if t in last_utt:
            return True
    return False


def dump_data(opt):
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    # ignorefields = opt.get('ignore_fields', '')
    ignorefields = 'label_candidates,id'

    print('Writing to %s...' % opt['outfile'])
    fw = open(opt['outfile'], 'w')

    num_resp = 0
    num_nonresp = 0

    num = 0

    while True:
        num += 1
        if num % 100 == 0:
            print(num)

        world.parley()
        world.acts[0]['labels'] = world.acts[0].get('labels', world.acts[0].pop('eval_labels', None))
        msg = world.acts[0]
        reply = msg.get('labels', world.acts[0].get('eval_labels'))[0] # string
        last_utt = msg['text'].split('\n')[-1]

        cont_user_word = contains_user_word(reply, last_utt) # bool

        if cont_user_word:
            num_resp += 1
        else:
            num_nonresp += 1

        # import pdb; pdb.set_trace()

        msg['target_clusterid'] = int(cont_user_word) # 0 or 1

        txt = msg_to_str(msg, ignore_fields=ignorefields)

        fw.write(txt + '\n')

        if msg['episode_done']:
            fw.write('\n')

        if world.epoch_done():
            print('EPOCH DONE')
            break

    fw.close()

    print("num_resp: ", num_resp)
    print("num_nonresp: ", num_nonresp)


def main():
    random.seed(42)

    parser = ParlaiParser()
    parser.add_argument('-n', '--num-examples', default=1000000000, type=int)
    parser.add_argument('-of', '--outfile', default='/tmp/dump', type=str)
    parser.add_argument('-if', '--ignore-fields', default='id', type=str)
    parser.set_defaults(datatype='train:stream')
    opt = parser.parse_args()
    opt['task'] = 'fromfile:parlaiformat'

    # MAKE BUCKETED DATASETS
    opt['datatype'] = 'valid'
    opt['fromfile_datapath'] = "/private/home/abisee/ParlAI/data/ConvAI2_parlaiformat/valid_self_original.txt"
    opt['outfile'] = "/private/home/abisee/ParlAI/data/ConvAI2_responsiveness/valid_contuserword.txt"
    dump_data(opt)

    opt['datatype'] = 'train:ordered'
    opt['fromfile_datapath'] = "/private/home/abisee/ParlAI/data/ConvAI2_parlaiformat/train_self_original.txt"
    opt['outfile'] = "/private/home/abisee/ParlAI/data/ConvAI2_responsiveness/train_contuserword.txt"
    dump_data(opt)


if __name__ == '__main__':
    main()
