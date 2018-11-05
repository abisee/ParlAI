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


def contains_question(reply):
    return "?" in reply


def dump_data(opt):
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    # ignorefields = opt.get('ignore_fields', '')
    ignorefields = 'label_candidates,id'

    print('Writing to %s...' % opt['outfile'])
    fw = open(opt['outfile'], 'w')

    num_qs = 0
    num_nonqs = 0

    num = 0

    while True:
        num += 1
        if num % 100 == 0:
            print(num)

        world.parley()
        world.acts[0]['labels'] = world.acts[0].get('labels', world.acts[0].pop('eval_labels', None))
        msg = world.acts[0]
        reply = msg.get('labels', world.acts[0].get('eval_labels'))[0] # string

        cont_question = contains_question(reply) # bool

        if cont_question:
            num_qs += 1
        else:
            num_nonqs += 1

        # import pdb; pdb.set_trace()

        msg['target_clusterid'] = int(cont_question) # 0 or 1

        txt = msg_to_str(msg, ignore_fields=ignorefields)

        fw.write(txt + '\n')

        if msg['episode_done']:
            fw.write('\n')

        if world.epoch_done():
            print('EPOCH DONE')
            break

    fw.close()

    print("num_qs: ", num_qs)
    print("num_nonqs: ", num_nonqs)


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
    opt['outfile'] = "/private/home/abisee/ParlAI/data/ConvAI2_questions/valid.txt"
    dump_data(opt)

    opt['datatype'] = 'train:ordered'
    opt['fromfile_datapath'] = "/private/home/abisee/ParlAI/data/ConvAI2_parlaiformat/train_self_original.txt"
    opt['outfile'] = "/private/home/abisee/ParlAI/data/ConvAI2_questions/train.txt"
    dump_data(opt)


if __name__ == '__main__':
    main()
