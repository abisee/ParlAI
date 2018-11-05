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


def dump_data(opt, datasets):
    primary_key = list(datasets.keys())[0]

    key2world = {}
    for key, datapath in datasets.items():
        opt['fromfile_datapath'] = datapath
        agent = RepeatLabelAgent(opt)
        world = create_task(opt, agent)
        key2world[key] = world

    print('Writing to %s...' % opt['outfile'])
    fw = open(opt['outfile'], 'w')
    num = 0
    ignorefields = 'label_candidates,id'

    while True:
        num += 1
        if num % 100 == 0:
            print(num)

        key2msg = {}

        for key,world in key2world.items():
            world.parley()
            world.acts[0]['labels'] = world.acts[0].get('labels', world.acts[0].pop('eval_labels', None))
            # print("")
            # print("on step %i %s gave this message: " % (num, key))
            msg = world.acts[0]
            # print(msg)
            key2msg[key] = msg

        # check the text of the messages are all the same
        main_msg = key2msg[primary_key]
        texts = [msg['text'] for key,msg in key2msg.items()]
        assert all([t==main_msg['text'] for t in texts])

        for key,msg in key2msg.items():
            main_msg[key] = msg['target_clusterid']

        del main_msg['target_clusterid']

        txt = msg_to_str(main_msg, ignore_fields=ignorefields)

        fw.write(txt + '\n')

        if main_msg['episode_done']:
            fw.write('\n')

        if world.epoch_done():
            print('EPOCH DONE')
            break

    fw.close()


def main():
    random.seed(42)

    datasets = {
        "question": "/private/home/abisee/ParlAI/data/ConvAI2_questions/valid.txt",
        "niwf_10buckets": "/private/home/abisee/ParlAI/data/ConvAI2_specificity/valid_self_original_classification_10buckets.txt",
        "contuserword": "/private/home/abisee/ParlAI/data/ConvAI2_responsiveness/valid_contuserword.txt",
    }

    opt = {}
    opt['task'] = 'fromfile:parlaiformat'
    opt['datatype'] = 'valid'
    opt['outfile'] = "/private/home/abisee/ParlAI/data/ConvAI2_controlled/valid_question_contuserword_niwf10buckets.txt"

    dump_data(opt, datasets)

    datasets = {
        "question": "/private/home/abisee/ParlAI/data/ConvAI2_questions/train.txt",
        "niwf_10buckets": "/private/home/abisee/ParlAI/data/ConvAI2_specificity/train_self_original_classification_10buckets.txt",
        "contuserword": "/private/home/abisee/ParlAI/data/ConvAI2_responsiveness/train_contuserword.txt",
    }

    opt = {}
    opt['task'] = 'fromfile:parlaiformat'
    opt['datatype'] = 'train:ordered'
    opt['outfile'] = "/private/home/abisee/ParlAI/data/ConvAI2_controlled/train_question_contuserword_niwf10buckets.txt"

    dump_data(opt, datasets)


if __name__ == '__main__':
    main()
