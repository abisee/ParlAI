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
import extrep
import pickle
import random

rouge_string = "rl"

rougebuckets = {
    "r1": [0.0, 0.05263157894736842, 0.18181818181818182, 0.25, 1.0],
    "r2": [0.0, 0.045454545454545456, 0.08333333333333333, 0.125, 1.0],
    "rl": [0.0, 0.05263157894736842, 0.16666666666666666, 0.23076923076923078, 1.0],
}

buckets = rougebuckets[rouge_string]


def bucket_data(opt):
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    # ignorefields = opt.get('ignore_fields', '')
    ignorefields = 'label_candidates,id'

    print('Writing to %s...' % opt['outfile'])
    fw = open(opt['outfile'], 'w')

    num = 0

    while True:
        world.parley()
        num +=1

        if num % 100 == 0:
            print(num)

        world.acts[0]['labels'] = world.acts[0].get('labels', world.acts[0].pop('eval_labels', None))
        msg = world.acts[0]

        rouge_score = float(msg['target_extrep_'+rouge_string]) # float
        for idx, ub in enumerate(buckets):
            if rouge_score <= ub:
                bucket_id = idx
                break
        assert(bucket_id in range(5))

        msg['target_clusterid'] = bucket_id

        txt = msg_to_str(msg, ignore_fields=ignorefields)
        fw.write(txt + '\n')

        if msg.get('episode_done'):
            fw.write('\n')

        if world.epoch_done():
            print('EPOCH DONE')
            break

    fw.close()


def dump_data(opt):
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    # ignorefields = opt.get('ignore_fields', '')
    ignorefields = 'label_candidates,id'

    print('Writing to %s...' % opt['outfile'])
    fw = open(opt['outfile'], 'w')

    persona = []
    own_utts = []
    partner_utts = []

    num = 0

    while True:
        world.parley()
        num +=1

        if num % 100 == 0:
            print(num)

        world.acts[0]['labels'] = world.acts[0].get('labels', world.acts[0].pop('eval_labels', None))
        msg = world.acts[0]
        reply = msg.get('labels', world.acts[0].get('eval_labels'))[0] # string

        text = msg['text'].split('\n')
        if len(text)>1:
            for s in text[:-1]:
                assert "your persona: " == s[:14]
                if "." == s[-1] and " " != s[-2]:
                    s = s[:-1] + " ."
                persona.append(s[14:])
            text = [text[-1]]
        assert len(text)==1
        text = text[0]
        partner_utts.append(text)

        max_r1, max_r2, max_rl = extrep.get_extrep(reply, own_utts+partner_utts)

        own_utts.append(reply)

        msg['target_extrep_r1'] = max_r1
        msg['target_extrep_r2'] = max_r2
        msg['target_extrep_rl'] = max_rl

        txt = msg_to_str(msg, ignore_fields=ignorefields)

        fw.write(txt + '\n')

        if msg.get('episode_done'):
            persona = []
            own_utts = []
            partner_utts = []
            fw.write('\n')

        if world.epoch_done():
            print('EPOCH DONE')
            break

    fw.close()


def main():
    random.seed(42)

    parser = ParlaiParser()
    parser.add_argument('-n', '--num-examples', default=1000000000, type=int)
    parser.add_argument('-of', '--outfile', default='/tmp/dump', type=str)
    parser.add_argument('-if', '--ignore-fields', default='id', type=str)
    parser.set_defaults(datatype='train:stream')
    opt = parser.parse_args()
    opt['task'] = 'fromfile:parlaiformat'

    # MAKE R1/R2/RL DATASETS

    # opt['datatype'] = 'train:ordered'
    # opt['fromfile_datapath'] = "/private/home/abisee/ParlAI/data/ConvAI2_parlaiformat/train_self_original.txt"
    # opt['outfile'] = "/private/home/abisee/ParlAI/data/ConvAI2_extrep/train_r1r2rl.txt"

    # opt['datatype'] = 'valid'
    # opt['fromfile_datapath'] = "/private/home/abisee/ParlAI/data/ConvAI2_parlaiformat/valid_self_original.txt"
    # opt['outfile'] = "/private/home/abisee/ParlAI/data/ConvAI2_extrep/valid_r1r2rl.txt"

    # dump_data(opt)


    # MAKE BUCKETED DATASETS
    opt['datatype'] = 'valid'
    opt['fromfile_datapath'] = "/private/home/abisee/ParlAI/data/ConvAI2_extrep/valid_r1r2rl.txt"
    opt['outfile'] = "/private/home/abisee/ParlAI/data/ConvAI2_extrep/valid_%s_5buckets.txt" % rouge_string
    bucket_data(opt)

    opt['datatype'] = 'train:ordered'
    opt['fromfile_datapath'] = "/private/home/abisee/ParlAI/data/ConvAI2_extrep/train_r1r2rl.txt"
    opt['outfile'] = "/private/home/abisee/ParlAI/data/ConvAI2_extrep/train_%s_5buckets.txt" % rouge_string
    bucket_data(opt)



if __name__ == '__main__':
    main()
