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
from parlai_internal.projects.seq2plan2seq.controlled_seq2seq.niwf import load_word2iwf, get_niwf
import pickle

import random


def dump_data(opt, sent2niwf, bucket_boundaries):
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    # ignorefields = opt.get('ignore_fields', '')
    ignorefields = 'label_candidates'

    print('Writing to %s...' % opt['outfile'])
    fw = open(opt['outfile'], 'w')
    while True:
        world.parley()
        world.acts[0]['labels'] = world.acts[0].get('labels', world.acts[0].pop('eval_labels', None))
        msg = world.acts[0]
        reply = msg.get('labels', world.acts[0].get('eval_labels'))[0] # string
        reply_niwf = sent2niwf[reply] # float

        # ====================================
        # run this to check that our niwf calculations match what's written in the datafile
        # ====================================
        # try:
        #     assert float(msg['target_niwf']) == reply_niwf
        #     assert int(msg['target_clusterid']) == niwf_to_clusterid(reply_niwf, bucket_boundaries)
        # except:
        #     print("assertion error")
        #     import pdb; pdb.set_trace()
        # ====================================

        msg['target_niwf'] = reply_niwf
        msg['target_clusterid'] = niwf_to_clusterid(reply_niwf, bucket_boundaries)

        txt = msg_to_str(msg, ignore_fields=ignorefields)

        fw.write(txt + '\n')

        if world.epoch_done():
            print('EPOCH DONE')
            break

    fw.close()


def main():
    random.seed(42)

    word2iwf, min_iwf, max_iwf = load_word2iwf()

    for num_buckets in [10]:
        # get bucket boundaries and write to file
        bucket_boundaries = get_niwf_buckets(sent2niwf, num_buckets)
        bucket_outfile = '/private/home/abisee/ParlAI/data/ConvAI2_specificity/%ibucket_boundaries.txt' % (num_buckets)
        with open(bucket_outfile, "wb") as f:
            pickle.dump(bucket_boundaries, f)

        # write train file
        opt = train_opts(num_buckets)
        opt['outfile'] = '/private/home/abisee/ParlAI/data/ConvAI2_specificity/train_self_original_classification_%ibuckets.txt' % (num_buckets)
        dump_data(opt, sent2niwf, bucket_boundaries)

        # write valid file
        opt = valid_opts(num_buckets)
        opt['outfile'] = '/private/home/abisee/ParlAI/data/ConvAI2_specificity/valid_self_original_classification_%ibuckets.txt' % (num_buckets)
        dump_data(opt, sent2niwf, bucket_boundaries)


if __name__ == '__main__':
    main()
