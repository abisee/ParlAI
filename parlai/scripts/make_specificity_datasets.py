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
from parlai.scripts.niwf import learn_niwf, get_niwf_buckets, train_opts, valid_opts, niwf_to_clusterid, load_niwf

import random

# def separate_text(text):
#     lines = text.split('\n')
#     persona_lines, last_utterance = [], []
#     for line in lines:
#         if "your persona: " == line[:14]:
#             persona_lines.append(line[14:])
#         else:
#             last_utterance.append(line)
#     assert len(last_utterance)==1
#     return persona_lines, last_utterance[0]
#
#
# def read_clusterfile(cluster_fname, with_dups=False):
#     """
#     if with_dups is True, then clusterid2lst contains duplicates.
#     text2cluster never contains duplicates.
#     """
#     text2cluster = {}
#     clusterid2lst = {}
#     with open(cluster_fname, 'r') as f:
#         for line in f:
#             line = line.strip()
#             cluster_id = line.split(' ')[0] # string
#             text = line[len(cluster_id)+1:]
#             cluster_id = int(cluster_id)
#             text2cluster[text] = cluster_id
#
#             if cluster_id not in clusterid2lst:
#                 clusterid2lst[cluster_id] = []
#
#             if with_dups or (not with_dups and text not in clusterid2lst[cluster_id]):
#                 clusterid2lst[cluster_id].append(text)
#
#     return text2cluster, clusterid2lst



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
    # Get command line arguments
    parser = ParlaiParser()
    opt = parser.parse_args()
    opt['task'] = 'fromfile:parlaiformat'

    # sent2niwf = learn_niwf(opt)
    sent2niwf = load_niwf(opt)

    for num_buckets in [2,3,4,5,10]:
        bucket_boundaries = get_niwf_buckets(opt, sent2niwf, num_buckets)

        opt = train_opts(opt, num_buckets)
        dump_data(opt, sent2niwf, bucket_boundaries)
        opt = valid_opts(opt, num_buckets)
        dump_data(opt, sent2niwf, bucket_boundaries)


if __name__ == '__main__':
    main()
