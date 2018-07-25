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

import random


def read_clusterfile(cluster_fname):
    text2cluster = {}
    with open(cluster_fname, 'r') as f:
        for line in f:
            line = line.strip()
            cluster_id = line.split(' ')[0] # string
            text = line[len(cluster_id)+1:]
            cluster_id = int(cluster_id)
            text2cluster[text] = cluster_id

    clusterid2lst = {}
    for text, clusterid in text2cluster.items():
        if clusterid not in clusterid2lst:
            clusterid2lst[clusterid] = []
        clusterid2lst[clusterid].append(text)

    return text2cluster, clusterid2lst

def dump_data(opt):
    cluster_fname = "/private/home/abisee/ParlAI/data/ConvAI2_clusters/clusters/input_target_clusters_200.txt"
    text2cluster, _ = read_clusterfile(cluster_fname)

    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    ignorefields = opt.get('ignore_fields', '')

    print('[ starting to convert.. ]')
    fw = open(opt['outfile'], 'w')
    for _ in range(opt['num_examples']):
        world.parley()
        world.acts[0]['labels'] = world.acts[0].get(
            'labels', world.acts[0].pop('eval_labels', None))

        label = world.acts[0]['labels']
        assert len(label)==1
        label = label[0]
        target_clusterid = text2cluster[label]
        world.acts[0]['target_clusterid'] = target_clusterid

        text = world.acts[0]['text'].split('\n')[-1]
        input_clusterid = text2cluster[text]
        world.acts[0]['input_clusterid'] = input_clusterid

        # target_clusterid_token = "CLUSTER_%i" % target_clusterid
        # world.acts[0]['text'] += " " + target_clusterid_token

        txt = msg_to_str(world.acts[0], ignore_fields=ignorefields)
        fw.write(txt + '\n')
        if world.acts[0].get('episode_done', False):
            fw.write('\n')

        if world.epoch_done():
            print('EPOCH DONE')
            break
    fw.close()


def main():
    random.seed(42)
    # Get command line arguments
    parser = ParlaiParser()
    parser.add_argument('-n', '--num-examples', default=1000000000, type=int)
    parser.add_argument('-of', '--outfile', default='/tmp/dump', type=str)
    parser.add_argument('-if', '--ignore-fields', default='id', type=str)
    parser.set_defaults(datatype='train:stream')
    opt = parser.parse_args()
    dump_data(opt)


if __name__ == '__main__':
    main()
