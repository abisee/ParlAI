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

def separate_text(text):
    lines = text.split('\n')
    persona_lines, last_utterance = [], []
    for line in lines:
        if "your persona: " == line[:14]:
            persona_lines.append(line[14:])
        else:
            last_utterance.append(line)
    assert len(last_utterance)==1
    return persona_lines, last_utterance[0]


def read_clusterfile(cluster_fname, with_dups=False):
    """
    if with_dups is True, then clusterid2lst contains duplicates.
    text2cluster never contains duplicates.
    """
    text2cluster = {}
    clusterid2lst = {}
    with open(cluster_fname, 'r') as f:
        for line in f:
            line = line.strip()
            cluster_id = line.split(' ')[0] # string
            text = line[len(cluster_id)+1:]
            cluster_id = int(cluster_id)
            text2cluster[text] = cluster_id

            if cluster_id not in clusterid2lst:
                clusterid2lst[cluster_id] = []

            if with_dups or (not with_dups and text not in clusterid2lst[cluster_id]):
                clusterid2lst[cluster_id].append(text)

    return text2cluster, clusterid2lst

def dump_data(opt):
    cluster_fname = "/private/home/abisee/ParlAI/data/ConvAI2_clusters/clusters/input_target_clusters_200.txt"
    persona_cluster_fname = "/private/home/abisee/ParlAI/data/ConvAI2_clusters/clusters/persona_clusters_200.txt"
    text2cluster, _ = read_clusterfile(cluster_fname)
    persona2cluster, _ = read_clusterfile(persona_cluster_fname)

    num_clusters = 200

    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    ignorefields = opt.get('ignore_fields', '')

    print('[ starting to convert.. ]')
    fw = open(opt['outfile'], 'w')
    for _ in range(opt['num_examples']):

        episode_persona = None
        dialog_history = []
        label = None

        while True:
            world.parley()
            world.acts[0]['labels'] = world.acts[0].get(
                'labels', world.acts[0].pop('eval_labels', None))
            msg = world.acts[0]
            text = msg['text']
            persona_lines, last_utterance = separate_text(text)

            if episode_persona is None:
                assert len(persona_lines) > 0
                episode_persona = persona_lines
            if label is None: # first utterance of episode
                dialog_history += [last_utterance]
            else:
                dialog_history += [label, last_utterance]

            label = msg['labels']
            assert len(label)==1
            label = label[0] # str

            target_clusterid = text2cluster[label]
            msg['target_clusterid'] = target_clusterid
            msg['persona_clusterids'] = [persona2cluster[l] for l in episode_persona]
            msg['dialog_hist_clusterids'] = [text2cluster[l] for l in dialog_history]

            txt = msg_to_str(msg, ignore_fields=ignorefields)

            fw.write(txt + '\n')

            if msg['episode_done']:
                fw.write('\n')
                break

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
