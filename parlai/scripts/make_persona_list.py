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
from collections import Counter

import random


def dump_data(opt):
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    ignorefields = opt.get('ignore_fields', '')

    all_persona_lines = Counter()

    print('[ starting to convert.. ]')
    for _ in range(opt['num_examples']):
        world.parley()

        text = world.acts[0]['text']
        lines = text.split('\n')
        persona_lines = [l for l in lines if "your persona: "==l[:14]]
        persona_lines = [l[14:] for l in persona_lines]
        all_persona_lines.update(persona_lines)

        if world.epoch_done():
            print('EPOCH DONE')
            break

    with open(opt['outfile'], 'w') as fw:
        for persona in all_persona_lines.keys():
            fw.write(persona + "\n")


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
