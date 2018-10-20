from parlai.scripts.eval_wordstat import setup_args, eval_wordstat

def set_defaults(parser):

    parser.set_defaults(
        task="fromfile:parlaiformat",
        fromfile_datapath="/private/home/abisee/ParlAI/data/ConvAI2_controlled/valid_question_contuserword_niwf10buckets.txt",
        model="parlai_internal.projects.seq2plan2seq.controlled_seq2seq.controlled_seq2seq:ControlledSeq2seqAgent",
        batch_size=64,
    )
    return parser


if __name__ == '__main__':
    parser = setup_args()
    parser = set_defaults(parser)
    opt = parser.parse_args()
    if opt['beam_size'] == 1:
        opt['batchsize'] = 64
        opt['override']['batchsize'] = 64
    eval_wordstat(opt)
