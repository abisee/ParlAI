import torch
from parlai.core.agents import create_agent
# from torch.utils.model_zoo import load_url
# from torchvision import models

def load_model(modelfile):
    print("Loading model...")
    opt = {}
    opt['model_file'] = modelfile
    agent = create_agent(opt)
    return agent


def save_model(state_dict, savepath):
    print("Saving to %s..." % savepath)
    torch.save(state_dict, savepath)


def rename_weights(state_dict, orig_key, new_key):
    print("renaming %s to %s" % (orig_key, new_key))
    state_dict[new_key] = state_dict[orig_key]
    del state_dict[orig_key]
    return state_dict




modelfile_old = "/private/home/abisee/models/input2cluster_classifier_2layer_concatspsbefore1stand2ndmlp"

modelfile_new = "/private/home/abisee/models/input2cluster_classifier_2layer_concatspsbefore1stand2ndmlp_copy"

old_model = torch.load(modelfile_old)

new_agent = load_model(modelfile_new)

sd = old_model['model']

for l in ["first_mlp", "second_mlp"]:
    for (i,j) in [(0,0),(2,1)]:
        for w in ["weight", "bias"]:
            orig_key = "classifier.%s.%i.%s" % (l, i, w)
            new_key = "classifier.%s.%i.lin.%s" % (l, j, w)
            sd[new_key] = sd[orig_key]
            del sd[orig_key]

assert sorted(new_agent.model.state_dict().keys()) == sorted(sd.keys())

new_agent.model.load_state_dict(sd)

new_agent.save()


import pdb; pdb.set_trace()
