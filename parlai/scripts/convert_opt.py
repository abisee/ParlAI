import sys
import pickle
import json

if __name__=="__main__":
    assert len(sys.argv)==2
    optfile = sys.argv[1]
    with open(optfile, 'rb') as handle:
        opt = pickle.load(handle)
    import pdb; pdb.set_trace()
    with open(optfile+'.json', 'w') as f:
        json.dump(opt, f)
