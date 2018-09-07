import sys
import pickle


if __name__=="__main__":
    assert len(sys.argv)==2
    optfile = sys.argv[1]
    with open(optfile, 'rb') as handle:
        new_opt = pickle.load(handle)

    new_opt['dict_file'] = "/checkpoint/abisee/jase/all.dict"

    for key in sorted(new_opt.keys()):
        print(key, new_opt[key])

    with open(optfile+'.new', 'wb') as handle:
        pickle.dump(new_opt, handle)