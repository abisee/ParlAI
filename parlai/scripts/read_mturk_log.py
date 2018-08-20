import pickle
import sys


if __name__=="__main__":
    assert len(sys.argv) == 2
    fname = sys.argv[1]
    with open(fname, "rb") as f:
        data = pickle.load(f)
    print("mturk log loaded as 'data'")
    import pdb; pdb.set_trace()
