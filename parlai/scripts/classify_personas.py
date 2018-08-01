from parlai_internal.projects.nlg_plan.eval_starspace_classifier import classify
from parlai_internal.projects.nlg_plan.cluster_classifier import load_model, load_centroids_directly
import torch

model = load_model()
cluster_centers = load_centroids_directly()
cluster_centers = torch.Tensor(cluster_centers)
k = cluster_centers.size(0)

def read_cands(candf):
    f = open(candf)
    cands = f.read().replace('\n\n','\n').split('\n')
    cands = [c for c in cands if c!=""]
    return cands

candf_train = "/private/home/abisee/ParlAI/data/ConvAI2_clusters/clusters/train_personas.txt"
candf_valid = "/private/home/abisee/ParlAI/data/ConvAI2_clusters/clusters/valid_personas.txt"

cands = []
cands += read_cands(candf_train)
cands += read_cands(candf_valid)

personas = cands

_, ranking = classify(personas, model, "output", cluster_centers) # shape (num_exs, k)
labels = ranking[:, 0].tolist() # len num_exs

outfile = "/private/home/abisee/ParlAI/data/ConvAI2_clusters/clusters/persona_clusters_200.txt"
with open(outfile, "w") as fw:
    for clusterid in range(k):
        for (p, id) in zip(personas, labels):
            if id==clusterid:
                fw.write(str(id) + " " + p + "\n")
