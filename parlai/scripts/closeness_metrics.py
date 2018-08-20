import torch
from parlai_internal.projects.nlg_plan.cluster_classifier import embed_sentences


def init_closeness_metrics():
    closeness_metrics = {
        "min_persona_dist": 0.0,
        "last_utt_dist": 0.0,
        "min_hist_dist": 0.0,
        "num_exs": 0.0,
        "persona_count": 0,
        "last_utt_count": 0,
        "other_hist_count": 0,
    }
    return closeness_metrics

def add_to_dialoghist(agent_1, act_0):
    input_text = act_0['text'].split('\n')
    persona_lines = [line for line in input_text if "your persona: "==line[:14]]
    if len(persona_lines)>0:
        assert len(persona_lines) == len(input_text)-1
        agent_1.dialoghist_persona = persona_lines
        agent_1.dialoghist_convo = []
    last_utt = input_text[-1]
    agent_1.dialoghist_convo.append(last_utt)


def calc_closeness(prediction, persona, convo, starspace_model, dist_metric="eucl"):
    # Compute dist between prediction and each thing in dialoghist_persona and dialoghist_convo
    pred_emb = torch.Tensor(embed_sentences([prediction], starspace_model, type="output"))
    persona_emb = torch.Tensor(embed_sentences(persona, starspace_model, type="output"))
    hist_emb = torch.Tensor(embed_sentences(convo, starspace_model, type="output"))

    if dist_metric=="eucl":
        persona_dists = torch.nn.functional.pairwise_distance(pred_emb, persona_emb) # shape (persona_len)
        hist_dists = torch.nn.functional.pairwise_distance(pred_emb, hist_emb) # shape (hist_len)
    elif dist_metric=="cosine":
        def cosine_dist(x1, x2):
            # x1 and x2 are shape (len1, dim) and (len2, dim) respectively
            len1 = x1.size(0)
            len2 = x2.size(0)
            x1 = x1.unsqueeze(2).repeat(1,1,len2) # shape (len1, dim, len2)
            x2 = torch.transpose(x2, 0, 1).unsqueeze(0).repeat(len1, 1, 1) # shape (len1, dim, len2)
            return -torch.nn.functional.cosine_similarity(x1, x2) # shape (len1, len2)
        persona_dists = cosine_dist(pred_emb, persona_emb).squeeze(0) # shape (persona_len)
        hist_dists = cosine_dist(pred_emb, hist_emb).squeeze(0) # shape (hist_len)
    else:
        raise Exception()

    min_persona_dist = torch.min(persona_dists).item()
    min_hist_dist = torch.min(hist_dists).item()
    last_utt_dist = hist_dists[-1].item()

    if min_persona_dist < min_hist_dist:
        choice = "persona"
    elif last_utt_dist == min_hist_dist:
        assert last_utt_dist < min_persona_dist
        choice = "last_utt"
    else:
        assert min_hist_dist < min(last_utt_dist, min_persona_dist)
        choice = "other_hist"

    # print("")
    # print(dist_metric)
    # least = min(min_persona_dist, min_hist_dist)

    # print(choice)

    # for (l,d) in zip(agent_1.dialoghist_persona, persona_dists):
    #     blah = "***" if d==least else "   "
    #     print("%s %2f %s" % (blah, d.item(), l))
    # for (l,d) in zip(agent_1.dialoghist_convo, hist_dists):
    #     blah = "***" if d==least else "   "
    #     print("%s %2f %s" % (blah, d.item(), l))
    # print("prediction: ", prediction)
    # print("")

    # import pdb; pdb.set_trace()

    return choice, min_persona_dist, min_hist_dist, last_utt_dist


def update_closeness_metrics(prediction, agent_1, closeness_metrics, starspace_model, dist_metric="eucl"):
    choice, min_persona_dist, min_hist_dist, last_utt_dist = calc_closeness(prediction, agent_1.dialoghist_persona, agent_1.dialoghist_convo, starspace_model, dist_metric=dist_metric)

    closeness_metrics['min_persona_dist'] += min_persona_dist
    closeness_metrics['last_utt_dist'] += last_utt_dist
    closeness_metrics['min_hist_dist'] += min_hist_dist
    closeness_metrics[choice+"_count"] += 1
    closeness_metrics['num_exs'] += 1

    return closeness_metrics, choice


def show_closeness_metrics(closeness_metrics):
    num_exs = closeness_metrics['num_exs']
    closeness_metrics = {k:v/num_exs for k,v in closeness_metrics.items() if k!="num_exs"}
    print("Closeness stats from %i examples:" % (num_exs))
    print("avg last utt dist: %.4f, avg min persona dist: %.4f, avg min dialog dist: %.4f" % (closeness_metrics['last_utt_dist'], closeness_metrics['min_persona_dist'], closeness_metrics['min_hist_dist']))
    print("Counts: %.2f%% persona / %.2f%% last / %.2f%% other dialoghist" % (closeness_metrics['persona_count']*100, closeness_metrics['last_utt_count']*100, closeness_metrics['other_hist_count']*100))
    print("")
