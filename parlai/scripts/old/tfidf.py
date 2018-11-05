from collections import Counter
import math


def tokenize(str):
    return str.split(' ')

def get_idf(word, clusterid2hashmap, num_clusters):
    num_clusters_containing_w = len([1 for hashmap in clusterid2hashmap.values() if word in hashmap])
    return math.log(num_clusters / num_clusters_containing_w)

def get_tfidfs_wrtsents(clusterid2lst):
    # Convert sents to set of words (no dups)
    clusterid2lst = {clusterid:[list(set(tokenize(s))) for s in sent_lst] for clusterid,sent_lst in clusterid2lst.items()}

    # Get set of all words appearing in each cluster (no dups)
    clusterid2multiset = {clusterid:sorted([w for sent in lst for w in sent]) for clusterid, lst in clusterid2lst.items()} # int to sorted list of words incl repetition
    clusterid2set = {clusterid:sorted(list(set(lst))) for clusterid, lst in clusterid2multiset.items()} # int to sorted list with no repetition

    # Get a list of all words
    all_words = [w for set in clusterid2set.values() for w in set]
    all_words = sorted(list(set(all_words)))

    # Get a list of all sentences
    all_sents = [sent for sent_lst in clusterid2lst.values() for sent in sent_lst]
    num_sents = len(all_sents)

    # Calculate idf for every word in all_words
    print("calculating idf for all words...")
    word2idf = Counter()
    for sent in all_sents:
        word2idf.update(sent)
    word2idf = {word: math.log(num_sents/count) for word,count in word2idf.items()}

    # For each cluster, calculate tfidf for every word in the cluster
    print("calculating tfidfs for each cluster...")
    clusterid2tfidfs = {} # int to Counter mapping word to tfidf
    for clusterid, sent_lst in clusterid2lst.items():
        # print("clusterid %i of %i" % (clusterid, len(clusterid2lst)))
        words = clusterid2set[clusterid] # list of words, no dups
        num_sents_in_cluster = len(sent_lst)

        # Calculate tf for each word
        word2count = Counter()
        for sent in sent_lst:
            word2count.update(sent)
        word2tf = {word: count/num_sents_in_cluster for word, count in word2count.items()}

        # Calculate tfidf for each word
        word2tfidf = Counter({word: tf * word2idf[word] for word, tf in word2tf.items()})
        assert len(word2tfidf) == len(words)
        clusterid2tfidfs[clusterid] = word2tfidf

    return clusterid2tfidfs


def get_tfidfs(clusterid2lst):
    return get_tfidfs_wrtclusters(clusterid2lst)


def get_tfidfs_wrtclusters(clusterid2lst):

    num_clusters = len(clusterid2lst)

    clusterid2multiset = {clusterid:sorted([w for sent in lst for w in tokenize(sent)]) for clusterid, lst in clusterid2lst.items()} # int to sorted list of words incl repetition

    clusterid2set = {clusterid:sorted(list(set(lst))) for clusterid, lst in clusterid2multiset.items()} # int to sorted list with no repetition

    all_words = [w for set in clusterid2set.values() for w in set]
    all_words = sorted(list(set(all_words)))

    clusterid2hashmap = {clusterid:{w:True for w in lst} for clusterid,lst in clusterid2set.items()} # int to a dict which maps word to True

    # calculate idf for every word in all_words
    print("calculating idf for all words...")
    word2idf = Counter({word: get_idf(word, clusterid2hashmap, num_clusters) for word in all_words})

    # for each cluster, calculate tfidf for every word in the cluster
    print("calculating tfidfs for each cluster...")
    clusterid2tfidfs = {} # int to Counter mapping word to tfidf
    for clusterid, set_lst in clusterid2set.items():
        # print("clusterid %i of %i" % (clusterid, num_clusters))
        word2tfidf = Counter()
        multiset = clusterid2multiset[clusterid]
        num_words_in_cluster = len(multiset)


        # === SLOW VERSION ===
        # for word in set_lst:
        #     tf = len([1 for w in multiset if w==word]) / num_words_in_cluster
        #     tfidf = tf * word2idf[word]
        #     word2tfidf[word] = tfidf
        # ====================


        # === FAST VERSION ===

        setpointer = 0
        multisetpointer = 0

        while setpointer < len(set_lst):
            word = set_lst[setpointer]
            assert multiset[multisetpointer] == word
            old_multisetpointer = multisetpointer

            while multiset[multisetpointer] == word:
                multisetpointer += 1
                if multisetpointer == len(multiset):
                    assert setpointer == len(set_lst)-1
                    break

            num_occ = multisetpointer - old_multisetpointer

            tf = num_occ / num_words_in_cluster

            tfidf = tf * word2idf[word]
            word2tfidf[word] = tfidf

            setpointer += 1

        # ====================

        clusterid2tfidfs[clusterid] = word2tfidf

    return clusterid2tfidfs
