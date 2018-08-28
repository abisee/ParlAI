for i in range(10):
    print("specificity bucket %i:" % i)
    with open("/private/home/abisee/models/seq2seq_twitterpretrained_specificityclusters_10buckets.valid.wordstats.fixed_clusterid%i" % i, "r") as f:
        text = f.read()
    lines = text.split('\n')
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l != "" and "Faithfulness" not in l]
    for l in lines:
        print(l)
