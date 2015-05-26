# Word2Vec

[![Build Status](https://travis-ci.org/tanmaykm/Word2Vec.jl.svg?branch=master)](https://travis-ci.org/tanmaykm/Word2Vec.jl)

- Create an instance of `WordEmbedding`: `embed = WordEmbedding(100, Word2Vec.random_inited, Word2Vec.huffman_tree, subsampling = subsampling)`
- To train sequentially: `train(embed, inputfile)`
- Alternatively, to train in parallel:
    - add worker nodes: `addprocs(N)`
    - chunk the input file using `Blocks`: `b = Block(File(inputfile), nworkers())`
    - start training the chunks, also provide a filename that will be used to exchange data between workers and master node: `train(embed, b, "/tmp/emb")`
- After successful training, query for similar words: `find_nearest_words(embed, "query words")`

This is still work in progress. Parallel training with weight averaging does not yeild very good results. May need to implement asynchronous stochastic gradient descent used by Mikolov 2013.

## Datasets
- Google Code page for word2vec lists many sources: https://code.google.com/p/word2vec/
- Matt Mahoney's page: http://mattmahoney.net/dc/textdata.html
- GoLearn github repository: https://github.com/sjwhitworth/golearn/tree/master/examples/datasets

## Credits
This is based on this original code by Zhixuan Yang (https://github.com/yangzhixuan/embed)
