# Attention Word Embeddings

## Original README CONTENT :

The code is inspired from the following [github repository](https://github.com/florianmai/word2mat). 

*AWE* is designed to learn rich word vector representations. It fuses the attention mechanism with the CBOW model of word2vec to address the limitations of the CBOW model. CBOW equally weights the context words when making the masked word prediction, which is inefficient, since
some words have higher predictive value than others. We tackle this inefficiency by introducing
our Attention Word Embedding (*AWE*) model. We also propose AWE-S, which incorporates subword information (code for which is in the fastText branch).

Details of this method and results can be found in our [COLING PAPER](https://arxiv.org/pdf/2006.00988.pdf).

## Information on this fork

I forked the following [repository](https://github.com/luffycodes/attention-word-embedding) in order to use it originally, and then I figured that many little things were broken or unclear.

I fixed part of the issue for `train_cbow.py` and `dump_w2v_k_fmt.py` so that someone could use it with latest torch and their own dataset.


### Training

Some parameters have defaults specific to the original developer, I recommand heavily to take care of `outputdir`, `outputmodelname` and `dataset_path`. I heavily recommend to change --evaluation_... if you are using a small dataset, as the default split (0.0001) might make your eval vary a lot (eg. `--validation_fraction 0.005`)

```sh
python train_cbow.py \
   --n_epochs 5 \
   --batch_size 128 \
   --w2m_type acbow \
   --word_emb_dim 200 \
   --dataset_path SingleFileEndingWith.txt OR directory containing .txt files \
   --context_size 10 \
   --mode cbow \
   --outputmodelname acbow.200.lemma.model \
   --outputdir ./models \
   --temp_path tempdir \
   --max_words 20000 # Your vocabulary size !

```

The best model against the validation set is saved with the suffix `_val.cbow_net`.

You can then turn it into a classical matrix with vocabulary -> vector using the `dump_w2v_k_fmt.py` script:

```sh

```