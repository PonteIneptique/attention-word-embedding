import os
import pickle
import tqdm
import torch

import argparse

def _save_embeddings_to_word2vec(cbow_net, vocabulary, output_path):
    cbow_net = torch.load(cbow_net)
    encoder = cbow_net.module.encoder
    embeddings = encoder.key_table
    embeddings = embeddings.weight.data.cpu().numpy()

    # Load (inverse) vocabulary to match ids to words
    vocabulary = pickle.load(open(vocabulary, 'rb'))[0]
    inverse_vocab = {vocabulary[w]: w for w in vocabulary}

    # Open file and write values in word2vec format
    output_path = os.path.join(output_path)
    f = open(output_path, 'w')
    print(embeddings.shape[0] - 1, embeddings.shape[1], file=f)
    for i in tqdm.tqdm(range(1, embeddings.shape[0])):  # skip the padding token
        cur_word = inverse_vocab[i]
        f.write(" ".join([cur_word] + [str(embeddings[i, j]) for j in range(embeddings.shape[1])]) + "\n")

    f.close()

    return output_path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", required=True, help="Path to the vocab file")
    parser.add_argument("--cbow_net", required=True, help="Path to the cbow_net file")
    parser.add_argument("--output", required=True, help="Path to the output file")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    _save_embeddings_to_word2vec(args.cbow_net, args.vocab, args.output)
    print("Done")