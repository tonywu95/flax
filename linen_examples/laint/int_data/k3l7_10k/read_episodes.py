import numpy as np
import collections
import random
filename = "episodes.npz"

all_examples = []
tokens = []
eos = '<eos>'
pad = '<pad>'
start = '<s>'

def add_eos(sent):
    return " ".join(sent.split() + [eos])

def add_start(sent):
    return " ".join([start] + sent.split())

def encode(sent, vocab):
    sent = sent.split()
    encoded_sent = [vocab[s] for s in sent]
    return encoded_sent

def encode_dataset(dataset):
    encoded = []
    for s in dataset:
        st, at, st_1, r, v = s
        st = encode(st, vocab)
        at = encode(at, vocab)
        st_1 = encode(st_1, vocab)
        s = (st, at, st_1, r, v)
        encoded.append(s)
    return encoded


def preprocess_dataset(episodes):
    tokens = []
    new_episodes = []
    for episode_index in range(len(episodes)):
        episode = episodes[episode_index]
        for s in episode:
            st, at, st_1, r, v = s
            st = add_start(add_eos(st))
            at = add_start(add_eos(at))
            st_1 = add_eos(st_1)
            s = (st, at, st_1, r, v)
            for sent in s[:3]:
                tokens += sent.split()
            new_episodes.append(s)
    return new_episodes, tokens


if __name__ == "__main__":
    train_frac = 0.95
    valid_frac = (1-train_frac)/2
    test_frac = (1-train_frac)/2


    with open(filename, 'rb') as f:
        batch_episodes = np.load(f, allow_pickle=True)["a"]

    total_episodes = len(batch_episodes)
    num_train = round(total_episodes * train_frac)
    num_valid = round(total_episodes * valid_frac)
    random.shuffle(batch_episodes)
    train_episodes = batch_episodes[:num_train]
    valid_episodes = batch_episodes[num_train:num_train+num_valid]
    test_episodes = batch_episodes[num_train+num_valid:]

    train_dataset, train_tokens = preprocess_dataset(train_episodes)
    valid_dataset, valid_tokens = preprocess_dataset(valid_episodes)
    test_dataset, test_tokens = preprocess_dataset(test_episodes)

    tokens = train_tokens + valid_tokens + test_tokens
    counts = collections.Counter(tokens)
    vocab_size = len(counts)
    sorted_counts = collections.OrderedDict(
        sorted(counts.items(), key=lambda x: x[1], reverse=True))
    vocab = collections.OrderedDict()
    vocab.update({pad: 0})
    vocab.update({v:idx+1 for v, idx in zip(sorted_counts.keys(), range(vocab_size))})
    for v,idx in vocab.items():
        print(idx, v, counts[v])

    train_dataset = encode_dataset(train_dataset)
    valid_dataset = encode_dataset(valid_dataset)
    test_dataset = encode_dataset(test_dataset)
    random.shuffle(train_dataset)

    np.save("vocab.npy", vocab)
    np.save("train.npy", train_dataset)
    np.save("valid.npy", valid_dataset)
    np.save("test.npy", test_dataset)









