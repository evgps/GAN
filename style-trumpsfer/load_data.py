import numpy as np
from gensim.models.wrappers import FastText
from torch.utils.data import Dataset

class Numerator(object):
    def __init__(self, data, fasttext_location='wiki.simple'):        
        self.data = data            
        self.vocab_size = 0
        self.vocab = dict()
        self.index = dict()
        self.special_tokens = ["PAD", "UNK", "SOS", "EOS"]
        self.embedding_size = 300 + len(self.special_tokens)
        self.embedding_model = FastText.load_fasttext_format(fasttext_location)
        self.embeddings = []
        self.init_special_tokens()
        self.build_vocab()

    def init_special_tokens(self):
        for token in self.special_tokens:
            emb = np.zeros(self.embedding_size)
            emb[self.vocab_size] = 1
            self.embeddings.append(emb)
            self.vocab[token] = self.vocab_size
            self.vocab_size += 1
                                           
    def add_embedding(self, word):
        self.embeddings.append()
        
    def embed(self, word):
        try:
            return np.concatenate((np.zeros(len(self.special_tokens)), self.embedding_model[word]))
        except KeyError:
            unk = np.zeros(self.embedding_size)
            unk[1] = 1
            return unk
        
    def unembed(self, word):
        distances = [(np.linalg.norm(word - embedding), id) for id, embedding in enumerate(self.embeddings)]
        min_distance, closest_embedding_id = min(distances, key = lambda x: x[0])
        return self.index[closest_embedding_id]
    
    def add_word(self, word):
        self.embeddings.append(self.embed(word))
        self.vocab[word] = self.vocab_size
        self.vocab_size += 1

    def build_vocab(self):
        for sentence in self.data:
            for word in str(sentence).strip().split(" "):
                if word not in self.vocab:
                    self.add_word(word)
                        

        self.index = dict([reversed(i) for i in self.vocab.items()])
        self.embeddings = np.array(self.embeddings)
        
    def numerate(self):
        X = []
        
        for sentence in self.data:
            sentence_idx = []
            for word in str(sentence).strip().split(" "):
                try:
                    sentence_idx.append(self.vocab[word])
                    
                except KeyError:
                    pass
                
            X.append(np.array(sentence_idx))
            
        return np.array(X)



class StyleDataset(Dataset):
    def __init__(self, sentences, labels, embeddings, sentence_size=None):
        assert len(sentences) == len(labels)
        self.sentences = sentences
        self.labels = labels
        self.embeddings = embeddings
        self.sentence_size = sentence_size if sentence_size else max([len(sentence) for sentence in self.sentences])
        
    def __getitem__(self, idx):
        return self._get_embedding(self._resize(self.sentences[idx])), self.labels[idx]

    def __len__(self):
        return len(self.labels)
        
    def _get_embedding(self, sentence):
        try:
            return(np.vstack([self.embeddings[word] for word in sentence]))
        except IndexError:
            print(sentence)


    def _resize(self, sentence):
        if len(sentence) >= self.sentence_size:
            return sentence[:self.sentence_size]

        return np.concatenate((sentence, np.zeros(self.sentence_size - len(sentence)))).astype(int)


def load_data(data_paths, fasttext_location = None):
    """
    data_paths -- list of strings to corpora (which are just lines of text)
    """
    datasets = []

    for path in data_paths:
        with open(path, 'r', errors='replace') as f:
            datasets.append(f.readlines())

    lengths = [len(dataset) for dataset in datasets]
    if fasttext_location:
        n = Numerator([sentence for dataset in datasets for sentence in dataset], fasttext_location=fasttext_location)
    else:
        n = Numerator([sentence for dataset in datasets for sentence in dataset])
    X = n.numerate()
    # y contains one-hot vectors
    y = np.zeros((sum(lengths), len(lengths)))
    y[np.arange(sum(lengths)), np.repeat(np.arange(len(lengths)), lengths)] = 1

    return X, y, n
