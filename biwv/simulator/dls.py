import random
import pandas as pd
from nltk.corpus import wordnet

from base import BaseSimulator

from river.feature_extraction.vectorize import VectorizerMixin
from river.metrics import Accuracy, CohenKappa

random.seed(0)

class IncSeedLexicon(BaseSimulator, VectorizerMixin):

    def __init__(
            self, 
            stream, 
            method, 
            training_lexicon, 
            test_lexicon,
            clf,
            change=False, 
            f=None, d=None, 
            on=None,
            strip_accents=True,
            lowercase=True,
            preprocessor=None,
            tokenizer=None,
            ngram_range=(1, 1),
        ):
        BaseSimulator.__init__(self, stream, method, f=f, d=d, change=change)

        VectorizerMixin.__init__(
            self, 
            on=on,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            ngram_range=ngram_range,
        )

        self.training_lexicon = training_lexicon
        self.test_lexicon = test_lexicon
        
        self.clf = clf
        self.acc = CohenKappa()

        self.oposites_test_words = {}
        self.oposites_test_words_values = {}

        if self.with_change:
            self.change_lexicon_words()
        
    
    def train(self):
        if self.with_change:
            for (b_idx, batch) in enumerate(self.stream):
                self.counter += len(batch)
                if self.counter > self.d:
                    #print("holaaaa")
                    self.train_with_change(batch)
                else:
                    self.train_without_change(batch)    
        else:
            
            for (b_idx, batch) in enumerate(self.stream):
                self.counter += len(batch)
                self.train_without_change(batch)

    def train_without_change(self, batch):
        self.method.learn_many(batch)
        for text in batch:
            tokens = self.process_text(text)
            for token in tokens:
                if token in self.training_lexicon and token in self.method.vocab:
                    label = self.training_lexicon[token]
                    self.train_classifier(token, label)
                elif token in self.test_lexicon and token in self.method.vocab:
                    label = self.test_lexicon[token]
                    #print(f'token {token} {label} {self.clf.predict_one(self.method.embedding2dict(token))}')
                    self._updateEvatulator(token, label)
            
    def train_classifier(self, token, label):
        #print(self.method.embedding2dict(token))
        #print(self.clf.learn_one(self.method.embedding2dict(token), label))
        self.clf.learn_one(self.method.embedding2dict(token), label)

    def _updateEvatulator(self, token, label):
        y_true = label
        y_pred = self.clf.predict_one(self.method.embedding2dict(token))
        self.acc.update(y_true=y_true, y_pred=y_pred)

    def train_with_change(self, batch):
        batch = self.preprocess_batch(batch)
        self.method.learn_many(batch)
        for text in batch:
            tokens = self.process_text(text)
            for token in tokens:
                if token in self.training_lexicon and token in self.method.vocab:
                    print("hola1")
                    if token in self.oposites_test_words:
                        label = self.oposites_test_words_values[token]
                        self.train_classifier(token, label)
                    else: 
                        label = self.training_lexicon[token]
                        self.train_classifier(token, label)
                elif token in self.oposites_test_words_values and token in self.method.vocab:
                    print("----------------------------------------------------------------------")
                    label = self.oposites_test_words_values[token]
                    self._updateEvatulator(token, label)
                elif token in self.test_lexicon and token in self.method.vocab:
                    print("hola 3")
                    label = self.test_lexicon[token]
                    self._updateEvatulator(token, label)

    def change_lexicon_words(self):
        if self.f is not None:
            num = int(self.f * len(self.test_lexicon))
            words = tuple(self.test_lexicon.data.keys())
            words = random.choices(words, k=num)
            # print(f'words = {words}')
            for word in words:
                antonyms = []
                for syn in wordnet.synsets(word):
                    for lm in syn.lemmas():
                        if lm.antonyms():
                            antonyms.append(lm.antonyms()[0].name())
                if len(antonyms) > 0:
                    antonym = antonyms[0]
                    self.oposites_test_words[word] = antonym
                    self.oposites_test_words_values[antonym] = not self.test_lexicon[word]
    
    def preprocess_batch(self, batch):
        new_batch = []
        for text in batch:
            split_text = text.split(" ")
            for i, token in enumerate(split_text):
                if token in self.oposites_test_words:
                    split_text[i] = self.oposites_test_words[token]
            new_text = ' '.join(split_text)
            new_batch.append(new_text)
        return new_batch


class LexiconDataset:

    def __init__(self, lexicon):
        self.data = lexicon

    @staticmethod
    def from_pandas_series(X, y):
        data = {}
        for x_value, y_value in zip(X, y):
            if int(y_value) == 1:
                data[x_value] = True
            else:
                data[x_value] = False
        return LexiconDataset(data)
    
    @staticmethod
    def from_txt(path):
        data = {}
        with open(path, encoding='utf-8') as reader:
            for line in reader:
                data = line.split("\t")
                if int(data[1]) == 1:
                    data[data[0]] = True
                else:
                    data[data[0]] == False
        return LexiconDataset(data)

    def get_words(self):
        return list(self.data.keys())

    def __getitem__(self, word):
        return self.data[word]

    def __contains__(self, word):
        return word in self.data.keys()
    
    def __str__(self):
        return list(self.data.keys()).__str__()
    
    def __len__(self):
        return len(self.data.keys())

    
