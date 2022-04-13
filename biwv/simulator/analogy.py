import random
import statistics
from nltk.corpus import wordnet

from base import BaseSimulator
from river.feature_extraction.vectorize import VectorizerMixin

from web import evaluate_analogy
from web.embedding import Embedding

class AnalogySimulator(BaseSimulator, VectorizerMixin):
    
    def __init__(
            self, 
            stream, 
            method,
            analogy_dataset, 
            change=False, 
            f=None, d=2048, 
            on=None,
            strip_accents=True,
            lowercase=True,
            preprocessor=None,
            tokenizer=None,
            ngram_range=(1, 1),
        ):
        BaseSimulator.__init__(self, 
            stream, 
            method, 
            f, d,
            change=change
        )

        VectorizerMixin.__init__(
            self, 
            on=on,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            ngram_range=ngram_range,
        )

        self.analogy_dataset = analogy_dataset

        self.cand2antonyms = []
        self.word2antonym = {}

        if self.with_change:
            self.change_analogy_words()

        self.accuracies = []

    
    def train(self):
        if self.with_change:
            for (b_idx, batch) in enumerate(self.stream):
                if self.counter > self.d:
                    self.train_with_change(batch)
                else:
                    self.train_without_change(batch)
        else:
            for (b_idx, batch) in enumerate(self.stream):
                self.train_without_change(batch)
        return statistics.mean(self.accuracies)    
    
    def train_without_change(self, batch):
        self.counter += len(batch)
        self.method.learn_many(batch)
        if self.counter % self.d == 0:
            emb_dict = self.method.vocab2dict()
            self.accuracies.append(evaluate_analogy(
                Embedding.from_dict(emb_dict), 
                self.analogy_dataset.X, 
                self.analogy_dataset.y
            ))
    
    def train_with_change(self, batch):
        batch = self.preprocess_batch(batch)
        if self.counter % self.d == 0:
            emb_dict = self.method.vocab2dict()
            self.accuracies.append(evaluate_analogy(
                Embedding.from_dict(emb_dict), 
                self.analogy_dataset.X, 
                self.analogy_dataset.y
            ))

    def preprocess_batch(self, batch):
        new_batch = []
        for text in batch:
            # quizas buscar formas más inteligentes de hacer el split
            split_text = text.split(" ")
            for i, token in enumerate(split_text):
                if token in self.word2antonym:
                    split_text[i] = self.word2antonym[token]
            new_text = ' '.join(split_text)
            # print(text)
            # print(new_text)
            new_batch.append(new_text)
        return new_batch

    def change_analogy_words(self):
        if self.f is not None:
            num = int(self.f * len(self.analogy_dataset.X))
            self.load_words()
            words = random.choices(self.cand2antonyms, k=num)
            # print(f'words = {words}')
            for word in words:
                antonyms = []
                for syn in wordnet.synsets(word):
                    for lm in syn.lemmas():
                        if lm.antonyms():
                            antonyms.append(lm.antonyms()[0].name())
                if len(antonyms) > 0:
                    antonym = antonyms[0]
                    self.word2antonym[word] = antonym

    def load_words(self):
        for words in self.analogy_dataset.X:
            word = words[0]
            if word not in self.cand2antonyms:
                self.cand2antonyms.append(word)
        
    def _train(self, batch):
        self.method.learn_many(batch)
