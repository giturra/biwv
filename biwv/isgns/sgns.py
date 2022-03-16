import math
import random
from re import sub
from typing import Counter

from base.iwv import IncrementalWordVector
from .vocab import Vocab

class ISGNS(IncrementalWordVector):

    def __init__(
        self, 
        vocab_size, 
        vector_size, 
        window_size,
        subsampling_threshold=1e-3, 
        normalize=True,
        on=None,
        strip_accents=True,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        ngram_range=(1, 1),
    ):
        super().__init__(
            vocab_size,
            vector_size,
            window_size,
            on=on,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            ngram_range=ngram_range,
        )

        self.vocab_size = vocab_size
        self.vector_size = vector_size
        self.window_size = window_size

        self.vocab = Vocab(self.vocab_size)

        self.subsampling_threshold = subsampling_threshold
    
    def learn_one(self, x, y=None, **kwargs):
        tokens = self.process_text(x)
        for token in tokens:
            self.vocab.add(token)
        print(tokens)
        subsample_tokens = self._subsampling(tokens)
        print(subsample_tokens)

    def learn_many(self, X, y=None, **kwargs):
        ...
    
    def get_embedding(self, word):
        ...
    
    def transform_one(self, x: dict):
        ...
    
    def _subsampling(self, tokens):
        subsample_tokens = []
        for token in tokens:
            print(token, self._normalized_freq(token), self.vocab.total_counts)
            prob = 1 - math.sqrt(self._normalized_freq(token))
            if random.uniform(0, 1) > prob:
                subsample_tokens.append(token)
        return subsample_tokens

    def _normalized_freq(self, word):
        return (self.subsampling_threshold *  self.vocab.total_counts) / self.vocab.counter[word] 
