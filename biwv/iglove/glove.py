from base import IncrementalWordVector
from iwcm import WordContextMatrix


class IGlove(WordContextMatrix):

    def __init__(
        self, 
        v_size, 
        c_size, 
        w_size, 
        normalize=True,
        on=None,
        strip_accents=True,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        ngram_range=(1, 1),
        is_ppmi=False
    ):  
        super().__init__( 
            v_size, 
            c_size, 
            w_size, 
            normalize=normalize,
            on=on,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            ngram_range=ngram_range,
            is_ppmi=False
        )

    def learn_one(self, x, **kwargs):
        super().learn_one(x, **kwargs)
        word2idx = list(self.vocab.word2idx.items())
        coocurrence_matrix = []
        for word, idx in word2idx:
            wr = self.vocab[word]
            contexts = wr.contexts.items()
            for context, count in contexts:
                coocurrence_matrix.append((
                    self.vocab.word2idx[word],
                    self.vocab.word2idx[context],
                    count 
                ))
        print(coocurrence_matrix)
    def learn_many(self, X, y=None, **kwargs):
        super().learn_many(X)
        word2idx = list(self.vocab.word2idx.items())
        coocurrence_matrix = []
        for word, idx in word2idx:
            wr = self.vocab[word]
            contexts = wr.contexts.items()
            for context, count in contexts:
                coocurrence_matrix.append((
                    self.vocab.word2idx[word],
                    self.vocab.word2idx[context],
                    count 
                ))
        
        
    
    def transform_one(self, x: dict):
        ...

    def get_embedding(self, word):
        ...