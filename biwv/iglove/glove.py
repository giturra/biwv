from base import IncrementalWordVector
from iwcm import WordContextMatrix


class IGlove(IncrementalWordVector):

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
        is_ppmi=True
    ):
        super().__init__(
            v_size,
            c_size,
            w_size,
            on=on,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            ngram_range=ngram_range,
        )

        self.wcm = WordContextMatrix(v_size, c_size, w_size)

    def learn_one(self, x, **kwargs):
        ...
    
    def learn_many(self, X, y=None, **kwargs):
        ...
    
    def transform_one(self, x: dict):
        ...

    def get_embedding(self, word):
        ...