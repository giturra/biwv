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

    def learn_many(self, X, y=None, **kwargs):
        super().learn_many(X)
        
    
    def transform_one(self, x: dict):
        ...

    def get_embedding(self, word):
        ...