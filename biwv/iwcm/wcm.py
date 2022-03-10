from base.iwv import IncrementalWordVector


class WordContextMatrix(IncrementalWordVector):

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

        print(self.vocab_size)
    
    def learn_many(X, y=None, **kwargs):
        ...
    
    # todo preguntar al pablo si esto es mala práctica.

    def transform_one(self, x: dict):
        ...

    def get_embedding(self, word):
        ...