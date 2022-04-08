from base import BaseSimulator
from river.feature_extraction.vectorize import VectorizerMixin



class IncSeedLexicon(BaseSimulator, VectorizerMixin):

    def __init__(
            self, 
            stream, 
            method, 
            f, d, 
            training_lexicon, 
            test_lexicon, 
            clf,
            normalize=True,
            on=None,
            strip_accents=True,
            lowercase=True,
            preprocessor=None,
            tokenizer=None,
            ngram_range=(1, 1),
        ):
        super().__init__(stream, method, f, d)

        super().__init__(
            normalize=normalize,
            on=on,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            ngram_range=ngram_range,
        )

        self.training_lexicon = training_lexicon
        self.test_lexico = test_lexicon
        self.clf = clf
    
    
    def train(self):
        for (b_idx, batch) in enumerate(self.stream):
            self.method.learn_many(batch)
            for text in batch:
                tokens = self.process_text(text)
                for token in tokens:
                    if token in self.training_lexicon:
                        ...
    
    def _train_classifier(self, token, label):
        self.clf.learn_one(token, label)

    def _updateEvatulator(self, token, label):
    
    def train_with_change(self):
        ...