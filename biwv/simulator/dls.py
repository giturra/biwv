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
                        self._train_classifier()
    
    def _train_classifier(self, token, label):
        self.clf.learn_one({'token': self.method.get_embedding(token), 'label':label})

    def _updateEvatulator(self, token, label):
        ...

    def train_with_change(self):
        ...


class LexiconDataset:

    def __init__(self, path):
        self.path = path
        self.data = {}

        with open(path, encoding='utf-8') as reader:
            for line in reader:
                data = line.split("\t")
                print(data)
                self.data[data[0]] = data[1]
        
    def get_words(self):
        return list(self.data.keys())

    def __getitem__(self, word):
        return self.data[word]

    def __contains__(self, word):
        return word in self.data.keys()
    
    def __str__(self):
        return list(self.data.keys()).__str__()

    
