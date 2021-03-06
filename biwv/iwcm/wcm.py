import numpy as np

from base.iwv import IncrementalWordVector
from .vocab import Context, Vocab, WordRep


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
            normalize=normalize, 
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            ngram_range=ngram_range,
        )

        self.vocab = Vocab(self.vocab_size)
        self.contexts = Context(self.context_size)
        self.d = 0

        self.is_ppmi = is_ppmi

        self.vocab.add(WordRep('unk', self.context_size))
    
    def learn_one(self, x, **kwargs):
        tokens = self.process_text(x)
        for w in tokens:
            i = tokens.index(w)
            self.d += 1
            self.vocab.add(WordRep(w, self.context_size))
            contexts = _get_contexts(i, self.window_size, tokens)
            focus_word = self.vocab[w]
            for c in contexts:
                if c not in self.contexts:
                    self.contexts.add(c)
                if c not in self.contexts and len(self.contexts) == self.context_size and focus_word.word == 'unk':
                    focus_word.add_context('unk')
                elif c not in self.contexts:
                    focus_word.add_context('unk')
                elif c in self.contexts:
                    focus_word.add_context(c)   
            # print(f"{focus_word.word} {self.get_embedding(focus_word.word)}")

    def learn_many(self, X, y=None, **kwargs):
        for x in X:
            tokens = self.process_text(x)
            for w in tokens:
                i = tokens.index(w)
                self.d += 1
                self.vocab.add(WordRep(w, self.context_size))
                contexts = get_contexts(i, self.window_size, tokens)
                focus_word = self.vocab[w]
                for c in contexts:
                    if c not in self.contexts:
                        self.contexts.add(c)
                    if c not in self.contexts and len(self.contexts) == self.context_size and focus_word.word == 'unk':
                        focus_word.add_context('unk')
                    elif c not in self.contexts:
                        focus_word.add_context('unk')
                    elif c in self.contexts:
                        focus_word.add_context(c)
                if focus_word.word != 'unk':
                    print(f'{focus_word.word} {self.get_embedding(focus_word.word)}')

                
        
    
    # todo preguntar al pablo si esto es mala pr??ctica.

    def transform_one(self, x: dict):
        ...

    def get_wr(self, word):
        if word in self.vocab:
            ...

    def get_embedding(self, word):
        if word in self.vocab:
            word_rep = self.vocab[word]
            embedding = np.zeros(self.context_size, dtype=float)
            contexts = word_rep.contexts.items()
            #print(contexts)
            if self.is_ppmi:
                for context, coocurence in contexts:
                    #print(context, coocurence)
                    ind_c = self.contexts[context]
                    #print(ind_c)
                    #print(f'es un word rep {word_rep}')
                    #print(context, coocurence)
                    pmi = np.log2(
                        (coocurence * self.d) / (word_rep.counter * self.vocab[context].counter) 
                    )
                    #print(f'ppmi = {pmi}')
                    embedding[ind_c] = max(0, pmi)
            else:
                for context, coocurence in contexts:
                    ind_c = self.contexts[context]
                    embedding[ind_c] = coocurence 
            return embedding
        False

def get_contexts(ind_word, w_size, tokens):
    # to do: agregar try para check que es posible obtener los elementos de los tokens
    slice_start = ind_word - w_size if (ind_word - w_size >= 0) else 0
    slice_end = len(tokens) if (ind_word + w_size + 1 >= len(tokens)) else ind_word + w_size + 1
    first_part = tokens[slice_start: ind_word]
    last_part = tokens[ind_word + 1: slice_end]
    contexts = tuple(first_part + last_part)
    return contexts