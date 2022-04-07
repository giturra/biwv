import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from collections import Counter

from .model import GloVeModel, GloVeDataSet

from iwcm import WordContextMatrix


class IGlove(WordContextMatrix):

    def __init__(
        self,
        embedding_size,  
        v_size, 
        c_size, 
        w_size,
        min_occurrance=1,
        learning_rate=0.05, 
        normalize=True,
        on=None,
        strip_accents=True,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        ngram_range=(1, 1),
        is_ppmi=False,
        verbose=False,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

        self._glove_dataset = None
        self.model = GloVeModel(embedding_size, v_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # print(self.model.parameters()[0])
        self.verbose = verbose
        self.device = device

        self.min_occurrance = min_occurrance

    def learn_one(self, x, **kwargs):
        ...
    
    def learn_many(self, X, y=None, **kwargs):
        word_counts = Counter()
        for x in X:
            tokens = self.process_text(x)
            if len(self.vocab) <= self.vocab_size: 
                for w in tokens:
                    self.vocab.add(w)
            if len(self.contexts) <= self.context_size:
                for w in tokens:
                    self.contexts.add(w)
            idxs = self.tokens2idxs(tokens)
            # word_counts.update(idxs)
            for left_context, word, right_context in context_windows(idxs, self.window_size, self.window_size):
                self.d += 1
                if word == -1:   
                    continue
                else:
                    for i, context_word in enumerate(left_context[::-1]):
                        if context_word in self.contexts:
                            word_counts[(word, context_word)] += 1
                            self.coocurence_matrix[word, context_word] += 1 / (i + 1)
                    for i, context_word in enumerate(right_context):
                        if context_word in self.contexts:
                            word_counts[(word, context_word)] += 1
                            self.coocurence_matrix[word, context_word] += 1 / (i + 1)
        cooc_matrix = [
            (idx[0], idx[1], count) for (idx, count) in word_counts.items()
        ]
        self._glove_dataset = DataLoader(GloVeDataSet(cooc_matrix), len(cooc_matrix))
        total_loss = 0
        for idx, batch in enumerate(self._glove_dataset):
            #print(batch)
            self.optimizer.zero_grad()
            i_s, j_s, counts = batch
            #print(len(i_s))
            i_s = i_s.to(self.device)
            j_s = j_s.to(self.device)
            counts = counts.to(self.device)
            loss = self.model._loss(i_s, j_s, counts)

            total_loss += loss.item()
            # if idx % loop_interval == 0:
            #     avg_loss = total_loss / loop_interval
            #     print("epoch: {}, current step: {}, average loss: {}".format(
            #         epoch, idx, avg_loss))
            #     total_loss = 0

            loss.backward()
            self.optimizer.step()
            print(f'largo tensor {self.model.embedding_for_tensor(torch.tensor([0]))}')
        
        


    def transform_one(self, x: dict):
        ...

    # def get_embedding(self, word):
    #     ...




def context_windows(region, left_size, right_size):
    """generate left_context, word, right_context tuples for each region

    Args:
        region (str): a sentence
        left_size (int): left windows size
        right_size (int): right windows size
    """

    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = window(region, start_index, i - 1)
        right_context = window(region, i + 1, end_index)
        yield (left_context, word, right_context)


def window(region, start_index, end_index):
    """Returns the list of words starting from `start_index`, going to `end_index`
    taken from region. If `start_index` is a negative number, or if `end_index`
    is greater than the index of the last word in region, this function will pad
    its return value with `NULL_WORD`.

    Args:
        region (str): the sentence for extracting the token base on the context
        start_index (int): index for start step of window
        end_index (int): index for the end step of window
    """
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):
                             min(end_index, last_index) + 1]
    return selected_tokens
