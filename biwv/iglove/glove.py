import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .model import GloVeModel, GloVeDataSet

from iwcm import WordContextMatrix


class IGlove(WordContextMatrix):

    def __init__(
        self,
        embedding_size,  
        v_size, 
        c_size, 
        w_size,
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
        self._glove_dataset = DataLoader(GloVeDataSet(coocurrence_matrix), 1)
        print(self._glove_dataset)
        total_loss = 0
        for idx, batch in enumerate(self._glove_dataset):
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
            # print(self.model.embedding_for_tensor(torch.tensor([0])))
        
    def learn_many(self, X, y=None, **kwargs):
        super().learn_many(X, y=y, **kwargs)
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
        self._glove_dataset = DataLoader(GloVeDataSet(coocurrence_matrix), 1)
        #print(self._glove_dataset)
        total_loss = 0
        for idx, batch in enumerate(self._glove_dataset):
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
            print(self.model.embedding_for_tensor(torch.tensor([1])))
        
        
    
    def transform_one(self, x: dict):
        ...

    def get_embedding(self, word):
        ...




