import torch
import numpy as np

from base import IncrementalWordVector
from vocab import Vocab
from unigram_table import UnigramTable
from rand import RandomNum
from sg import SkipGram


class ISGNS(IncrementalWordVector):

    def __init__(self, 
    vocab_size=1e6, 
    vector_size=100, 
    window_size=5,
    unigram_table_size=1e8,
    neg_sample_num=5,
    alpha=0.75,
    subsampling_threshold=1e-3, 
    normalize=True, 
    on=None, 
    strip_accents=True, 
    lowercase=True, 
    preprocessor=None, 
    tokenizer=None, 
    ngram_range=False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        super().__init__(
            vocab_size, 
            vector_size, 
            window_size, 
            normalize=normalize, 
            on=on, 
            strip_accents=strip_accents, 
            lowercase=lowercase, 
            preprocessor=preprocessor, 
            tokenizer=tokenizer, 
            ngram_range=ngram_range
        )

        self.vector_size = int(vector_size)

        self.vocab_size = int(2 * vocab_size)
        self.vocab = Vocab(int(self.vocab_size * 2))

        self.unigram_table_size = unigram_table_size
        self.unigram_table = UnigramTable(self.unigram_table_size, device=device)

        self.device = device

        self.counts = torch.zeros(int(2 * self.vocab_size))
        self.counts.to(self.device)

        self.total_count = 0

        self.neg_sample_num = neg_sample_num
        
        self.alpha = alpha
        self.window_size = window_size
        self.subsampling_threshold = subsampling_threshold
        
        self.randomizer = RandomNum(1234)

        # net in pytorch
        self.model = SkipGram(self.vocab_size, self.vector_size)
        if self.device == 'cuda':
            self.model.cuda()
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.5, momentum=0.9)
        self.optimizer = torch.optim.Adagrad(self.model.parameters())
        self.criterion = torch.nn.BCEWithLogitsLoss()
    
    def learn_many(self, X, y=None):
        X_input = None
        y_true = None
        for phrase in X:
            tokens = self.process_text(phrase)
            n = len(tokens)
            neg_samples = torch.zeros(int(self.neg_sample_num))
            neg_samples.to(self.device)
            for target, w in enumerate(tokens):
                self.update_unigram_table(w)
                target_index = self.vocab[w]
                if target_index == -1:
                    continue
                random_window_size = self.randomizer.uniform(1, self.window_size + 1)
                for offset in range(int(-random_window_size), int(random_window_size)):
                    if offset == 0 or (target + offset) < 0:
                        continue
                    if (target + offset) == n:
                        break
                    context_index = self.vocab[tokens[target + offset]]
                    if context_index == -1:
                        continue
                    if 0 < self.counts[context_index] and np.sqrt(
                        (self.subsampling_threshold * self.total_count) / self.counts[context_index]
                    ) < self.randomizer.uniform(0, 1):
                        continue
                    for k in range(0, self.neg_sample_num):
                        neg_samples[k] = int(self.unigram_table.sample(self.randomizer))

                    if X_input is None and y_true is None:
                        X_input, y_true = _create_input(target_index, context_index, neg_samples)    
                        #print(X_input.size())
                        X_input.to(self.device)
                        y_true.to(self.device)
                    else:
                        x_input, labels = _create_input(target_index, context_index, neg_samples)
                        X_input = torch.vstack((X_input, x_input))
                        #print(X_input.size())
                        X_input.to(self.device)
                        y_true = torch.vstack((y_true, labels))
                        y_true.to(self.device)

        y_pred = self.model(X_input)
        y_pred.to(self.device)
                
        self.model.zero_grad()

        loss = self.criterion(y_true.float(), y_pred.float())
        loss.backward()
                
        
        self.optimizer.step()

    def update_unigram_table(self, word: str):
        word_index = self.vocab.add(word)
        self.total_count += 1
        if word_index != -1:
            self.counts[word_index] += 1
            F = np.power(self.counts[word_index], self.alpha) - np.power(self.counts[word_index] - 1, self.alpha)
            self.unigram_table.update(word_index, F, self.randomizer)
    

    def get_embedding(self, word):
        index = self.vocab[word]
        u  = self.model.embedding_u.weight[index]
        v  = self.model.embedding_v.weight[index]
        return ((u + v) / 2).cpu().detach().numpy()


def _create_input(target_index, context_index, neg_samples):
    input = [[int(target_index), int(context_index)]]
    labels = [1]
    for neg_sample in neg_samples:
        input.append([target_index, int(neg_sample)])
        labels.append(0)
    return torch.LongTensor([input]), torch.LongTensor([labels])