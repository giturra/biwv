from collections import Counter, defaultdict
from tabnanny import verbose
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class GloVeModel(nn.Module):
    """Implement GloVe model with Pytorch
    """

    def __init__(self, embedding_size, vocab_size, min_occurrance=1, x_max=100, alpha=3 / 4):
        super(GloVeModel, self).__init__()

        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.min_occurrance = min_occurrance
        self.x_max = x_max

        self._focal_embeddings = nn.Embedding(
            vocab_size, embedding_size).type(torch.float64)
        self._context_embeddings = nn.Embedding(
            vocab_size, embedding_size).type(torch.float64)
        self._focal_biases = nn.Embedding(vocab_size, 1).type(torch.float64)
        self._context_biases = nn.Embedding(vocab_size, 1).type(torch.float64)
        self._glove_dataset = None

        for params in self.parameters():
            init.uniform_(params, a=-1, b=1)



    def embedding_for_tensor(self, tokens):
        if not torch.is_tensor(tokens):
            raise ValueError("the tokens must be pytorch tensor object")

        return self._focal_embeddings(tokens) + self._context_embeddings(tokens)

    def _loss(self, focal_input, context_input, coocurrence_count):
        x_max, alpha = self.x_max, self.alpha

        focal_embed = self._focal_embeddings(focal_input)
        context_embed = self._context_embeddings(context_input)
        focal_bias = self._focal_biases(focal_input)
        context_bias = self._context_biases(context_input)
        weight_factor = torch.pow(coocurrence_count / x_max, alpha)
        weight_factor[weight_factor > 1] = 1
        embedding_products = torch.sum(focal_embed * context_embed, dim=1)
        log_cooccurrences = torch.log(coocurrence_count)
        distance_expr = (embedding_products + focal_bias +
                         context_bias - log_cooccurrences) ** 2
        single_losses = weight_factor * distance_expr
        mean_loss = torch.mean(single_losses)
        return mean_loss


class GloVeDataSet(Dataset):

    def __init__(self, coocurrence_matrix):
        self._coocurrence_matrix = coocurrence_matrix

    def __getitem__(self, index):
        return self._coocurrence_matrix[index]

    def __len__(self):
        return len(self._coocurrence_matrix)
