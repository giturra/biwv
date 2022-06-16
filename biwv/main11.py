import scipy
import numpy as np
from isgns import ISGNS
from streamdataloader import TweetStreamLoader
from iwcm import WordContextMatrix

from web.datasets.similarity import fetch_MEN
from web.evaluate import evaluate_similarity

fn = 'C:/Users/gabri/Desktop/biwv/biwv/proccess_tweets.txt'
bat_size = 256 
buff_size = 2048
emp_ldr = TweetStreamLoader(fn, bat_size, buff_size, shuffle=False)

k = 0
men = fetch_MEN()
isn = ISGNS(device='cuda:0')
#wcm = WordContextMatrix(1000000, 50000, 1)
with open('similarity.txt', "w") as writer:
    for i, batch in enumerate(emp_ldr):
        isn.learn_many(batch)
        if k % 256 == 0:
            A = []
            B = []
            y = []

            for words, weight in zip(men.X, men.y):
                if words[0] in isn.vocab and words[1] in isn.vocab:
                    A.append(isn.get_embedding(words[0]))
                    B.append(isn.get_embedding(words[1]))
                    y.append(weight)
            A = np.vstack(A)
            B = np.vstack(B)
            scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
            result = scipy.stats.spearmanr(scores, y).correlation
            writer.write(f'resultado similarity = {result}\n')
        k += len(batch)  
    emp_ldr.fin.close()

