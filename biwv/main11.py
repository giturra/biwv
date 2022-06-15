from isgns import ISGNS
from streamdataloader import TweetStreamLoader
isn = ISGNS(device='cuda:0')

from web.datasets.similarity import fetch_MEN
from web.evaluate import evaluate_similarity

fn = 'C:/Users/gabri/Desktop/biwv/biwv/proccess_tweets.txt'
bat_size = 256 
buff_size = 2048
emp_ldr = TweetStreamLoader(fn, bat_size, buff_size, shuffle=False)
i = 0
men = fetch_MEN()
for i, batch in enumerate(emp_ldr):
    isn.learn_many(batch)
    if i % 256 == 0:
        embeddings = isn.vocab2dict()
        print(evaluate_similarity(embeddings, men.X, men.y))
    i += 1      
emp_ldr.fin.close()
