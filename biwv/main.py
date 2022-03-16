from isgns.vocab import Vocab
from isgns.sgns import ISGNS   

from river.datasets import SMSSpam

# v = Vocab(3)
# print(v.add("hello"))
# print(v.add("are"))
# print(v.add("you"))
# print(v.add("?"))
# print(v.add("you"))
# print(v.add("hello"))
# print(v.table)
# print(v.counter)
# print(v.total_counts)

dataset = SMSSpam()

isg = ISGNS(5, 5, 5)
for xi, yi in dataset:
    isg.learn_one(xi['body'])