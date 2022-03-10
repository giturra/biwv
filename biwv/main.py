from iwcm.vocab import Vocab, WordRep

wr = WordRep("hola", 3)
print(wr.size)
wr.add_context("estas")
print(wr.size)
wr.add_context("?")
print(wr.size)
print(wr.contexts)
wr.add_context("estas")
print(wr.size)
print(wr.contexts)
wr.add_context("mucho")
print(wr.contexts)