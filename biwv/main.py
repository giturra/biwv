from iwcm.vocab import Vocab, WordRep

wr = WordRep("hola", 4)
wr.add_context("estas")
wr.add_context("?")
print(wr.contexts)
wr.add_context("bien")
