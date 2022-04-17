import heapq
from typing import Counter


class SpaceSavingAlgorithm:
    """
    Efficient `Counter`-like structure for approximating the top `m` elements of a stream, in O(m)
    space (https://www.cse.ust.hk/~raywong/comp5331/References/EfficientComputationOfFrequentAndTop-kElementsInDataStreams.pdf).

    Specifically, the resulting counter will contain the correct counts for the top k elements with
    k ≈ m.  The interface is the same as `collections.Counter`.
    """

    def __init__(self, m):
        self._m = m
        self._elements_seen = 0
        self._counts = Counter()  # contains the counts for all elements
        self._queue = []  # contains the estimated hits for the counted elements

    def _update_element(self, x):
        self._elements_seen += 1

        if x in self._counts:
            self._counts[x] += 1
        elif len(self._counts) < self._m:
            self._counts[x] = 1
            self._heappush(1, self._elements_seen, x)
        else:
            self._replace_least_element(x)

    def _replace_least_element(self, e):
        while True:
            count, tstamp, key = self._heappop()
            assert self._counts[key] >= count
            if self._counts[key] == count:
                del self._counts[key]
                count = 0
                break
            else:
                self._heappush(self._counts[key], tstamp, key)

        
        self._counts[e] = count + 1
        print
        self._heappush(count, self._elements_seen, e)

    def _heappush(self, count, tstamp, key):
        heapq.heappush(self._queue, (count, tstamp, key))

    def _heappop(self):
        return heapq.heappop(self._queue)

    def most_common(self, n=None):
        return self._counts.most_common(n)

    def elements(self):
        return self._counts.elements()

    def __len__(self):
        return len(self._counts)

    def __getitem__(self, key):
        return self._counts[key]

    def __iter__(self):
        return iter(self._counts)

    def __contains__(self, item):
        return item in self._counts

    def __reversed__(self):
        return reversed(self._counts)

    def items(self):
        return self._counts.items()

    def keys(self):
        return self._counts.keys()

    def values(self):
        return self._counts.values()
    
    def update(self, iter):
        for e in iter:
            self._update_element(e)


# def test_SpaceSavingAlgorithm():
#     #ssc = SpaceSavingAlgorithm(3)
#     #arr1 = [1, 5, 3, 4, 2, 7, 7, 1, 3, 1, 3, 1, 3, 1, 3]
#     # tweet =  "el perro ladra había un perro perro perro llamado juanito, perro habia donde el perro ladraba habia habia mucho hola habia".split(" ") 
#     # for word in tweet:
#     #     #print(word)
#     #     ssc.update_element(word)
#     #     print(ssc._counts)
#     # print(ssc._queue)
#     # print(ssc._counts)



#     ssc = SpaceSavingAlgorithm(4)
#     tweet = "el perro ladra había un perro perro perro llamado juanito, perro habia donde el perro ladraba habia habia mucho hola habia".split(" ")
#     ssc.update(tweet)
#     print(ssc.keys())
#     # arr1 = [1, 5, 3, 4, 2, 7, 7, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1]
#     # for x in arr1:
#     #     ssc.update_element(x)
#     # print(ssc.keys())
#     # print(ssc._counts)
#     # assert ssc.keys() == {3, 2}

#     ssc = SpaceSavingAlgorithm(2)
#     ssc.update([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2])
#     assert ssc.keys() == {3, 2}

#     ssc = SpaceSavingAlgorithm(1)
#     ssc.update([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2])
#     assert ssc.keys() == {2}

#     ssc = SpaceSavingAlgorithm(3)
#     ssc.update([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2])
#     assert ssc.keys() == {1, 2, 3}

#     ssc = SpaceSavingAlgorithm(2)
#     ssc.update([])
#     assert ssc.keys() == set()

# ssa = SpaceSavingAlgorithm(100)
# i = 1
# keys1 = None
# keys2 = None
# with open('C:/Users/gabri/Desktop/biwv/biwv/proccess_tweets.txt', encoding='utf-8') as tweets:
#     for tweet in tweets:
#         tweet = tweet.split(" ")
#         ssa.update(tweet)

# ssa = SpaceSavingAlgorithm(10)
# fn = 'C:/Users/gabri/Desktop/biwv/biwv/proccess_tweets.txt'
# bat_size = 256 
# buff_size = 2048
# stream = TweetStreamLoader(fn, bat_size, buff_size, shuffle=False)
# i = 0
# for batch in stream:
#     for tweet in batch:
#         tweet = tweet.split(" ")
#         #print(tweet)
#         ssa.update(tweet)
#         #print()
#         # print(ssa.keys())
#         print(ssa._queue)
#         print(i)
#         i += 1
# stream.close()
        

