# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 03:37:34 2021

@author: arami
"""

###TASK 1

import re
import numpy as np

# load up all of the 19997 documents in the corpus
corpus = sc.textFile ("s3://chrisjermainebucket/comp330_A6/20_news_same_line.txt")

# each entry in validLines will be a line from the text file
validLines = corpus.filter(lambda x : 'id' in x)

# now we transform it into a bunch of (docID, text) pairs
keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:]))

# now we split the text in each (docID, text) pair into a list of words
# after this, we have a data set with (docID, ["word1", "word2", "word3", ...])
# we have a bit of fancy regular expression stuff here to make sure that we do not
# die on some of the documents
regex = re.compile('[^a-zA-Z]')
keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

# now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))

# now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.reduceByKey (lambda a, b: a + b)

# and get the top 20,000 words in a local array
# each entry is a ("word1", count) pair
topWords = allCounts.top (20000, lambda x : x[1])

# and we'll create a RDD that has a bunch of (word, dictNum) pairs
# start by creating an RDD that has the number 0 thru 20000
# 20000 is the number of words that will be in our dictionary
twentyK = sc.parallelize(range(20000))

# now, we transform (0), (1), (2), ... to ("mostcommonword", 1) ("nextmostcommon", 2), ...
# the number will be the spot in the dictionary used to tell us where the word is located
# HINT: make use of topWords in the lambda that you supply
#dictionary = twentyK.map (??????????????????????)
dictionary = twentyK.map (lambda x: (topWords[x][0], x))

# finally, print out some of the dictionary, just for debugging
#dictionary.top (10)

word_id = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
word_id_pos = word_id.join(dictionary)  #(word, (id, pos))
id_pos = word_id_pos.values().groupByKey()

def convert_to_array(l):
    results = np.zeros(20000)
    for i in l:
        results[i] += 1
    return results

id_freq = id_pos.map(lambda x:(x[0], convert_to_array(x[1]) ))

a = id_freq.lookup("20_newsgroups/comp.graphics/37261")[0]
print(a[a.nonzero()])
b = id_freq.lookup("20_newsgroups/talk.politics.mideast/75944")[0]
print(b[b.nonzero()])
c = id_freq.lookup("20_newsgroups/sci.med/58763")[0]
print(c[c.nonzero()])


#########Task 2
tf = id_freq.map(lambda x: (x[0], x[1]/x[1].sum()))
numDocs = corpus.count()
c1 = id_freq.map(lambda x: (x[0], np.clip(x[1], 0, 1)))
c2 = c1.map(lambda x: (1, x[1]) )
numDocswithWord = c2.reduceByKey(lambda a, b: a+b)

idf = np.log(numDocs/numDocswithWord.lookup(1)[0])

tf_idf = tf.map(lambda x: (x[0], x[1]*idf))

a2 = tf_idf.lookup("20_newsgroups/comp.graphics/37261")[0]
print(a2[a2.nonzero()])
b2 = tf_idf.lookup("20_newsgroups/talk.politics.mideast/75944")[0]
print(b2[b2.nonzero()])
c2 = tf_idf.lookup("20_newsgroups/sci.med/58763")[0]
print(c2[c2.nonzero()])


#####Task 3
def bestkey(d):
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]


def predictLabel(k, s):
    s_ = regex.sub(' ', s).lower().split()
    Words = sc.parallelize(s_)
    words_ = Words.map(lambda x: (x, 1))
    words_pos = words_.join(dictionary)
    ids_pos = words_pos.values().groupByKey()
    id_freq2 = ids_pos.map(lambda x:(x[0], convert_to_array(x[1]) ))
    
    
    tf2 = id_freq2.map(lambda x: (1, x[1]/sum(x[1])))
    tf_idf2 = tf2.map(lambda x: (x[0], x[1]*idf))
    tfidf2 = tf_idf2.lookup(1)[0]
    
    
    l2Norm = tf_idf.map(lambda x:(x[0], np.linalg.norm(x[1] - tfidf2)))
    knn = l2Norm.takeOrdered(k, lambda x:x[1])
    knn2 = list(map(lambda x: (x[0][14:], x[1]), knn))
    knn3 = list(map(lambda x: (x[0][:x[0].index('/')], x[1]), knn2))
    knn4 = list(map(lambda x: x[0], knn3))
    
    best = {}
    for doc in knn4:
        try:
            best[doc] +=1
        except:
            best[doc] = 1
    return bestkey(best)
