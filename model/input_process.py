import pickle
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
# w2v_vectors = KeyedVectors.load_word2vec_format('/Users/duty/downloads/zhijiang_race/word2vec_c2v_model_50.txt', binary=False)
w2v_vectors = KeyedVectors.load('/Users/duty/downloads/zhijiang_race/word2vec_c2v_model_50.kv', mmap="r")
# vocab_size = w2v_vectors.vocab_size
vocabs = w2v_vectors.vocab
vocab_size = len(vocabs)
embed_dim = 50
##把训练出来的词向量写到文件，待模型的embedding层调用
def build_embedding(model,vocab_dict, embedding_dim=50):
    # model = Word2Vec(corpus, size=embedding_dim, min_count=1, window=5, sg=1, iter=10)
    weights = model.wv.syn0
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    emb = np.zeros(shape=(len(vocab_dict) + 2, embedding_dim), dtype='float32')

    count = 0
    for w, i in vocab_dict.items():
        if w not in d:
            count += 1
            emb[i, :] = np.random.uniform(-0.1, 0.1, embedding_dim)
        else:
            emb[i, :] = weights[d[w], :]
    print('embedding out of vocabulary：', count)
    return emb

for vocab in vocabs:
    print(str(vocab))
print(vocab_size)
char_vocabulary = dict()
for i, vocab in enumerate(vocabs):
    char_vocabulary[str(vocab)] = i+1
pickle.dump(char_vocabulary, open("../data/char_vocabulary_dict.pkl","wb"))
char_embedding = build_embedding(w2v_vectors, char_vocabulary, embedding_dim=50)
np.save("../data/char_embedding.npy", char_embedding)
tag_vocabulary = dict()
##读取训练数据取出所有tag
file = open("../data/train.ner.v2", "r")
tags = set()
for line in file.readlines():
    columns = line.split("\t")
    if(len(columns)!=2):
        continue
    tags.add(columns[1].strip("\n"))
for i, tag in enumerate(tags):
    tag_vocabulary[tag] = i
pickle.dump(tag_vocabulary, open("../data/tag_vocabulary_dict.pkl", "wb"))
print("ddd")