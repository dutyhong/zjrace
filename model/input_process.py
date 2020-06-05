import pickle
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from model_config import Config
import jieba
import jieba.posseg as pseg
import pandas as pd
concat_train_data = pd.read_csv("../data/concat_train_content.csv", header=None)
concat_train_data.columns = ["id", "reviews"]
jieba.load_userdict("../src_data/usrdict.txt")

def postag(x):
    postags = pseg.cut(x)
    words = []
    poses = []
    for word, pos in postags:
        words.append(word)
        poses.append(pos)
    return " ".join(poses)
concat_train_data["poses"] = concat_train_data["reviews"].apply(postag)
pos_tags = set()
for row in list(concat_train_data["poses"]):
    tags = row.split(" ")
    for tag in tags:
        pos_tags.add(tag)
pos_vocabulary = {}
for i, tag in enumerate(pos_tags):
    pos_vocabulary[tag] = i+1
pickle.dump(pos_vocabulary, open("../data/pos_vocabulary_dict.pkl", "wb"))
# w2v_vectors = KeyedVectors.load_word2vec_format('/Users/duty/downloads/zhijiang_race/word2vec_c2v_model_50.txt', binary=False)
# c2v_vectors = KeyedVectors.load('/Users/duty/downloads/zhijiang_race/word2vec_c2v_model_50.kv', mmap="r")
c2v_vectors = KeyedVectors.load_word2vec_format("/Users/duty/downloads/zhijiang_race/glove_c2v_model_50.txt", binary=False)
w2v_vectors = KeyedVectors.load_word2vec_format("/Users/duty/downloads/zhijiang_race/glove_w2v_model_50.txt", binary=False)
# vocab_size = w2v_vectors.vocab_size
char_vocabs = c2v_vectors.vocab
vocab_size = len(char_vocabs)
embed_dim = 50
word_vocabs = w2v_vectors.vocab
word_vocab_size = len(word_vocabs)
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


char_vocabulary = dict()
for i, vocab in enumerate(char_vocabs):
    char_vocabulary[str(vocab)] = i+1
word_vocabulary = dict()
for i, vocab in enumerate(word_vocabs):
    word_vocabulary[str(vocab)] = i+1
pickle.dump(char_vocabulary, open("../data/char_vocabulary_dict.pkl","wb"))
pickle.dump(word_vocabulary, open("../data/word_vocabulary_dict.pkl", "wb"))
char_embedding = build_embedding(c2v_vectors, char_vocabulary, embedding_dim=50)
word_embedding = build_embedding(w2v_vectors, word_vocabulary, embedding_dim=50)
np.save("../data/char_embedding.npy", char_embedding)
np.save("../data/word_embedding.npy", word_embedding)

###统计bigram的个数


###tag都一样的 不需要在做了
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