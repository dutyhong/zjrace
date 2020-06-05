# ner_file = open("result_ner","r")
# char_sentence_list = list()
# tag_sentence_list = list()
# char_list = list()
# tag_list = list()
# for line in ner_file.readlines():
#     if(len(line)>2):
#         columns = line.split(" ")
#         if(len(columns)!=2):
#             continue
#         else:
#             if(columns[0]!="-1"):
#                 char_list.append(columns[0])
#                 tag_list.append(columns[1].strip("\n"))
#         # x_test_word_list.append(columns[0])
#         # y_test_word_list.append(columns[1])
#     else:
#         char_sentence_list.append(char_list)
#         tag_sentence_list.append(tag_list)
#         char_list = list()
#         tag_list = list()
# write_file= open("cla_input", "r")
# for i, (sentence, tags) in enumerate(zip(char_sentence_list, tag_sentence_list)):
#     ##定义一个Alist Olist 分别放每句话所有的A和O
#     a_tag_list = list()
#     o_tag_list = list()
#     a_tag_index_list = list()
#     o_tag_index_list = list()
#     flag = 0
#     a_strs = ""
#     for j,  (word, tag) in enumerate(zip(sentence, tags)):
#         if("A" in tag):
#             a_tag_list.append(word)
#             a_tag_index_list.append(j)
#         if("A_end"==tag):
#             a_strs = a_strs + "".join(a_tag_list)+","
#             # a_tag_list.append("end")
#             # a_tag_index_list.append(-1)
#             flag = 1
#
#         if("O" in tag):
#             o_tag_list.append(word)
#             o_tag_index_list.append(j)
#         if("O_end"==tag):
#             o_tag_list
#             # o_tag_list.append("end")
#             # o_tag_index_list.append(-1)
#     sentence_str = "".join(sentence)
#     o_strs = list()
#     o_str = ""
#     a_strs = list()
#     a_str = ""
#     if(len(a_tag_list)==0):
#         for i, tag in zip(o_tag_index_list, o_tag_list):
#             if(i!=-1):
#                 o_str = o_str+sentence[i]
#             else:
#                 o_strs.append(o_str)
#                 o_str = ""
#                 continue
#     else:
#         for i, tag in zip(a_tag_index_list, a_tag_list):
#             if (i != -1):
#                 a_str= a_str + sentence[i]
#             else:
#                 a_strs.append(a_str)
#                 a_str = ""
#                 continue
#         for i, tag in zip(o_tag_index_list, o_tag_list):
#             if (i != -1):
#                 o_str = o_str + sentence[i]
#             else:
#                 o_strs.append(o_str)
#                 o_str = ""
#                 continue


from pyltp import Postagger
from pyltp import Parser
import os
import jieba
import pandas as pd
def parserS(tag):
  idL,idO = [],[]
  tag_len = len(tag)
  for j in range(len(tag)):
    if "beg" in tag[j]:
      idL = []
      idL.append(j)
    if "mid" in tag[j]:
      idL.append(j)
      # if j == tag_len-1:
      #   idO.append(idL)
      # elif "mid" not in tag[j+1] and "end" not in tag[j+1]:
      #   idO.append(idL)
    if "end" in tag[j]:
      if idL != []:
        idL.append(j)
        idO.append(idL)
      idL = []
    if tag[j] in ["A","O"]:
      idO.append([j])
      idL = []
    if tag[j] == "U":
      idL = []
  return idO

def parserQ(idO, sen, tag):
  idQ = 0
  segl = []
  for l in idO:
    head = l[0]
    tail = l[-1]
    segl.extend(jieba.lcut(sen[idQ:head], cut_all=False))
    segl.extend([sen[head:tail+1]])
    idQ = tail+1
  segl.extend(jieba.lcut(sen[idQ:len(sen)], cut_all=False))
  labl,idQ = [],0
  for l in segl:
    head = idQ
    tail = idQ+len(l)
    if list(range(head, tail)) in idO:
      labl.append(tag[head][0])
    else:
      labl.append("U")
    idQ = idQ+len(l)
  return segl, labl


def parserE(dic, segl, labl, ids, rely_id, relation):
    rely_list = []
    for l in range(len(segl) - 1):
        if labl[l] == "A" and labl[l + 1] == "O":
            rely_list.append((segl[l], segl[l + 1]))
            labl[l], labl[l + 1] = "U", "U"

    for l in range(len(segl) - 2):
        if labl[l] == "A" and labl[l + 1] == "U" and labl[l + 2] == "O" and segl[l + 1] not in [",", ".", "!", "?"]:
            rely_list.append((segl[l], segl[l + 2]))
            labl[l], labl[l + 2] = "U", "U"

    for l in range(len(segl) - 3):
        if labl[l] == "A" and labl[l + 1] == "U" and labl[l + 2] == "U" and labl[l + 3] == "O" and segl[l + 1] not in [
            ",", ".", "!", "?"] and segl[l + 2] not in [",", ".", "!", "?"]:
            rely_list.append((segl[l], segl[l + 3]))
            labl[l], labl[l + 3] = "U", "U"

    for l in range(len(segl) - 1):
        if labl[l] == "O" and labl[l + 1] == "A":
            rely_list.append((segl[l + 1], segl[l]))
            labl[l], labl[l + 1] = "U", "U"


    for l in range(len(segl)):
        cur_id, rel_id = l, rely_id[l] - 1
        if rel_id < 0:
            continue
        if (labl[cur_id], labl[rel_id]) == ("A", "O"):
            rely_list.append((segl[cur_id], segl[rel_id]))
            labl[cur_id], labl[rel_id] = "U", "U"
        if (labl[cur_id], labl[rel_id]) == ("O", "A"):
            rely_list.append((segl[rel_id], segl[cur_id]))
            labl[cur_id], labl[rel_id] = "U", "U"

    for l in range(len(segl)):
        if labl[l] == "A":
          rely_list.append((segl[l], "_"))
        if labl[l] == "O":
            rely_list.append(("_", segl[l]))

    for l in rely_list:
        if [l[0], l[1], "_", "_"] not in dic[ids]["opi"]:
            dic[ids]["opi"].append([l[0], l[1], "_", "_"])
    return dic


def _par_compile_(idsl, senl, tagl):
    # len1 =
    dic = {}
    LTP_DATA_DIR = "/Users/duty/downloads/ltp_data_v3.4.0"
    postagger = Postagger()
    parser = Parser()
    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
    postagger.load(pos_model_path)
    parser.load(par_model_path)
    fid = open("ner.pos", "w", encoding="utf-8")
    for i in range(len(idsl)):
        dic[idsl[i]] = {}
        dic[idsl[i]]["sen"] = senl[i]
        dic[idsl[i]]["opi"] = []
        idO = parserS(tagl[i])
        segl, labl = parserQ(idO, senl[i], tagl[i])
        postags = postagger.postag(segl)
        arcs = parser.parse(segl, postags)
        rely_id = [arc.head for arc in arcs]
        relation = [arc.relation for arc in arcs]
        fid.write("\t".join(segl) + "\n")
        fid.write("\t".join([str(w) for w in rely_id]) + "\n")
        fid.write("\t".join(relation) + "\n")
        fid.write("///////" + "\n")
        dic = parserE(dic, segl, labl, idsl[i], rely_id, relation)
    cla_data = []
    src_data = []
    for i in dic.keys():
        sens = dic[i]["sen"]
        labs = dic[i]["opi"]
        if labs == []:
            src_data.append((i, sens, "_", "_", "_", "_"))
            # cla_data.append((i, ("_" + "_").strip("_"), sens))
            cla_data.append((i, ("_" + ",_"), sens))
            continue
        for l in labs:
            src_data.append((i, sens, l[0], l[1], l[2], l[3]))
            # cla_data.append((i,(l[0] + l[1]).strip("_"), sens))
            cla_data.append((i, (str(l[0])+"," + str(l[1])), sens))
    return cla_data

if __name__ == "__main__":
    ##idsl为句子的idlist， senl为句子list， tagl为二维list tag
    ner_file = open("result_ner", "r")
    char_sentence_list = list()
    tag_sentence_list = list()
    char_list = list()
    tag_list = list()
    for line in ner_file.readlines():
        if (len(line) > 2):
            columns = line.split("\t")
            if (len(columns) != 2):
                continue
            else:
                if (columns[0] != "-1"):
                    char_list.append(columns[0])
                    tag_list.append(columns[1].strip("\n"))
            # x_test_word_list.append(columns[0])
            # y_test_word_list.append(columns[1])
        else:
            char_sentence_list.append(char_list)
            tag_sentence_list.append(tag_list)
            char_list = list()
            tag_list = list()
    idsl = []
    senl = []
    tagl = []
    for i, (sentence, tags) in enumerate(zip(char_sentence_list, tag_sentence_list)):
        idsl.append(i)
        senl.append("".join(sentence))
        tag_list = []
        for ch, tag in zip(sentence, tags):
            tag_list.append(tag)
        tagl.append(tag_list)

    print(len(idsl), len(senl), len(tagl))
    print([(senl[k], tagl[k]) for k in range(len(senl)) if len(senl[k])!=len(tagl[k])])

    result = _par_compile_(idsl, senl, tagl)
    # result_file = open("../data/result.csv", "w")
    result_file = open("../data/concat_ner_result.csv", "w")
    # result_file.write("index,aspect_content,sen_content\n")
    result_file.write("id,Aspect_terms,Opinion_terms,sen_content\n")
    # for (i, aspect_content, sen_content) in result:
    #     aspect_content = aspect_content.replace(',', '，')
    #     sen_content = sen_content.replace(',', '，')
    #     result_file.write(str(i)+","+aspect_content+","+sen_content+"\n")
    for (i, aspect_content, sen_content) in result:
        result_file.write(str(i+1)+","+aspect_content+","+sen_content+"\n")
    result_file.close()
    result_data = pd.read_csv("../data/concat_result.csv")
    ##读取对应的原始content数据
    # test_data = pd.read_csv("../data/test_content.csv", header=None)
    # test_data.to_csv("../data/test_content_index.csv", index=True, header=["id", "Reviews"])
    test_data_index = pd.read_csv("../data/test_content_index.csv")
    join_data = pd.DataFrame.merge(result_data, test_data_index, on="index")
    # test_label = pd.read_csv("../data/test_label.csv", header=None)
    # test_label.to_csv("../data/test_label_index.csv",
    #                   header=["id","aspect_terms","ab","ae","opinion_terms","ab","ae","Categories","Polarities"])
    test_label = pd.read_csv("../data/test_label_index.csv")
    test_label[["id", "aspect_terms","opinion_terms", "Categories", "Polarities"]].to_csv("../data/test_label_index.csv",
                header=True)
    ##将ner获得的结果与原始数据进行join
    tmp = pd.DataFrame.merge(result_data, test_data_index, on="index")
    final = pd.DataFrame.merge(tmp, test_label, on="id")
    print("ddd")

