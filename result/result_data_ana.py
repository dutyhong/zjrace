import pandas as pd
# result_data = pd.read_csv("/Users/duty/downloads/zhijiang_race/复赛训练集2019-09-01/TRAIN/Train_makeup_labels.csv")
# id_data = list(result_data["id"])
# total_length = len(id_data)
# aspect_null_cnt = 0
# for aspect_terms, opinion_terms in zip(list(result_data["AspectTerms"]), list(result_data["OpinionTerms"])):
#     if aspect_terms=="_" and opinion_terms!="_":
#         aspect_null_cnt = aspect_null_cnt + 1
# print(aspect_null_cnt/total_length)

##分析结果数据中有多少在训练样本中出现却没有被提取出来的
train_laptop_label_data = pd.read_csv("/Users/duty/downloads/zhijiang_race/复赛训练集2019-09-01/TRAIN/Train_laptop_labels.csv")
result_data = pd.read_csv("/Users/duty/downloads/zhijiang_race/Result_0926.csv")
result_data.columns = ["id", "aspect_terms", "opinion_terms", "categories", "polarities"]
aspect_terms_set = set(train_laptop_label_data["AspectTerms"])
opinion_terms_set = set(train_laptop_label_data["OpinionTerms"])
test_laptop_reviews_data = pd.read_csv("/Users/duty/downloads/复赛测试集2019-09-10/Test_laptop_reviews.csv")
join_data = pd.merge(result_data, test_laptop_reviews_data, on="id")
aspect_term_dict = {}
opinion_term_dict = {}
sentence_dict = {}
for index, row in join_data.iterrows():
    id = row.id
    aspect_term = row.aspect_terms
    if id not in aspect_term_dict.keys():
        tmp_set = set()
        tmp_set.add(aspect_term)
        aspect_term_dict[id] = tmp_set
    else:
        tmp_set = aspect_term_dict[id]
        tmp_set.add(aspect_term)
        aspect_term_dict[id] = tmp_set
aspect_file = open("../result/aspect_file", "w")
for aspect_term in aspect_terms_set:
    for id, reviews in zip(list(test_laptop_reviews_data["id"]), list(test_laptop_reviews_data["Reviews"])):
        if aspect_term in reviews and aspect_term not in aspect_term_dict[id]:
            aspect_file.write(str(id)+","+reviews+","+ aspect_term+"\n")

print("ddd")