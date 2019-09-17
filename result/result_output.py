import pandas as pd
cla_result = pd.read_csv("../data/predict_file.csv")
ner_result = pd.read_csv("../data/ner_result.csv")
# join_data = pd.DataFrame.merge(ner_result, cla_result, on="id")
# # final_result = join_data[["id", "Aspect_terms", "Opinion_terms", "Categories", "Polarities"]]
# # final_result = final_result.drop_duplicates()
final_result = pd.concat([ner_result, cla_result.drop(["id"], axis=1)], axis=1)
result_csv = final_result[["id", "Aspect_terms", "Opinion_terms", "Categories", "Polarities"]]
##对最终的result进行处理去掉逗号，对于A和O都是空的就全都是空
result_csv["Aspect_terms"] = result_csv["Aspect_terms"].apply(lambda x: x.replace("，", ""))
result_csv["Opinion_terms"] = result_csv["Opinion_terms"].apply(lambda x: x.replace("，", ""))
def func(x, y, z):
    if x=="_" and y=="_":
        return "_"
    else:
        return z
result_csv["Categories"] = result_csv.apply(lambda x: func(x.Aspect_terms, x.Opinion_terms, x.Categories), axis=1)
# result_csv[(result_csv["Aspect_terms"]=="_") & (result_csv["Opinion_terms"]=="_")]["Categories"]="_"
# result_csv["Polarities"].apply(lambda x: "_" if result_csv["Aspect_terms"]=="_" and result_csv["Opinion_terms"]=="_" else x)
# result_csv[(result_csv["Aspect_terms"]=="_") & (result_csv["Opinion_terms"]=="_")]["Polarities"] = "_"
result_csv["Polarities"] = result_csv.apply(lambda x: func(x.Aspect_terms, x.Opinion_terms, x.Polarities), axis=1)
pd.DataFrame.to_csv(result_csv, "../data/final_result.csv",index=False)
###校验最终结果 所有的id必须有，不能出现双逗号
# result_csv["id"] = result_csv["id"].drop([2,3,4,5,6])
ids = set(result_csv["id"])
for i in range(3911):
    a = i+1
    if a not in ids:
        print("Error id %s   not in file "%a)
        break

print("ddd")