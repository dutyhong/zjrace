import pandas as pd
concat_train_data = pd.read_csv("../data/concat_train_content.csv", header=None)

concat_train_data.columns = ["id", "reviews"]
def vary_data(concat_train_data: pd.DataFrame, validation_cnt=5):
    row_cnt = len(list(concat_train_data["id"]))
    validation_sets = []
    # for i, (id, reviews) in enumerate(zip(list(concat_train_data["id"]), list(concat_train_data["reviews"]))):
    #     validation_sets[i] = (id, reviews)
    # validation_cnt = 5
    one_validation_set_cnt = int(row_cnt/validation_cnt)
    validation_indices = []
    all_indices = []
    all_row = []
    for i in range(row_cnt):
        all_row.append(i)
    for i in range(validation_cnt-1):
        one_indices = []
        for j in range(one_validation_set_cnt):
            one_indices.append(j+i*one_validation_set_cnt)
        all_indices.append(one_indices)
    last_indices = []
    for i in range(row_cnt-(validation_cnt-1)*one_validation_set_cnt):
        last_indices.append(i+(validation_cnt-1)*one_validation_set_cnt)
    all_indices.append(last_indices)


    for i in range(len(all_indices)):
        remove_indices = all_indices[i]
        validation_index = []
        for i in range(row_cnt):
            if all_row[i] not in remove_indices:
                validation_index.append(all_row[i])
        validation_indices.append(validation_index)
    # for j in range(len(validation_indices)):
    #     validation_data = []
    #     remove_indices = all_indices[j]
    #     for i, id_reviews in enumerate(validation_sets):
    #         if i not in remove_indices:

    return validation_indices
print("ddd")




