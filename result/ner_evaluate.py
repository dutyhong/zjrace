import pickle

char_dict = pickle.load(open("../data/char_vocabulary_dict.pkl", "rb"))
for key, value in char_dict.items():
    if "婚" in key:
        print("dddddd")
test_true = [x.strip('\n').split('\t') for x in open('../data/concat_test.ner.v2', 'r', encoding='utf-8').readlines()]
test_predict = [x.strip('\n').split('\t') for x in open('../model/result_ner4', 'r', encoding='utf-8').readlines()]
test_true = [x[1] for x in test_true if x!=['']]
test_predict = [x[1] for x in test_predict if x!=['']]

total = len(test_true)
test_true_A = len([x for x in test_true if 'A' in x])
test_true_O = len([x for x in test_true if 'O' in x])
test_predict_A = len([x for x in test_predict if 'A' in x])
test_predict_O = len([x for x in test_predict if 'O' in x])
print(total, test_true_A, test_true_O, test_predict_A, test_predict_O)

index_same = len([k for k in range(total) if test_true[k]==test_predict[k]])
acc = index_same/total #都是total 量级一样
rec = index_same/total
f1 = index_same/total
print('total')
print(index_same, acc, rec, f1)

index_same = len([k for k in range(total) if test_true[k]==test_predict[k] and 'A' in test_true[k]])
acc = index_same/test_predict_A #都是total 量级一样
rec = index_same/test_true_A
f1 = 2*acc*rec/(acc+rec)
print('A')
print(index_same, acc, rec, f1)

index_same = len([k for k in range(total) if test_true[k]==test_predict[k] and 'O' in test_true[k]])
acc = index_same/test_predict_O #都是total 量级一样
rec = index_same/test_true_O
f1 = 2*acc*rec/(acc+rec)
print('O')
print(index_same, acc, rec, f1)