dictP = {}
dictT = {}
dictS = {}
with open("../data/test_content.csv", "r", encoding="utf-8") as fid:
	for line in fid.readlines():
			line = line.strip()
			ids,sen = line.split(",")
			dictS[ids] =sen

with open("Result.csv", "r", encoding="utf-8") as fid:
	for line in fid.readlines():
		line = line.strip()
		ids, a, b, c, d = line.split(",")
		if ids not in dictP:
			dictP[ids] = [(a,b,c,d)]
		else:
			dictP[ids].append((a,b,c,d))

with open("../data/test_label.csv", "r", encoding="utf-8") as fid:
	for line in fid.readlines():
		line = line.strip()
		ids, a, a1, a2, b, b1, b2, c, d = line.split(",")
		if ids not in dictT:
			dictT[ids] = [(a,b,c,d)]
		else:
			dictT[ids].append((a,b,c,d))
with open("cmp.csv", "w", encoding="utf-8") as fid:
	for ids in dictP.keys():
		sen = dictS[ids]
		predict = dictP[ids]
		trueth  = dictT[ids]
		fid.write(sen+"\n")
		for i in predict:
			fid.write(",".join(list(i))+"\n")
		fid.write("//////"+"\n")
		for i in trueth:
			fid.write(",".join(list(i))+"\n")
		fid.write("\n")

S = 0
P = 0
G = 0
for ids in dictP.keys():
	predict = dictP[ids]
	trueth  = dictT[ids]

	P = P+len(predict)
	G = G+len(trueth)

	for i in predict:
		if i in trueth:
			S = S+1

accuracy = S/P
recall = S/G
print("accuracy: %f" %(accuracy))
print("recall: %f" %(recall))
print("f1: %f" %((2*accuracy*recall)/(accuracy+recall)))
