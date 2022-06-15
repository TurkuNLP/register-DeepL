import sys

data = sys.stdin.readlines()
indeces_delete = [] # indeces marked for deletion in the data

for i in range(len(data)):
    data[i] = data[i].replace("\n", "")
    data[i] = data[i].split("\t") # split at the labels

    labels = data[i][0].split()
    labelstring = ""
    for label in labels:
        if (label != "" or label != " ") and labelstring == "":
            labelstring = label
        elif label == "" or label == " ":
            continue
        else:
            labelstring = labelstring + " " + label
    data[i][0] = labelstring
    if data[i][0] == "":
        indeces_delete.append(i)

for i in sorted(indeces_delete, reverse=True):
    data.pop(i)



# print to command line to save to files
final = []
for i in range(len(data)):
    dummy = data[i][0] + '\t' + data[i][1]
    final.append(dummy)

for i in range(len(final)):
    print(final[i])