import sys

data = sys.stdin.readlines()

indeces_delete = [] # indeces marked for deletion in the data
# split to lists
for i in range(len(data)):
    data[i] = data[i].replace("\n", "")
    data[i] = data[i].split("\t") # split at the labels
    if data[i][0] == "":
        #delete the mistake
        indeces_delete.append(i)
    elif data[i][0] == " ":
        indeces_delete.append(i)

# remove the faulty lines
for i in sorted(indeces_delete, reverse=True):
    data.pop(i)


# print to command line to save to files
final = []
for i in range(len(data)):
    dummy = data[i][0] + '\t' + data[i][1]
    final.append(dummy)

for i in range(len(final)):
    print(final[i])