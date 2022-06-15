import sys

data = sys.stdin.readlines()

labels = {} # empty dictionary where to put the different label combinations and the count
count=0
for i in range(len(data)):
    data[i] = data[i].replace("\n", "")
    data[i] = data[i].split("\t") # split at the labels

    if data[i][0] in labels:
            count = labels.get(data[i][0])
            labels[data[i][0]] = count + 1 # update the dictionary
            count = 0
    else:
        labels[data[i][0]] = 1 # add to the dictionary
        count = 0

# sort so that biggest labels are at the start (descending order)
sorted_dict = dict(sorted(labels.items(), key=lambda item: item[1], reverse=True))

print(len(data))
print(sorted_dict) # either print the amount of lines per label

# or make a list with the percentages
percents={}
for key in sorted_dict:
        percents[key] = round(sorted_dict[key] / len(data) * 100, 2)
print(percents)