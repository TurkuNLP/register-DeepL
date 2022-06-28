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

print("amount of examples:", len(data)) # amount of lines/examples as a whole
print("amount of labels:", len(sorted_dict)) # amount of labels

hybrid=0
single=0
for key in sorted_dict:
        length =len(key.split())
        if length > 1:
                hybrid = hybrid + sorted_dict[key]
        else:
                single = single + sorted_dict[key]
print("hybrid labelled examples:", hybrid, round(hybrid / len(data), 2), "%") # amount of hybrid labels
print("single labelled examples:", single, round(single / len(data), 2), "%") # amount of single labels


print(sorted_dict) # either print the amount of lines per label

# or make a list with the percentages
percents={}
for key in sorted_dict:
        percents[key] = round(sorted_dict[key] / len(data) * 100, 2)
print(percents)