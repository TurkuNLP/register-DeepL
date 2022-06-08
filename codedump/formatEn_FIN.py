import sys

data = sys.stdin.readlines()
indeces_delete = [] # indeces marked for deletion in the data

for i in range(len(data)):
    data[i] = data[i].replace("\n", "")
    data[i] = data[i].split("\t") # split at the labels

    #check here that there are always 2 columns (label and text)
    try:
        assert len(data[i]) == 2
    except AssertionError:
        #print(data[i])
        if data[i] == "":
            indeces_delete.append(i)
            #with en data we end up here
        elif len(data[i]) == 1:
            indeces_delete.append(i)
            # if there is only one column (text most likely), then put the text into the second column and the label to the other
            data[i].append(data[i][0])
            data[i][1] = data[i][1].replace("\t", "") # try to remove the tab and fail at least with the testi.txt, might not count as a true \t
            data[i][0] = "other"
            #print(data[i], i)

        # this is for the finnish data, there is somehow 3 columns and the second is empty
        elif len(data[1]) == 3:
            #move the the third column to the second
            data[i][1] = data[i][2]
            # remove the third column
            data[i].pop(len(data[i]) -1)
        else:
            indeces_delete.append(i)

    # if the first column is empty we put it to other or delete (for english data ONLY)
    if data[i][0] == "":
        data[i][0] = "other"
        indeces_delete.append(i)
        #print(data[i], i)

# remove the faulty lines
for i in sorted(indeces_delete, reverse=True):
    data.pop(i)

final=[]
for i in range(len(data)):
        dummy = data[i][0] + '\t' + data[i][1]
        final.append(dummy)
   
for i in range(len(final)):
    print(final[i])