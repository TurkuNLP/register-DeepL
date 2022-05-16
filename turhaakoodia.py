import sys

data = sys.stdin.readlines()


# the fre and swe files are a bit different so I have to make a new script for those.

# this works for the finnish and english data
def formatEN_FIN(data):
    for i in range(len(data)):
        data[i] = data[i].replace("\n", "")
        data[i] = data[i].split("\t") # split at the labels
        #check here that there are always 2 columns (label and text)

        # at least with en_train3 
        # we end up here because the sentence continues from the end ofen_train2
        # => must zcat them together to get them working right
        try:
            assert len(data[i]) == 2
        except AssertionError:
            # if there is only one column (text most likely), then put the text into the second column and the label to the other
            data[i].append(data[i][0])
            data[i][1] = data[i][1].replace("\t", "") # try to remove the tab and fail at least with the testi.txt, might not count as a true \t
            data[i][0] = "other"
            #print(data[i], i)
        # if the first column is empty we put it to other
        if data[i][0] == "":
            data[i][0] = "other"
            #print(data[i], i)

    labels =[one[0] for one in data]
    texts= [one[1] for one in data]

    #the labels are saved to their own file here, should maybe add ids?
    # the file name should correspond to the current thing so possbly save on the command line?
    with open('./preprocessed_texts/labelit.txt', 'w') as f:
        f.write('\n'.join(labels))
    
    print(texts)
    return texts


def formatFRE_SWE(data):
    print()
    for i in range(len(data)):
        #have to skip the first couple columns?
        data[i] = data[i].replace("\n", "")
        data[i] = data[i].split("\t")


    labels =[one[2] for one in data]
    texts= [one[3] for one in data]
    print(texts)
    return texts


formatEN_FIN(data)
formatFRE_SWE(data)

    # should there be an id somewehere? and the same thing with the labels?
    # or do I just make sure they are in the same order (something could go wrong)