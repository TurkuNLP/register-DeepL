
import sys


data = sys.stdin.readlines()

# have to add some sort of check to see what language the file is in
def format(data):
    test = []
    # check the first line for how many columns it has
    test.append(data[0].replace("\n", ""))
    test[0] = data[0].split("\t") # split the columns
    if len(test[0]) == 2:
        formatEN_FIN(data)
    else:
        formatFRE_SWE(data)


def formatEN_FIN(data):
    all = {} # an empty dictionary
    current_char_count = 0
    all_count = 0

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


        # count how many characters current text has
        current_char_count = len(data[i][1])
        


        # #simple line count instead
        # if data[i][0] in all:
        #     line_count = all.get(data[i][0])
        #     all[data[i][0]] = line_count + 1
        # else:
        #     all[data[i][0]] = 1

        # check whether the current label/key exist already
        if data[i][0] in all:
            all_count = all.get(data[i][0])
            all[data[i][0]] = all_count + current_char_count # update the dictionary
            current_char_count = 0
        else:
            all[data[i][0]] = current_char_count # add to the dictionary
            current_char_count = 0

    #print(all)
    sorted_dict = dict(sorted(all.items(), key=lambda item: item[1], reverse=True))
    print(sorted_dict)


def formatFRE_SWE(data):
    all = {} # an empty dictionary
    current_char_count = 0
    all_count = 0

    for i in range(len(data)):
        # if there is an empty line we ignore it
        if data[i] =="\n" or "":
            continue

        data[i] = data[i].replace("\n", "")
        data[i] = data[i].split("\t")

        #there should be four columns id, source, labels, text 
        # in swe_dev some line has 'SUS' at the end???
        try:
            assert len(data[i]) == 4
        except:
            if len(data[i]) == 5:
                #print(data[i], i)
                data[i].pop(len(data[i]) -1)
            elif len(data[i]) == 3:
                # line 151, 608, 775 in swe_dev has no text even in the original tsv file
                #print(data[i], i)
                continue
            else:
                # line 2 in swe_train ends up here in preprocessed smaller??
                continue
   
        # count how many characters current text has
        current_char_count = len(data[i][3])


        # # simple line count instead
        # if data[i][2] in all:
        #     line_count = all.get(data[i][2])
        #     all[data[i][2]] = line_count + 1
        # else:
        #     all[data[i][2]] = 1


        # check whether the current label/key exist already
        if data[i][2] in all:
            all_count = all.get(data[i][2])
            all[data[i][2]] = all_count + current_char_count # update the dictionary
            current_char_count = 0
        else:
            all[data[i][2]] = current_char_count # add to the dictionary
            current_char_count = 0

    #print(all)
    sorted_dict = dict(sorted(all.items(), key=lambda item: item[1], reverse=True))
    print(sorted_dict)


format(data)




 # there are so many different label combinationst so maybe we should look at them too?



# other than english there are a lot of different combinations of labels

# it seems that english has just the basic labels, sometimes many of them but no sublabels

# swedish and french has base labels first
# and then the corresponding sublabels e.g. NA IP (base) + OA DS (sub)

# finnish has sublabel baselabel sublabel baselabel style (?)
# sometimes baselabel sublabel baselabel
# there seem to be some typos for some labels, and for one atleast NA was mentioned twice
# IP was IG instead in a few places
