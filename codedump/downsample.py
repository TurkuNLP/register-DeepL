import sys
import random


# we just split the data to those sets after translation
# the english lines with empty labels are changed to have other as the label

# first we shuffle the data to make sure they are not in order (texts?), and finally we shuffle again so that the labels are not in order
# the idea is to get to below 25 million for all of the texts so we can translate to four different languages

def main(data):
    random.Random(1234).shuffle(data)
    #use argv to choose what downsampling strategy to choose
    if len(sys.argv) == 1:
        no_downsample(data)
    elif len(sys.argv) == 3:
        downsampleByAll(data)
    else:
        label_dict, sampled = format(data)
        final = downsampleByPercent(sampled)
        for i in range(len(final)):
            print(final[i])
  

def downsampleByAll(data):
    # here we downsample the biggest label classes
    cap = int(sys.argv[1])
    sampled = downsampleByLabel(data, cap)
    
    final = downsampleByPercent(sampled)

    for i in range(len(final)):
        print(final[i])
    #print(len(final), len(sampled), len(data))


# here we do the overall downsampling by x percent
def downsampleByPercent(sampled):
    if (len(sys.argv) == 3):
        percent = float(sys.argv[2]) # the amount by which we will downsample defined in the bash script
    else:
        percent = float(sys.argv[1])
    percent = percent / 100
    all_dict = {} # dictionary that includes all the texts as values (list) and their labels as keys
    final = [] # the final list of labels and their texts

    # downsample EVERY category 
    for i in range(len(sampled)):
        # check whether the current label/key exist already
        if sampled[i][0] in all_dict:
            currentlist = all_dict.get(sampled[i][0])
            currentlist.append(sampled[i][1])
            all_dict[sampled[i][0]] = currentlist # update the dictionary
        else:
            all_dict[sampled[i][0]] = [sampled[i][1]] # put the first sample to a list so it is possible to append later

    key_list = list(all_dict.keys())
    for i in range(len(key_list)): # key_list should be same length as all_dict
        textlist = all_dict.get(key_list[i]) # get the texts by looking at the key
        # count the number to keep from the texts
        num = round(len(textlist) * (1 - percent))
        shortened = textlist[:num] # here I am just getting the percent of lines left (not counting characters here at all)
        #print(key_list[i], percent, num, len(textlist), len(shortened))

        # now we change the format back to tsv
        for j in range(len(shortened)):
            dummy = key_list[i] + '\t' + shortened[j] # a dummy text
            final.append(dummy)
        
    # shuffle the data because it is in order now
    random.Random(1234).shuffle(final)

    return final


# this is for the rest of the files which do not need specific downsampling
def no_downsample(data):
    label_dict, newData = format(data)
    final = []

    for i in range(len(newData)):
        dummy = newData[i][0] + '\t' + newData[i][1]
        final.append(dummy)
   
    for i in range(len(final)):
        print(final[i])


def get_biggest_labels(data, cap):
    label_dict, newData = format(data)
    # take first (biggest) labels
    biggest_labels = []
    for i in range(cap):
        # get the biggest key
        biggest_labels.append(list(label_dict.items())[i][0])
        #print(list(label_dict.items())[i][1])
    
    max = list(label_dict.items())[cap][1]
    #print(max)

    return biggest_labels, max, newData


def downsampleByLabel(data, cap):
     # new list for downsampled data
    sampled = []
    # counter for the biggest labels
    big_dict = {}
    current_char_count = 0
    # get the maximum number of characters for a single label (max = fifth label)
    biggest_labels, max, newData = get_biggest_labels(data, cap)
    for i in range(len(newData)):
        # if in the biggest class
        if newData[i][0] in biggest_labels:
            current_char_count = len(data[i][1])
            # check whether the current label/key exist already
            if data[i][0] in big_dict:
                big_count = big_dict.get(data[i][0])
                max_check = big_count + current_char_count
                # if class is full skip
                if max_check >= max:
                    continue
                else:
                    big_dict[data[i][0]] = big_count + current_char_count # update the dictionary
                    current_char_count = 0
                    sampled.append(newData[i])
            
            else:
                big_dict[data[i][0]] = current_char_count # add to the dictionary
                current_char_count = 0
        # if not a big class just append to the new list
        else:
            sampled.append(newData[i])

    # count all of the chars in the downsampled texts
    all_together = 0
    for i in range(len(sampled)):
        all_together = all_together + len(sampled[i][1])

    #print(big_dict)
    #print(all_together)
    
    return sampled


def format(data):
    test = []
    # check the first line for how many columns it has
    # this could be unreliable because what if the first line is broken?
    test.append(data[0].replace("\n", ""))
    test[0] = data[0].split("\t") # split the columns
    if len(test[0]) == 2:
        label_dict, newData = formatEN_FIN(data)
    else:
        label_dict, newData = formatFRE_SWE(data)
    
    return label_dict, newData


def formatEN_FIN(data):
    all = {} # an empty dictionary
    current_char_count = 0
    all_count = 0
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
        
        # count how many chars the current text has
        current_char_count = len(data[i][1])
        # check whether the current label/key exist already
        if data[i][0] in all:
            all_count = all.get(data[i][0])
            all[data[i][0]] = all_count + current_char_count # update the dictionary
            current_char_count = 0
        else:
            all[data[i][0]] = current_char_count # add to the dictionary
            current_char_count = 0

    for i in sorted(indeces_delete, reverse=True):
        data.pop(i)

    sorted_dict = dict(sorted(all.items(), key=lambda item: item[1], reverse=True))

    return sorted_dict, data


def formatFRE_SWE(data):
    all = {} # an empty dictionary
    current_char_count = 0
    indeces_delete = [] # indeces marked for deletion in the data

    for i in range(len(data)):
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
            else:
                # line 151, 608, 775 in swe_dev has no text even in the original tsv file
                # gets rid of empty lines as well
                indeces_delete.append(i)
                continue
        
        # count how many characters current text has
        current_char_count = len(data[i][3])
        # check whether the current label/key exist already
        if data[i][2] in all:
            all_count = all.get(data[i][2])
            all[data[i][2]] = all_count + current_char_count # update the dictionary
            current_char_count = 0
        else:
            all[data[i][2]] = current_char_count # add to the dictionary
            current_char_count = 0

    # remove the faulty lines
    for i in sorted(indeces_delete, reverse=True):
        data.pop(i)
    # delete unnecessary fields
    new_list = [[row[2], row[3]] for row in data]

    sorted_dict = dict(sorted(all.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_dict, new_list



data = sys.stdin.readlines()
main(data)