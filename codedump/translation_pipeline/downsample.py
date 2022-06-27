import sys
import random


# the idea is to get to below 25 million for all of the texts so we can translate to four different languages

def main(data):
    random.Random(1234).shuffle(data)
    #use argv to choose what downsampling strategy to choose
    if len(sys.argv) == 1:
        # no downsampling done so only put it to a new file
        print(data)
    elif len(sys.argv) == 3:
        downsampleByAll(data)
    else:
        label_dict, sampled = count_chars(data)
        downsampleByPercent(sampled)
  

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


        # here the splicing has to be changed if the same file is to be used again.
        # a check that the num:num+num does not go above is needed
        # if num+num > len(textlist):
        #     shortened = textlist[num:]
        # else:
        #     shortened = textlist[num:num+num]

        shortened = textlist[:num] # here I am just getting the percent of lines left (not counting characters here at all)
        #print(key_list[i], percent, num, len(textlist), len(shortened))


        # now we change the format back to tsv
        for j in range(len(shortened)):
            dummy = key_list[i] + '\t' + shortened[j] # a dummy text
            final.append(dummy)
        
    # shuffle the data because it is in order now
    random.Random(1234).shuffle(final)

    for i in range(len(final)):
        print(final[i])


def get_biggest_labels(data, cap):
    label_dict, newData = count_chars(data)
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


def count_chars(data):
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
            print("ERROR, a mistake in formatting")
            print(data[i])


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


data = sys.stdin.readlines()
main(data)