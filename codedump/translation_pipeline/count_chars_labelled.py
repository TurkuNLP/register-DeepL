
import sys

def count_chars(data):
    all = {} # an empty dictionary
    current_char_count = 0
    all_count = 0

    for i in range(len(data)):
        data[i] = data[i].replace("\n", "")
        data[i] = data[i].split("\t") # split at the labels

        #check here that there are always 2 columns (label and text)
        try:
            assert len(data[i]) == 2
        except AssertionError:
            print("ERROR, mistake in formatting")
            print(data[i])

        # #simple line count instead
        # if data[i][0] in all:
        #     line_count = all.get(data[i][0])
        #     all[data[i][0]] = line_count + 1
        # else:
        #     all[data[i][0]] = 1

        # count how many characters current text has
        current_char_count = len(data[i][1])
        
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


data = sys.stdin.readlines()

count_chars(data)