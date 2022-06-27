import sys

# cat multilingual-register-data-new/originals/swe_train.tsv | python3 codedump/formatSwe_Fre.py | python3 codedump/change_labeling_for_train.py > monolingual_benchmark_files/swe_train.tsv

data = sys.stdin.readlines()

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

    if data[i][2] == "":
        indeces_delete.append(i)
        #print(data[i], i)
    if data[i][3] == "":
        indeces_delete.append(i)

# remove the faulty lines
for i in sorted(indeces_delete, reverse=True):
    data.pop(i)

# delete unnecessary fields (id and source link etc.)
new_list = [[row[2], row[3]] for row in data]

final=[]
for i in range(len(new_list)):
        dummy = new_list[i][0] + '\t' + new_list[i][1]
        final.append(dummy)
   
for i in range(len(final)):
    print(final[i])