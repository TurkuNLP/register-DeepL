import sys


# before running this we need to shorten like we did for the downsampled data to make the texts match
# zcat old-datasets/multilingual-register-data-new/originals/en_full_train.tsv.gz | python3 codedump/not_in_current_use/shorten.py | gzip > en_shortened_full.tsv.gz
# actually even this does not work because the shortening was done on the whole line (including labels) so now that there are more labels everything breaks

# what if I just take some 100 characters of the text and compare those, should work all the same I guess


# the biggest problem with doing this is matching the labels for the translated texts since the labels are in files of their own
# one thing I could try is just counting from the downsampled file for the new label files and check that the counts matchs
# another is running the todoxc again for the english data so it makes new label files
# ooor another is doing the counting for the final translated files since everything is in the same order
# todoxc might actually be better because then I can reuse the label files to make new files???? => done, now I remake those translated files


# hey 
# @Anni Eskelinen
#  since you can rely on texts matching (right?) then you can simply make a dictionary from Anna's data with the texts as keys and fine-grained labels as values

# then simply use this dictionary to look up the labels for your texts

# epsilon-it.utu.fi has crapload of memory so you can simply make a dictionary like this directly

# a more memory-aware approach would be to hash the texts and store only their hashes in the dictionary as keys




# read the full core from the command line zcat old-datasets/multilingual-register-data-new/originals/en_full_train.tsv.gz | python3 fix_en_labels.py > en_train_downsampled_full.tsv


data = sys.stdin.readlines()

for i in range(len(data)):
    data[i] = data[i].replace("\n", "")
    data[i] = data[i].split("\t")
    data[i][1] = data[i][1][:100] # take first 100 chars from the text


fname = "downsampled/en_train.downsampled.tsv"
with open (fname) as f:
    lines = f.readlines()

for i in range(len(lines)):
    lines[i] = lines[i].replace("\n", "")
    lines[i] = lines[i].split("\t")


# then map to a dictionary texts as keys and labels as values

dictionary = {}

for i in range(len(data)):
    dictionary[data[i][1]] = data[i][0] # text (100 chars) as key and value as label

# then go through the downsampled en file and change the labels by looking the labels up with the text

for i in range(len(lines)):
    if dictionary.get(lines[i][1][:100]) == None: #.get() returns None if it is not found
        # I would want to know if this happens though....
        print("NOT FOUND")
        continue
    else:
        lines[i][0] = dictionary[lines[i][1][:100]] # take first 100 chars from the text # this could throw an error if no key is found which is why I have the if else

# then print for the new downsampled en file

final=[]
for i in range(len(lines)):
        dummy = lines[i][0] + '\t' + lines[i][1]
        final.append(dummy)
   
for i in range(len(final)):
    print(final[i])