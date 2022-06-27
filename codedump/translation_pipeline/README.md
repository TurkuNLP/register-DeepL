**This README.md serves as a documentation for the python code and bash scripts used in making docx files for translation and turning the files back to tsv.**

The files here are meant to make translating tsv data which are in labels, text format to another language easier and more straightforward. These files are specifically made for the register data (labels text), so for data that is in any other format these files would need to be modified.

## Formatting
FORMAT is a file meant for formatting the texts to labels, texts and getting rid of faulty lines which either have no labels or text or have some extra information to get rid of before they can be used, e.g. id's and source links and extra columns.

# TODO HAVE SEPARATE ONES AND EXPLAIN THAT, CHANGE THE FORMAT BASH SCRIPT TO USE THOSE => SO A BUNCH OF IF CLAUSES


# shortening
shorten.py is meant for shortening tsv data to a more reasonable length so that one example in the data does not take too much of translation space. HOX! The length must be specified in terms of characters, not words. Before deciding on a length for the texts it might be a good idea to check how long the texts are in the original data.

example usage:

```
cat [FILE] | python3 shorten.py [LENGTH] > truncated.tsv
```
```
cat [FILE] | python3 shorten.py [LENGTH] | gzip > truncated.tsv.gz
```

There is also an example script shorten_all.sh for running all the files in a loop, file paths for the script and where to save the files need to be changed.


# downsampling
downsample.py is meant for downsampling the data. There are two ways to use it: downsample the biggest classes so that they are the same size as the next class and then downsample every class by X percent so that the overall character count is below a certain treshold or just downsample by X percent. Of course no downsampling is also an option, then you give no arguments to python script and it just saves to a new file.

example usage:

CLASS decides which class in size order is the cap for the amount of characters for other classes

```
zcat [FILE] | python3 downsample.py [CLASS] [PERCENT] > downsampled.tsv
```
```
cat [FILE] | python3 downsample.py [PERCENT] > downsampled.tsv
```

There is a bash script that takes into account all the register files and how to sample them and what numbers to use which can be used as an example. File paths may have to be changed to make it work.


To use downsampling on the same file again, splicing of examples needs to be changed to continue from the previous spot. E.g., [num:num+num]

To make sure there are no duplicates, a script could compare the different files so that things are not translated twice but this would need the shortening to be identical. (duplicate detection will not work with register stuff because e.g. shortening and labels have changed since)


# counting characters
count_chars_labelled.py is meant to count how many characters there are in all label combinations, it prints a dictionary where the character counts are and what label they belong to. A line count for labels is also possible but is currently commented out.

Usage example:

```
cat $f | python3 count_chars_labelled.py
```

A bash script is also available which goes through all the files, file paths may need to be changed.


count_chars.sh is a script for seeing how many characters a file has.


# from tsv to docx
todocx.py is meant for turning the data to a docx format which can then be fed to DeepL. The texts are saved to .docx and the labels to corresponding .txt files.

The directory where to save the labels and the docx files has to changed in the code because it is currently hardcoded. 
# TODO CHANGE THAT? ADD SYSARGS FOR THE PATHS

The python script should be used with the bash script todocx.sh.



# from docx to tsv
fromdocx.py is used to turn the translated .docx files back to .tsv files. It uses the red color of the id's to see that nothing is missing and finds the correct label from a .txt file by counting the paragraphs which have id's. 

The python script should be used with the bash script fromdocx.sh.



# TODO ADD THE CHECK FOR EMPTY PARAGRAPHS DUE TO WARNING FROM FILIP



# TODO SAY SOMETHING ABOUT THE AUTOMATE DOWNSAMPLING SCRIPT?