import sys


# delete the sublabels altogether since they are unnecessary for now at least -Veronika

def change_labels(data):
    #list of main labels (8)
    main_labels = ["IN", "NA", "HI", "LY", "IP", "SP", "ID", "OP"]
    # split to lists
    for i in range(len(data)):
        data[i] = data[i].replace("\n", "")
        data[i] = data[i].split("\t") # split at the labels
        # split the labels and only take the the ones in caps (main labels)
        labels = data[i][0].split()
        labelstring = ""
        for label in labels:
            if label in main_labels:
                if labelstring == "":
                    labelstring = label
                else:
                    labelstring = labelstring + " " + label
        data[i][0] = labelstring
        # check the right shape
        assert len(data[i]) == 2

    
    # print to command line to save to files
    final = []
    for i in range(len(data)):
        dummy = data[i][0] + '\t' + data[i][1]
        final.append(dummy)
   
    for i in range(len(final)):
        print(final[i])



data = data = sys.stdin.readlines()

change_labels(data)