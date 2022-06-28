import sys

def change_labels(data):
    #list of main labels (8)
    main_labels = ["IN", "NA", "HI", "LY", "IP", "SP", "ID", "OP"]
    # list of full simplified labels (24)
    labels_full = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP', 'av', 'ds', 'dtp', 'ed', 'en', 'fi', 'it', 'lt', 'nb', 'ne', 'ob', 'ra', 're', 'rs', 'rv', 'sr']

    if sys.argv[1] == "full":
        target_labels = labels_full
    else:
        target_labels = main_labels
    indeces_delete = [] # indeces marked for deletion in the data
    
    # split to lists
    for i in range(len(data)):
        data[i] = data[i].replace("\n", "")
        data[i] = data[i].split("\t") # split at the labels
        # split the labels and only take the the ones in caps (main labels)
        labels = data[i][0].split()
        labelstring = ""
        for label in labels:
            if label.lower() in target_labels or label in target_labels: 
                if labelstring == "":
                    labelstring = label
                else:
                    labelstring = labelstring + " " + label
        data[i][0] = labelstring
        # check the right shape
        assert len(data[i]) == 2

        if data[i][0] == "":
            #delete the mistake
            indeces_delete.append(i)

    # remove the faulty lines  (MT machine translation removed etc. )
    for i in sorted(indeces_delete, reverse=True):
        data.pop(i)


    # print to command line to save to files
    final = []
    for i in range(len(data)):
        dummy = data[i][0] + '\t' + data[i][1]
        final.append(dummy)
   
    for i in range(len(final)):
        print(final[i])



data = sys.stdin.readlines()

change_labels(data)