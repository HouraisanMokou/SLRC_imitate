this is the folder for placing the original data and corpus

the format of original data is not sure, please check by look at the files

the format of corpus is shown in **process.py**, which is also pinned below

    four files: train.csv, test.csv, val.csv, item_meta.csv
    the formats of each files:

    train.csv: user_id \t item_id
    test.csv: user_id \t item_id \t neg_items
    dev.csv: user_id \t item_id \t neg_items
    item_meta.csv: user_id \t item_id \t reconsumption

    neg_items is a list of all items this user never consume
    reconsumption is a list, the value of ith index represents the time gap from i-1th to ith consumption