"""
Join separate SRC and TG files into tsv
"""

# temp = open(filename,'r').read().split('\n')

def zip_write(src, tg, new_name):
    with open(new_name, "w") as f:
        for (src_segment, tg_segment) in zip(src, tg):
            f.write(src_segment + "\t" + tg_segment + "\n")
            print(src_segment, tg_segment)
    print("Done")



src_train = open("src_train.txt", "r").read().split("\n")
tg_train = open("tg_train.txt", "r").read().split("\n")

src_val = open("src_val.txt", "r").read().split("\n")
tg_val = open("tg_val.txt", "r").read().split("\n")

src_test = open("src_test.txt", "r").read().split("\n")
tg_test = open("tg_test.txt", "r").read().split("\n")

zip_write(src_test, tg_test, "keyvalue_test.txt")
zip_write(src_train, tg_train, "keyvalue_train.txt")
zip_write(src_val, tg_val, "keyvalue_val.txt")