""" Read/write from/to disk
"""
import pandas, os, config


def print_dict(dirname="", d={}, name="text"):
    if not dirname == "":
        dirname += "/"
    name += ".txt"
    with open(dirname + "0_" + name, "w") as text_file:
        print(name + "\n", file=text_file)
        for k, v in d.items():
            # print(f"{k}:{v}", file=text_file) # pythonw, python3
            print('{:s}, {:s}'.format(str(k), str(v)), file=text_file)


def save_dict_to_csv(dirname, name, data):
    # panda df requires data to be NOT of type {key: scalar}
    # but rather: {'name':['value']}
    if not dirname[-1] == '/':
        dirname += '/'
    filename = dirname + name + ".csv"
    df = pandas.DataFrame(data=data)
    df.to_csv(filename, sep=',', index=False)
    # mkdir filename
    # for k in d.keys(): gen png
    return filename
