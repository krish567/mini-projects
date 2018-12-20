import os
import sys
import shutil
import numpy as np


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def isdir(path):
    return os.path.isdir(path)


def isfile(path):
    return os.path.isfile(path)


def join(*paths):
    return os.path.join(*paths)


def getfiles(path, fls=False, cond=''):
    if fls:
        if cond == '':
            return [join(path, f) for f in os.listdir(path)]
        else:
            return [join(path, f) for f in os.listdir(path) if cond in f]
    else:
        if cond == '':
            return os.listdir(path)
        else:
            return [f for f in os.listdir(path) if cond in f]


def numkeys(dt):
    assert type(dt) == dict
    return len(list(dt.keys()))


def ifdulicates(lst):
    assert type(lst) == list
    return len(lst) == len(set(lst))


def returnRandomKey(dt):
    keys_list = list(dt.keys())
    r = np.random.randint(len(keys_list))
    return keys_list[r]


def numelems(var, criterion, th):
    if type(var) == np.ndarray:
        if criterion == ">":
            return sum(i > th)
        elif criterion == "<":
            return sum(i < th)
        elif criterion == "=":
            return sum(i == th)
        else:
            raise 2
            print(
                "criterion should be one of the following '<' ' > ' '=' and not"
                + str(criterion)
            )

    elif type(var) == list:
        if criterion == ">":
            return sum(i > th for i in var)
        elif criterion == "<":
            return sum(i < th for i in var)
        elif criterion == "=":
            return sum(i == th for i in var)
        else:
            raise 2
            print(
                "criterion should be one of the following '<' '>' '=' and not"
                + str(criterion)
            )
    else:
        raise 1
        print(
            "supported types for var are : list, numpy.ndarray but given "
            + str(type(var))
        )


"""

TODO
1. Test each of them
2. Write better or faster versions if available

"""


def copyfolder(folder, dest):
    shutil.copytree(folder, dest)
