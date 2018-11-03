import os
import re

CHINESE_PATTERN = re.compile("^[\u4E00-\u9FA5]+$")


def check_file_exists(file):
    if not os.path.exists(file):
        raise FileNotFoundError("File %s does not exist." % file)


def read_file_line_by_line(file, callback, buffer=None):
    check_file_exists(file)
    with open(file, mode="rt", encoding="utf8", buffering=buffer) as f:
        for line in f:
            if not line:
                continue
            callback(line.strip("\n"))


def is_chinese_word(word):
    return CHINESE_PATTERN.findall(word)
