import os
import re
import shutil
import subprocess
import tempfile

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


def count_file_lines(file):
    check_file_exists(file)
    p = os.path.abspath(file)
    return int(subprocess.check_output(["wc", "-l", p]).split()[0])


def split_file(file, parts=128, workdir=None, prefix="part"):
    if not workdir:
        workdir = tempfile.gettempdir()
    if parts <= 1:
        return [os.path.abspath(file)]
    lines = count_file_lines(file)
    m = lines % parts
    lines_each_part = 0 if m == 0 else int((lines - m) / (parts - 1))
    part_files = []
    with open(file, mode="rt", encoding="utf8", buffering=8192) as r:
        for i in range(parts):
            output = os.path.join(workdir, prefix + str(i) + ".txt")
            with open(output, mode="wt", encoding="utf8", buffering=8192) as w:
                for j in range(lines_each_part):
                    line = r.readline()
                    if not line:
                        break
                    w.write(line)
            part_files.append(os.path.abspath(output))

    return part_files


def concat_files(files, output_file):
    with open(output_file, mode="a+", encoding="utf8") as w:
        for f in files:
            check_file_exists(f)
            with open(f, mode="r", encoding="utf8") as r:
                shutil.copyfileobj(r, w)


def remove_dir(directory):
    shutil.rmtree(directory)
