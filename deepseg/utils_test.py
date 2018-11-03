import tensorflow as tf
from deepseg import utils
import subprocess
import os
import shutil


class UtilsTest(tf.test.TestCase):

    def _createTestSegFile(self):
        tmp_dir = "/tmp/deepseg"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        output = os.path.join(tmp_dir, "big_file.txt")

        with open(output, mode="w+", encoding="utf8", buffering=8192) as f:
            for _ in range(100000):
                f.write("Hello World!\n")
        return tmp_dir, output

    def testSplitAndConcatFile(self):
        tmp_dir, file = self._createTestSegFile()
        workdir = os.path.join(tmp_dir, "split")
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        parts = utils.split_file(file, workdir=workdir)
        print(subprocess.check_output(["head", "-n", "10", parts[0]]))
        for p in parts:
            print(p)
        concat = os.path.join(tmp_dir, "concat.txt")
        utils.concat_files(parts, concat)
        shutil.rmtree(workdir)


if __name__ == "__main__":
    tf.test.main()
