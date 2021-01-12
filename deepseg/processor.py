import logging
import os
import re


def read_files(input_files, callback=None, mode='rb', **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for f in input_files:
        if not os.path.exists(f):
            logging.warning('File %s does not exist, skiped.', f)
            continue
        with open(f, mode='rb', encoding=kwargs.get('encoding', None)) as fin:
            while True:
                line = fin.readline()
                if not line:
                    break
                line = line.decode('utf8')
                line = line.rstrip('\n')
                if callback:
                    callback(line)
        logging.info('Finished to read file: %s', f)
    logging.info('Finished to read all files.')


class SIGHANDataProcessor:

    def __init__(self, data_dir, pad_token='[PAD]', unk_token='[UNK]'):
        self.sighan_data_dir = data_dir
        self.train_files = self._parse_training_files()
        self.test_files = self._parse_test_files()
        self.pad_token = pad_token
        self.unk_token = unk_token

    def process(self,
                train_output_file,
                test_features_output_file,
                test_labels_output_file,
                words_output_file,
                vocab_output_file,
                **kwargs):

        w1 = self._collect_examples(self.train_files, train_output_file)
        w2 = self._collect_test_examples(test_features_output_file, test_labels_output_file)

        words = sorted(w1.union(w2))
        with open(words_output_file, mode='wt', encoding='utf8') as fout:
            for w in words:
                fout.write(w + '\n')

        with open(vocab_output_file, mode='wt', encoding='utf8') as fout:
            vocabs = set()
            for w in words:
                for c in w:
                    if not c:
                        continue
                    if c == self.pad_token:
                        continue
                    if c == self.unk_token:
                        continue
                    vocabs.add(c)
            if kwargs.get('add_vocab_pad_token', True):
                fout.write(self.pad_token + '\n')
            if kwargs.get('add_vocab_unk_token', True):
                fout.write(self.unk_token+'\n')
            for v in sorted(vocabs):
                fout.write(v + '\n')

    def _parse_training_files(self):
        training_dir = os.path.join(self.sighan_data_dir, 'training')
        training_files = []
        for f in os.listdir(training_dir):
            if str(f).endswith('.utf8'):
                training_files.append(os.path.join(training_dir, f))
        return training_files

    def _parse_test_files(self):
        test_dir = os.path.join(self.sighan_data_dir, 'testing')
        test_feature_files = []
        for f in os.listdir(test_dir):
            if str(f).endswith('utf8'):
                test_feature_files.append(os.path.join(test_dir, f))
        gold_dir = os.path.join(self.sighan_data_dir, 'gold')
        gold_files = ['as_testing_gold.utf8', 'cityu_test_gold.utf8', 'msr_test_gold.utf8', 'pku_test_gold.utf8']
        test_label_files = [os.path.join(gold_dir, f) for f in gold_files]
        test_files = []
        for f, l in zip(sorted(test_feature_files), sorted(test_label_files)):
            test_files.append((f, l))
        return test_files

    def _collect_examples(self, input_files, output_file):
        words = set()
        with open(output_file, mode='wt', encoding='utf8') as fout:

            def _parse_example(line):
                parts = re.split('\\s+', line)
                parts = [x.strip() for x in parts if x.strip()]
                if not parts:
                    return
                for p in parts:
                    words.add(p)

                example = ' '.join(parts)
                fout.write(example + '\n')

            read_files(input_files, callback=_parse_example)
        return words

    def _collect_test_examples(self, feature_output_file, label_output_file):
        words = set()
        with open(feature_output_file, mode='wt', encoding='utf8') as fea_out, \
                open(label_output_file, mode='wt', encoding='utf8') as lab_out:
            for f, l in self.test_files:
                with open(f, mode='rb') as fea_in, open(l, mode='rb') as lab_in:
                    while True:
                        feature = fea_in.readline()
                        label = lab_in.readline()
                        if not feature or not label:
                            break
                        feature = feature.decode('utf8').rstrip('\r\n').rstrip('\n')
                        label = label.decode('utf8').rstrip('\r\n').rstrip('\n')
                        fea_out.write(feature + '\n')
                        lab_out.write(label + '\n')

                        for w in re.split('\\s+', feature):
                            w = w.strip()
                            if not w:
                                continue
                            words.add(w)
        return words
