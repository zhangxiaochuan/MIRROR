# -*- coding: utf-8 -*- 
# @Time : 2019-10-24 19:15 
# @Author : Xiaochuan Zhang


import re
import os
import pickle
from tqdm import tqdm
import random
from data_processor import BasicBlockProcessor, TokenIDManager
import hashlib
import torch
import torch.nn as nn
from init_models import init_transformer


def list_hash(data):
    string = ""
    step = 1000
    index = 0
    while index < len(data):
        string += str(data[index])
        index += step
    return hashlib.md5(string.encode()).hexdigest()


def deepcopy(data):
    if not os.path.exists('.tmp'):
        os.mkdir('.tmp')
    data_hash = list_hash(data)
    file_path = os.path.join('.tmp', data_hash + '.pkl')
    if not os.path.exists(file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    with open(file_path, 'rb') as f:
        return pickle.load(f)


class Dataset(object):
    """manage the dataset"""

    def __init__(self, dataset_dir='../MISA', x86_source_dir='source/x86/', arm_source_dir='source/arm/',
                 x86_vocab_path='x86_vocab.txt', arm_vocab_path='arm_vocab.txt',
                 x86_train_path='basic_blocks/x86_train.txt', arm_train_path='basic_blocks/arm_train.txt',
                 data_path='basic_blocks/data.pkl'):
        """
        init Dataset class
        :param dataset_dir: the root directory of MISA
        :param x86_source_dir: the directory which stores source files on x86
        :param arm_source_dir: the directory which stores source files on ARM
        :param x86_vocab_path: the vocabulary file of x86
        :param arm_vocab_path: the vocabulary file of ARM
        :param x86_train_path: the file path stores x86 basic blocks
        :param arm_train_path: the file path stores ARM basic blocks
        :param data_path: the output pkl file
        """
        self.dataset_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), '../../', dataset_dir))
        self.x86_source_dir = os.path.join(self.dataset_dir, x86_source_dir)
        self.arm_source_dir = os.path.join(self.dataset_dir, arm_source_dir)
        self.x86_vocab_path = os.path.join(self.dataset_dir, x86_vocab_path)
        self.arm_vocab_path = os.path.join(self.dataset_dir, arm_vocab_path)
        self.x86_train_path = os.path.join(self.dataset_dir, x86_train_path)
        self.arm_train_path = os.path.join(self.dataset_dir, arm_train_path)
        self.data_path = os.path.join(self.dataset_dir, data_path)

    def load(self):
        """load the dataset"""
        if not self.__been_preprocessed():
            self.preprocess()

        arm_basic_block_processor = BasicBlockProcessor('arm', self.arm_vocab_path)
        x86_basic_block_processor = BasicBlockProcessor('x86', self.x86_vocab_path)

        with open(self.data_path, 'rb') as f:
            raw_data = pickle.load(f)
            random.shuffle(raw_data)

        data = []
        for sample in tqdm(raw_data, desc="token2id"):
            if sample["x86"].__class__ is str:
                x86 = sample["x86"]
                arm = sample["arm"]
            else:
                x86 = sample["x86"].decode()
                arm = sample["arm"].decode()
            data.append({"x86": x86_basic_block_processor.to_ids(x86),
                         "arm": arm_basic_block_processor.to_ids(arm)})

        x86_token_id_manager = TokenIDManager(self.x86_vocab_path)
        arm_token_id_manager = TokenIDManager(self.arm_vocab_path)
        return data, len(x86_token_id_manager.vocab_list) + 1, len(arm_token_id_manager.vocab_list) + 1  # [UNK]

    def preprocess(self):
        """preprocess the assembly files, generate the .pkl file and vocabulary files"""
        if os.path.exists(self.data_path):
            return

        if not os.path.exists(self.x86_train_path) and not os.path.exists(self.arm_train_path):
            x86_bbs = []
            arm_bbs = []
            arm_basic_block_processor = BasicBlockProcessor('arm', self.arm_vocab_path)
            x86_basic_block_processor = BasicBlockProcessor('x86', self.x86_vocab_path)

            arm_failed_sum = 0
            x86_failed_sum = 0
            for file_name in tqdm(os.listdir(self.x86_source_dir), "generating data set"):
                if file_name[0] == '.':
                    continue
                arm_source_file = os.path.join(self.arm_source_dir, file_name)
                x86_source_file = os.path.join(self.x86_source_dir, file_name)
                if os.path.exists(x86_source_file) and os.path.exists(arm_source_file):
                    arm_blocks, arm_failed_count = self.__parser(arm_source_file, "arm")
                    x86_blocks, x86_failed_count = self.__parser(x86_source_file, "x86")

                    arm_failed_sum += arm_failed_count
                    x86_failed_sum += x86_failed_count

                    # remove the basic blocks which parse failed
                    if not len(arm_blocks) == len(x86_blocks):
                        keys = (set(arm_blocks.keys()) | set(x86_blocks.keys())) - (
                                set(arm_blocks.keys()) & set(x86_blocks.keys()))
                        for key in keys:
                            if key in arm_blocks:
                                arm_blocks.pop(key)
                            if key in x86_blocks:
                                x86_blocks.pop(key)

                    for bb in arm_blocks:
                        normalized_bb = arm_basic_block_processor.normalize(arm_blocks[bb])
                        arm_bbs.append(normalized_bb)
                    for bb in x86_blocks:
                        normalized_bb = x86_basic_block_processor.normalize(x86_blocks[bb])
                        x86_bbs.append(normalized_bb)

            print("arm failed sum: ", arm_failed_sum)
            print("x86 failed sum: ", x86_failed_sum)

            with open(self.arm_train_path, "w") as f_arm:
                f_arm.write("\n".join(arm_bbs))

            with open(self.x86_train_path, "w") as f_x86:
                f_x86.write("\n".join(x86_bbs))

            self.__vocab_generator(x86_bbs, arm_bbs)

        print("generating pkl file...")
        data = []
        with open(self.data_path, 'wb') as f_data, \
                open(self.x86_train_path, "rb") as f_x86, \
                open(self.arm_train_path, "rb") as f_arm:
            for bb_x86, bb_arm in zip(f_x86, f_arm):
                data.append(
                    {
                        "x86": bb_x86,
                        "arm": bb_arm
                    }
                )
            pickle.dump(data, f_data)
        print("done.")

    @staticmethod
    def __parser(filename, arch):
        """
        parse basic block ID
        :param filename: assembly file name
        :param arch: architecture，x86 or ARM
        :return: a map of basic blocks and their ids, failure count
        """

        blocks = {}
        blocks_id = {}
        regs = {
            "x86": [r"movabsq\s\$(\.L\d*),\s%rdi;movb\s\$0,\s%al;callq\sprintf;",
                    r"movl\s\$(\.L\d*),\s%edi;xorl\s%eax,\s%eax;callq\sprintf;"],
            "arm": [r"ldr\sr0,\s(\.LCPI\d*_\d*);(.*?)bl\sprintf;"]
        }
        regs = regs[arch]

        label = ""
        for line in open(filename):
            if line == "\n": continue
            if len(re.findall(r"\.Ltmp[0-9_]+:", line)) != 0:
                continue
            if len(re.findall(r"LPC[0-9_]+:", line)) != 0:
                continue
            if not line[0] in ["\t", " "]:
                label = line.split(":")[0]
                while blocks.__contains__(label):
                    label += "1"
                blocks[label] = []
            elif not label == "":
                instruction = line.strip()
                if not instruction[0] in ["#", "@"]:
                    # remove the annotations
                    if arch is 'x86':
                        blocks[label].append(re.sub(r"#.+", "", instruction))
                    elif arch is 'arm':
                        blocks[label].append(re.sub(r"@.+", "", instruction))

        failed_count = 0
        for i in blocks:
            for reg in regs:
                ret = re.findall(reg, ";".join(blocks[i]))
                if len(ret) > 0:
                    break
            if len(ret) > 0:
                if arch == "x86":
                    try:
                        bid = blocks[ret[0]][0].split('"')[1].split('"')[0]
                        blocks_id[bid] = blocks[i]
                    except KeyError:
                        pass
                elif arch == "arm":
                    try:
                        bid = blocks[blocks[ret[0][0]][0].split()[1]][0].split()[1].replace('"', "")
                        blocks_id[bid] = blocks[i]
                    except KeyError:
                        pass
            elif "".join(blocks[i]).__contains__("printf") and len(blocks[i]) > 2:
                failed_count += 1
        for bid in list(blocks_id):
            bb = list(blocks_id[bid])
            if len(bb) == 0:
                del (blocks_id[bid])
            for b in blocks_id[bid]:
                if b[0] == '.':
                    if b.__contains__(".cfi"):
                        pass
                    else:
                        bb.remove(b)

            bb_str = ";".join(bb)
            if arch == "x86":
                for reg in regs:
                    bb_str = re.sub(reg, "", bb_str)
                bb = bb_str.split(";")
            elif arch == "arm":
                reg = regs[0]
                try:
                    match = re.findall(reg, bb_str)[0][1]
                except BaseException:
                    pass
                if len(match) is not 0:
                    pass
                bb_str = re.sub(reg, match, bb_str)
                bb = bb_str.split(";")

            if len(bb) == 0 or len(bb) > 200 or len(re.split(";|,|\s", "\n".join(bb))) > 500:
                blocks_id.pop(bid)
            else:
                blocks_id[bid] = bb
        return blocks_id, failed_count

    def __vocab_generator(self, x86_bbs, arm_bbs):
        """
        generate the vocabularies
        :param x86_bbs: x86 basic blocks
        :param arm_bbs: ARM basic blocks
        """
        print("generating vocabulary...")

        reserved_vocab = ["padding", "<s>", "</s>"]

        x86_tokens = self.__get_uniq_tokens(x86_bbs)
        arm_tokens = self.__get_uniq_tokens(arm_bbs)

        x86_vocab = reserved_vocab + x86_tokens
        arm_vocab = reserved_vocab + arm_tokens

        assert x86_vocab[0] is "padding", "「padding」 index should be 0 in x86 vocab"
        assert arm_vocab[0] is "padding", "「padding」 index should be 0 in arm vocab"

        with open(self.arm_vocab_path, "w") as f_arm:
            f_arm.write("\n".join(arm_vocab))
        with open(self.x86_vocab_path, "w") as f_x86:
            f_x86.write("\n".join(x86_vocab))

    @staticmethod
    def __get_uniq_tokens(bbs):
        """get unique tokens from given basic blocks"""
        tokens = []
        insts = []
        for bb in bbs:
            insts += bb.split("\n")
        for inst in insts:
            tokens += inst.split()
        tokens = list(set(tokens))
        print(tokens)
        return tokens

    def __been_preprocessed(self):
        """judge where the data has been preprocessed yet"""
        if os.path.exists(self.data_path) and os.path.exists(self.x86_vocab_path) and os.path.exists(
                self.arm_vocab_path):
            return True
        else:
            return False


class DataLoader(object):
    def __init__(self, data, batch_size, max_len=None, token_pad_idx=0,
                 require_negative_samples=True, seed=2019):
        """
        init DataLoader
        :param data: all data in MISA
        :param batch_size: batch size for model training
        :param max_len: max len of basic block
        :param token_pad_idx: padding token index
        :param require_negative_samples: if need negative samples
        :param seed: random seed
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.require_negative_samples = require_negative_samples

        if max_len is not None:
            self.max_len = max_len
        else:
            self.max_len = self.get_max_len(data)

        random.seed(seed)
        random.shuffle(data)

        data = self.padding(data, token_pad_idx)

        if self.require_negative_samples:
            data = self.insert_negative_samples(data, method='hard')

        x = int(0.8 * len(data))
        self.train = data[:x]
        self.val = data[x:]

        print("Train data")
        print(len(self.train))
        print("Val data")
        print(len(self.val))

        random.shuffle(self.train)
        random.shuffle(self.val)

    def get_train_and_val_size(self):
        return len(self.train), len(self.val)

    def insert_negative_samples(self, raw_data, n=6, method='random'):
        data_hash = list_hash(raw_data)
        data_path = os.path.join('.tmp', data_hash + '_' + method + "_" + str(n) + ".pkl")
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                return pickle.load(f)

        random.seed(2019)
        if method == 'random':
            data_copy = deepcopy(raw_data)
            for sample in tqdm(raw_data, "generating random negative samples"):
                random_samples = random.sample(data_copy, n + 1)
                try:
                    random_samples.remove(sample)
                except BaseException:
                    random_samples = random_samples[:n]

                if sample.keys().__contains__("negative"):
                    break

                for index in range(len(random_samples)):
                    if index < n / 2:
                        if index == 0:
                            sample['negative'] = random_samples[index]["arm"]
                            sample['negative_encoder'] = 1
                        else:
                            raw_data.append({"x86": sample["x86"], "arm": sample["arm"],
                                             "negative": random_samples[index]["arm"], "negative_encoder": 1})
                    else:
                        raw_data.append({"x86": sample["x86"], "arm": sample["arm"],
                                         "negative": random_samples[index]["x86"], "negative_encoder": 0})

        elif method == 'hard':
            random_negatives = self.insert_negative_samples(deepcopy(raw_data), int(n * 2 / 3), 'random')
            with_x86_encoding_file = os.path.join('.tmp', data_hash + "_x86_encoding.pkl")

            if os.path.exists(with_x86_encoding_file):
                with open(with_x86_encoding_file, 'rb') as f:
                    data_copy = pickle.load(f)
            else:
                pretrained_transformer, _ = init_transformer()
                encoder = pretrained_transformer.encoder.to(self.device)
                encoder.no_grads()
                n_gpu = torch.cuda.device_count()
                if n_gpu > 1:
                    encoder = torch.nn.DataParallel(encoder)
                pos = torch.LongTensor([list(range(1, 1 + len(raw_data[0]['x86'])))]).to(self.device)
                data_copy = deepcopy(raw_data)
                for sample in tqdm(data_copy, "x86 encoding"):
                    x86 = torch.LongTensor([sample["x86"]]).to(self.device)
                    masks = x86.gt(0).long()
                    x86_pos = torch.mul(pos, masks)
                    sample["x86_encoding"] = encoder(x86, x86_pos)[0].sum(1).to('cpu').numpy()
                with open(with_x86_encoding_file, 'wb') as f:
                    pickle.dump(data_copy, f)

            raw_data = deepcopy(data_copy)
            token = 0
            for sample in tqdm(raw_data, "generating hard negative samples"):
                if sample.keys().__contains__("negative"):
                    break
                sample_x86_encoding = torch.Tensor(sample["x86_encoding"])

                if token > len(data_copy) - 100:
                    random.shuffle(data_copy)
                    token = 0
                random_samples = data_copy[token:100 + token]
                token += 100

                encodings = []
                for r_sample in random_samples:
                    encodings.append(r_sample["x86_encoding"])
                encodings = torch.Tensor(encodings).to(self.device)
                sample_x86_encodings = sample_x86_encoding.repeat(100, 1).to(self.device)

                distances = nn.PairwiseDistance().to(self.device)(encodings, sample_x86_encodings).to(
                    'cpu').numpy().tolist()

                try:
                    self_index = distances.index(0)
                    distances[self_index] = 10000
                except BaseException:
                    pass

                indices = sorted(range(len(distances)), key=lambda i: distances[i])[:int(n / 3)]
                random.shuffle(indices)
                for i in range(len(indices)):
                    index = indices[i]
                    if i < len(indices) / 2:
                        if i == 0:
                            sample['negative'] = random_samples[index]["arm"]
                            sample['negative_encoder'] = 1
                        else:
                            raw_data.append({"x86": sample["x86"], "arm": sample["arm"],
                                             "negative": random_samples[index]["arm"], "negative_encoder": 1})
                    else:
                        raw_data.append({"x86": sample["x86"], "arm": sample["arm"],
                                         "negative": random_samples[index]["x86"], "negative_encoder": 0})

            for sample in raw_data:
                if sample.keys().__contains__("x86_encoding"):
                    sample.pop("x86_encoding")
            raw_data = raw_data + random_negatives

        with open(data_path, 'wb') as f:
            pickle.dump(raw_data, f)
        return raw_data

    def padding(self, data, token_pad_idx):
        index = 0
        while index < len(data):
            x86 = data[index]['x86']
            arm = data[index]['arm']
            if len(x86) > self.max_len or len(arm) > self.max_len:
                data.pop(index)
            else:
                data[index]['x86'] = x86 + [token_pad_idx] * (self.max_len - len(x86))
                data[index]['arm'] = arm + [token_pad_idx] * (self.max_len - len(arm))
                index += 1
        return data

    @staticmethod
    def get_max_len(data):
        max_len = 0
        for sample in data:
            max_len = max(len(sample["x86"]), len(sample["arm"]), max_len)
        return max_len

    def data_iterator(self, data_split, shuffle=True):
        if data_split == "train":
            data_list = self.train
        elif data_split == "val":
            data_list = self.val
        else:
            raise Exception

        if shuffle:
            random.shuffle(data_list)

        if not self.require_negative_samples:

            for i in range(len(data_list) // self.batch_size):
                inputs = [data["x86"] for data in data_list[i * self.batch_size:(i + 1) * self.batch_size]]
                outputs = [data["arm"] for data in data_list[i * self.batch_size:(i + 1) * self.batch_size]]

                batch_inputs = torch.LongTensor(inputs)
                batch_outputs = torch.LongTensor(outputs)

                pos = torch.LongTensor(list(range(1, 1 + self.max_len))).expand([self.batch_size, self.max_len])
                input_masks = batch_inputs.gt(0).long()
                output_masks = batch_outputs.gt(0).long()

                input_pos = torch.mul(pos, input_masks)
                output_pos = torch.mul(pos, output_masks)

                batch_inputs, input_pos, batch_outputs, output_pos = (
                    batch_inputs.to(self.device),
                    input_pos.to(self.device),
                    batch_outputs.to(self.device),
                    output_pos.to(self.device)
                )
                assert len(batch_outputs) == len(batch_inputs) == len(input_pos) == len(output_pos)
                yield batch_inputs, input_pos, batch_outputs, output_pos
        else:

            for i in range(len(data_list) // self.batch_size):
                x86s = [data["x86"] for data in data_list[i * self.batch_size:(i + 1) * self.batch_size]]
                arms = [data["arm"] for data in data_list[i * self.batch_size:(i + 1) * self.batch_size]]
                negatives = [data["negative"] for data in data_list[i * self.batch_size:(i + 1) * self.batch_size]]
                negative_encoders = [data["negative_encoder"] for data in
                                     data_list[i * self.batch_size:(i + 1) * self.batch_size]]

                batch_x86s = torch.LongTensor(x86s)
                batch_arms = torch.LongTensor(arms)
                batch_negatives = torch.LongTensor(negatives)
                batch_negative_encoders = torch.LongTensor(negative_encoders)

                pos = torch.LongTensor(list(range(1, 1 + self.max_len))).expand([self.batch_size, self.max_len])
                x86_masks = batch_x86s.gt(0).long()
                arm_masks = batch_arms.gt(0).long()
                negative_masks = batch_negatives.gt(0).long()

                x86_pos = torch.mul(pos, x86_masks)
                arm_pos = torch.mul(pos, arm_masks)
                negative_pos = torch.mul(pos, negative_masks)

                batch_x86s, x86_pos, batch_arms, arm_pos, batch_negatives, negative_pos, negative_encoders = (
                    batch_x86s.to(self.device),
                    x86_pos.to(self.device),
                    batch_arms.to(self.device),
                    arm_pos.to(self.device),
                    batch_negatives.to(self.device),
                    negative_pos.to(self.device),
                    batch_negative_encoders.to(self.device)
                )
                assert len(batch_x86s) == len(x86_pos) == len(batch_arms) == len(arm_pos) == len(
                    batch_negatives) == len(negative_pos) == len(negative_encoders)
                yield batch_x86s, x86_pos, batch_arms, arm_pos, batch_negatives, negative_pos, negative_encoders


if __name__ == '__main__':
    Dataset().preprocess()
