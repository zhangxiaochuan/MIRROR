# -*- coding: utf-8 -*- 
# @Time : 2019-10-23 16:04 
# @Author : Xiaochuan Zhang
import re
import os


class TokenIDManager(object):
    def __init__(self, vocab_path):
        self.vocab_list = self.load_vocab_list(vocab_path)
        self.token2id_map = self.__generate_token2id_vocab_map()
        self.id2token_map = self.__generate_id2token_vocab_map()

    @staticmethod
    def load_vocab_list(vocab_path):
        """load vocabulary"""
        assert os.path.exists(vocab_path)
        with open(vocab_path, 'r') as f:
            vocab_list = f.readlines()
        return vocab_list

    def token2id(self, token):
        """convert token to id"""
        if self.token2id_map.__contains__(token):
            vocab_id = self.token2id_map[token]
        else:
            vocab_id = len(self.token2id_map)
        return vocab_id

    def __generate_token2id_vocab_map(self):
        """generate the map from token to id"""
        vocab_map = {}
        for i in range(len(self.vocab_list)):
            vocab_map[self.vocab_list[i].strip()] = i
        return vocab_map

    def __generate_id2token_vocab_map(self):
        """generate the map from id to token"""
        vocab_map = {}
        for i in range(len(self.vocab_list)):
            vocab_map[i] = self.vocab_list[i].strip()
        return vocab_map


class InstructionProcessor(object):
    def __init__(self, arch):
        """
        InstructionProcessor
        :param arch: architecture, x86 or ARM
        """
        if arch.lower() == 'x86':
            self.normalizer = self.X86Pass
        elif arch.lower() == 'arm':
            self.normalizer = self.ArmPass
        else:
            raise Exception

    def normalize(self, inst):
        """
        normalize assembly instruction
        :param inst: assembly instruction
        :return: normalized assembly instruction
        """
        inst = self.normalizer.remove_annotations(inst)
        inst = self.normalizer.remove_unusable_punctuation(inst)
        inst = self.normalizer.immediate_value(inst)
        inst = self.normalizer.address(inst)
        inst = self.normalizer.function_call(inst)
        inst = self.normalizer.bb_label(inst)
        inst = self.normalizer.reg_class(inst)
        inst = self.normalizer.variable(inst)
        return inst

    class X86Pass(object):
        @classmethod
        def remove_unusable_punctuation(cls, inst):
            return inst.replace(',', ' ')

        @classmethod
        def remove_annotations(cls, inst):
            return re.sub(r"#.+", "", inst)

        @classmethod
        def immediate_value(cls, inst):
            return re.sub(r"\$[\-]?[0-9]+", 'IMM', inst)

        @classmethod
        def address(cls, inst):
            return re.sub(r"[^\(\s]*\(.+\)", "ADDRESS", inst)

        @classmethod
        def variable(cls, inst):
            inst = re.sub(r"\$[\.a-zA-Z][^\s]+", "VAR", inst)
            inst_list = inst.split()
            for i in range(1, len(inst_list)):
                operand = inst_list[i]
                if operand not in ["VAR", "IMM", "ADDRESS", "FUNC", "BB"] and operand[0] != "%":
                    inst_list[i] = "VAR"
            return " ".join(inst_list)

        @classmethod
        def function_call(cls, inst):
            return re.sub(r"callq\s.+", "callq FUNC", inst)

        @classmethod
        def bb_label(cls, inst):
            return re.sub(r"\s\..+$", " BB", inst)

        @classmethod
        def reg_class(cls, inst):
            regs = {
                "%reg_data_64": r"%r[abcd]x\b",
                "%reg_data_32": r"%e[abcd]x\b",
                "%reg_data_16": r"%[abcd]x\b",
                "%reg_data_8": r"%[abcd][lh]\b",

                "%reg_addr_64": r"%r[sd]i\b",
                "%reg_addr_32": r"%e[sd]i\b",
                "%reg_addr_8": r"%[sd]il\b",
                "%reg_addr_16": r"%[sd]i\b",

                "%reg_data_float": r"%xmm[0-9]+\b",

                "%reg_gen_32": r"%r[0-9]+d\b",
                "%reg_gen_16": r"%r[0-9]+w\b",
                "%reg_gen_8": r"%r[0-9]+b\b",
                "%reg_gen_64": r"%r[0-9]+\b",

                "%reg_pointer": r"%(rbp|rsp|ebp|esp|bp|sp)\b"
            }
            for flag in regs:
                inst = re.sub(regs[flag], flag, inst)
            return inst

    class ArmPass(object):
        @classmethod
        def remove_unusable_punctuation(cls, inst):
            inst = inst.replace(',', ' ')
            inst = inst.replace('{', ' ')
            inst = inst.replace('}', ' ')
            return inst

        @classmethod
        def remove_annotations(cls, inst):
            return re.sub(r"@.+", "", inst)

        @classmethod
        def immediate_value(cls, inst):
            return re.sub(r"#[\-]?[0-9]+\b", "IMM", inst)

        @classmethod
        def variable(cls, inst):
            inst = re.sub(r"\.LCPI[^\s]+", "VAR", inst)
            return re.sub(r"-?\d+\b", "VAR", inst)

        @classmethod
        def address(cls, inst):
            return re.sub(r"\[.+\][!]?", "ADDRESS", inst)

        @classmethod
        def function_call(cls, inst):
            inst = re.sub(r"bl\s.+", "bl FUNC", inst)
            return re.sub(r"^b[\s]+[\w]+", "b FUNC", inst)

        @classmethod
        def bb_label(cls, inst):
            return re.sub(r"\.LBB[^\s]+", "BB", inst)

        @classmethod
        def reg_class(cls, inst):
            reg_gen = r"-?r[0-9]+!?"
            reg_pointer = r"(pc|sp|lr)!?"
            inst = re.sub(reg_gen, "%reg_gen", inst)
            inst = re.sub(reg_pointer, "%reg_pointer", inst)
            return inst


class BasicBlockProcessor(object):
    def __init__(self, arch, vocab_path):
        self.instruction_processor = InstructionProcessor(arch)
        self.token_id_manager = TokenIDManager(vocab_path)

    def normalize(self, basic_block):
        for index in range(0, len(basic_block)):
            inst = basic_block[index]
            inst = self.instruction_processor.normalize(inst)
            basic_block[index] = inst
        basic_block = " \n ".join(basic_block).replace("\t", " ")
        return " ".join(basic_block.split())

    def to_ids(self, basic_block):
        token_list = ["<s>"] + basic_block.strip().split() + ["</s>"]
        ids = []
        for token in token_list:
            ids.append(self.token_id_manager.token2id(token))
        return ids

