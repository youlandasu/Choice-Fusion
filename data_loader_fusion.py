""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension """


import csv
import glob
import json
import random
import logging
import os
from dataclasses import dataclass,field
from functools import partial
from collections import OrderedDict
from enum import Enum
from typing import List, Optional
import difflib
import torch
from torch.utils.data.dataset import Dataset

import tqdm

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
from transformers.tokenization_utils_base import TruncationStrategy
from utils.data_utils import get_slot_information, normalize_ontology, fix_general_label_error

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice
    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    intput_text: str
    output_text: str
    choices: str
    ID: Optional[str] = field(default=None)
    domains: Optional[str]= field(default=None)
    turn_id: Optional[str]= field(default=None)
    dialog_history: Optional[str]= field(default=None)
    turn_belief: Optional[List[str]]= field(default=None)
    slot_text: Optional[str]= field(default=None)
    value_text: Optional[str]= field(default=None)
    question_type: Optional[str]= field(default=None)


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    decoder_input_ids: List[List[int]]
    value_candidate_embedding: List[List[int]]
    labels: List[List[int]]
    attention_mask: Optional[List[List[int]]]= field(default=None)
    decoder_attention_mask: Optional[List[List[int]]]= field(default=None)
    value_candidate_mask: Optional[List[List[int]]]= field(default=None)


@dataclass(frozen=True)
class BundledFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    intput_text: str
    output_text: str
    choices: str
    input_ids: List[List[int]]
    decoder_input_ids: List[List[int]]
    value_candidate_embedding: List[List[int]]
    labels: List[List[int]]
    ID: Optional[str] = field(default=None)
    domains: Optional[str]= field(default=None)
    turn_id: Optional[str]= field(default=None)
    dialog_history: Optional[str]= field(default=None)
    turn_belief: Optional[List[str]]= field(default=None)
    slot_text: Optional[str]= field(default=None)
    value_text: Optional[str]= field(default=None)
    question_type: Optional[str]= field(default=None)
    attention_mask: Optional[List[List[int]]]= field(default=None)
    decoder_attention_mask: Optional[List[List[int]]]= field(default=None)
    value_candidate_mask: Optional[List[List[int]]]= field(default=None)

class Split(Enum):
    train = "train"
    val = "val"
    test = "test"
    domain = "domain"



class QADataset(Dataset):

    features: List[InputFeatures]

    def __init__(
        self,
        cached_data_dir: str,
        tokenizer: PreTrainedTokenizer,
        task: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Split = Split.train,
        history_turn=None,
        neg_num=None,
        neg_context_ratio=None,
        only_domain="none",
        except_domain="none",
        train_file=None,
        validation_file=None,
        test_file=None,
        ontology_file=None,
        description_file=None,
        percentage=None,
        domain = None,
    ):
        processor = processors[task]()
        if mode == Split.train:
            PREFIX = train_file.split("/")[-1].replace(".json", "")
        elif mode == Split.val:
            PREFIX = validation_file.split("/")[-1].replace(".json", "")
        elif mode == Split.test:
            PREFIX = test_file.split("/")[-1].replace(".json", "")
        else: 
            PREFIX = "domain_" + domain
        cached_features_file = os.path.join(
            cached_data_dir,
            "cached_{}_{}_{}".format(
                PREFIX,
                tokenizer.__class__.__name__,
                str(max_seq_length),
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {cached_data_dir}")
                #answer_list = processor.get_answers()
                if ontology_file is not None:
                    with open(ontology_file, "r", encoding="utf-8") as fin:
                        ontology = json.load(fin)
                if description_file is not None:
                    with open(description_file, "r", encoding="utf-8") as fd:
                        description = json.load(fd)
                all_slots = None if not ontology_file else get_slot_information(ontology)
                slot_config = {
                    "all_slots": all_slots, 
                    "only_domain": only_domain,
                    "except_domain": except_domain,
                    } if all_slots else None
                neg_sampling = {
                    "neg_num": neg_num,
                    "neg_context_ratio": neg_context_ratio,
                }

                if mode == Split.val:
                    examples = processor.get_val_examples(validation_file, tokenizer, neg_sampling, slot_config, percentage=percentage)
                elif mode == Split.test:
                    examples = processor.get_test_examples(test_file, tokenizer, neg_sampling, slot_config, description, history_turn)
                elif mode == Split.train:
                    examples = processor.get_train_examples(train_file, tokenizer, neg_sampling, slot_config, percentage=percentage)
                else:
                    slot_config["only_domain"] = domain
                    #print(slot_config["all_slots"])
                    #print(slot_config["only_domain"])
                    #print(slot_config["except_domain"])
                    examples = processor.get_test_examples(test_file, tokenizer, neg_sampling, slot_config, description, history_turn)
                logger.info("Num examples: %s", len(examples))
                self.features = convert_examples_to_features(
                    examples,
                    #answer_list,
                    max_seq_length,
                    tokenizer,
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

 
class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, file_path):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_val_examples(self, file_path):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, file_path):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    #def get_answers(self):
    #    """Gets the list of labels for this data set."""
    #    raise NotImplementedError()


class QAProcessor(DataProcessor):
    """Processor for the XCSQA data set."""

    def get_train_examples(self, file_path, tokenizer, neg_sampling, slot_config=None, description=None, history_turn=None, percentage=100):
        """See base class."""
        logger.info("LOOKING train AT {}".format(file_path))
        return self._create_examples(self._read_json(file_path), "train", tokenizer, neg_sampling, slot_config, description, percentage=percentage)

    def get_val_examples(self, file_path, tokenizer, neg_sampling, slot_config=None, description=None, history_turn=None, percentage=100):
        """See base class."""
        logger.info("LOOKING val AT {} ".format(file_path))
        return self._create_examples(self._read_json(file_path), "val", tokenizer, neg_sampling, slot_config, description, percentage=percentage)

    def get_test_examples(self, file_path, tokenizer, neg_sampling, slot_config=None, description=None, history_turn=None):
        """See base class."""
        logger.info("LOOKING test AT {} ".format(file_path))
        return self._create_dst_examples(self._read_json(file_path), "test", tokenizer, neg_sampling, slot_config, description, history_turn)

    #def get_labels(self, num_choices):
    #    """See base class."""
    #    return [str(i) for i in range(1,num_choices+1)]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = json.load(fin)
            return lines

    def _create_examples(self, lines, type, tokenizer, neg_sampling, slot_config=None, description=None, percentage=None):
        """Creates examples for the training and validation sets."""

        examples = []  
        # we deleted example which has more than or less than four choices
        if percentage is not None:
            random.seed(42)
            random.shuffle(lines)
            split_point = int(len(lines) * percentage / 100)
            lines = lines[:split_point]
        print(len(lines))
        for line in tqdm.tqdm(lines, desc="read data"):
            choice_token = " <extra_id_0> " 
            context = line["context"].strip()

            # examples = [{"context":"text", "qas":{"question":"..", "answer":"..", "negative_questions":[]}, }]
            #try:
            #    qa = random.choice(line["qas"]) # choose one qa for computation efficiency
            #except:
            #    continue
            for qa in line["qas"]:
                question = qa["question"].strip()
                if len(qa["choice"])>0:
                    choices = (choice_token + choice_token.join(qa["choice"])).lower()
                    input_text = f"multi-choice question: {question} context: {context}".lower() #choices: {choices} 
                    output_text = (qa["answer"]).lower() #+ f" {tokenizer.eos_token}"
                    examples.append(
                        InputExample(
                            intput_text=input_text,
                            output_text=output_text,
                            choices=(choice_token + " ".join(qa["choice"])).lower(),
                        )
                    )

                else:
                    input_text = f"extractive question: {question} context: {context}".lower()
                    output_text = (qa["answer"] ).lower() #+ f" {tokenizer.eos_token}"
                    examples.append(
                        InputExample(
                        intput_text=input_text,
                        output_text=output_text,
                        choices=choice_token,
                        )
                    )
                if random.random()<neg_sampling["neg_num"]:
                # for i in range(args["neg_num"]):
                    negative_context = ""
                    if len(qa["char_spans"])>0:
                        for i in range(qa["char_spans"][0],0, -1):
                            if line["context"][i]==".":
                                negative_context = line["context"][:i+1]
                                # print(qa["char_spans"][0], i)
                                break

                    if (negative_context!="") and (random.random()<neg_sampling["neg_context_ratio"]):
                        # use negative context
                        question = qa["question"].strip()
                        input_text = f"extractive question: {question} context: {negative_context}".lower()
                    else:
                        # use negative question
                        question = qa["negative_questions"][0].strip()
                        input_text = f"extractive question: {question} context: {context}".lower()


                        # print(input_text)
                        # print(qa["answer"])

                    output_text = "none" #+ f" {tokenizer.eos_token}"
                    examples.append(
                        InputExample(
                        intput_text=input_text,
                        output_text=output_text,
                        choices=choice_token,
                        )
                    )

        if type == "train":
            assert len(examples) > 1
            #assert examples[0].label is not None
        for f in examples[:2]:
            logger.info("*** Example ***")
            logger.info("example: %s" % f)
        logger.info("len examples: %s}", str(len(examples)))  

        return examples

    def _create_dst_examples(self, lines, type, tokenizer, neg_num, slot_config=None, description=None, history_turn=None, percentage=None):
        """Creates examples for the training and validation sets."""

        examples = []  
        domain_counter = {}
        # we deleted example which has more than or less than four choices
        if percentage is not None:
            random.seed(42)
            random.shuffle(lines)
            split_point = int(len(lines) * percentage / 100)
            lines = lines[:split_point]
        print(len(lines))
        for line in tqdm.tqdm(lines, desc="read dials data"):
            choice_token = " <extra_id_0> " 
            dialog_history = []
            # Counting domains
            for domain in line["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if slot_config["only_domain"] != "none" and slot_config["only_domain"] not in line["domains"]:
                continue
            if (slot_config["except_domain"] != "none" and type == "test" and slot_config["except_domain"] not in line["domains"]) or \
            (slot_config["except_domain"] != "none" and type != "test" and [slot_config["except_domain"]] == line["domains"]):
                continue
            
            # Reading data
            for idx, turn in enumerate(line["turns"]):
                turn_id = idx

                # accumulate dialogue utterances
                dialog_history.append(" system: " + turn["system"] + " user: " + turn["user"])
                if len(dialog_history) > history_turn:
                    dialog_history.pop(0)

                slot_values = fix_general_label_error(turn["state"]["slot_values"], slot_config["all_slots"])

                # input: dialogue history + slot
                # output: value

                # Generate domain-dependent slot list
                slot_temp = slot_values.keys()
                if type == "train" or type == "dev":
                    if slot_config["except_domain"] != "none":
                        slot_temp = [k for k in slot_config["all_slots"] if slot_config["except_domain"] not in k] #slot_values.keys()
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if slot_config["except_domain"] not in k])
                    elif slot_config["only_domain"] != "none":
                        slot_temp = [k for k in slot_config["all_slots"] if slot_config["only_domain"] in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if slot_config["only_domain"] in k])
                else:
                    if slot_config["except_domain"] != "none":
                        slot_temp = [k for k in slot_config["all_slots"] if slot_config["except_domain"] in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if slot_config["except_domain"] in k])
                    elif slot_config["only_domain"] != "none":
                        slot_temp = [k for k in slot_config["all_slots"] if slot_config["only_domain"] in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if slot_config["only_domain"] in k])
                turn_belief_list = []
                for k,v in slot_values.items():
                    if v!="none":
                        turn_belief_list.append(str(k)+'-'+str(v))

                for slot in slot_temp:
                    # skip unrelevant slots for out of domain setting
                    if slot_config["except_domain"] != "none" and type !="test":
                        if slot.split("-")[0] not in line["domains"]:
                            continue

                    slot_lang = description[slot]["question"]
                    slot_text = slot
                    value_text = slot_values.get(slot, 'none').strip()
                    #if value_text=="none":
                    #    continue

                    concat_history = "".join(dialog_history)
                    input_text = f"extractive question: {slot_lang} context: {concat_history}".lower()
                    output_text = (value_text).lower() #+ f" {tokenizer.eos_token}"
                    examples.append(
                        InputExample(
                        ID=line["dial_id"],
                        domains=line["domains"],
                        turn_id=turn_id,
                        dialog_history=dialog_history,
                        turn_belief=turn_belief_list,
                        intput_text=input_text,
                        output_text=output_text,
                        slot_text=slot_text,
                        value_text=value_text,
                        question_type="extractive",
                        choices=choice_token,
                        )
                    )

                    if len(description[slot]["values"])>0 and value_text!="none":
                        choices = (choice_token + choice_token.join(description[slot]["values"])).lower()
                        input_text = f"multi-choice question: {slot_lang} context: {concat_history}".lower() #choices: {choices} 
                        output_text = (value_text).lower() #+ f" {tokenizer.eos_token}"
                        examples.append(
                            InputExample(
                                ID=line["dial_id"],
                                domains=line["domains"],
                                turn_id=turn_id,
                                dialog_history=dialog_history,
                                turn_belief=turn_belief_list,
                                intput_text=input_text,
                                output_text=output_text,
                                slot_text=slot_text,
                                value_text=value_text,
                                question_type="multi-choice",
                                choices=(choice_token + " ".join(description[slot]["values"])).lower(),
                            )
                        )     

        if type == "train":
            assert len(examples) > 1
            #assert examples[0].label is not None
        for f in examples[:2]:
            logger.info("*** Example ***")
            logger.info("example: %s" % f)
        logger.info("len examples: {}".format(str(len(examples))))

        return examples

def convert_examples_to_features(
    examples: List[InputExample],
    #label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    #label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    num_all_statements = 0
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), total=len(examples), desc="convert examples to features"):
        # if ex_index % 100 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        inputs = tokenizer(
                example.intput_text,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                truncation_strategy = TruncationStrategy.ONLY_FIRST,
                return_overflowing_tokens=False,
            )
        outputs = tokenizer(
                example.output_text,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                truncation_strategy = TruncationStrategy.ONLY_FIRST,
                return_overflowing_tokens=False,
            )
        choices = tokenizer(
                example.choices,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                truncation_strategy = TruncationStrategy.ONLY_FIRST,
                return_overflowing_tokens=False,
            )
        '''
        try:
            eoa = outputs["input_ids"].index(1)
            matcher = difflib.SequenceMatcher(None,outputs["input_ids"],choices["input_ids"])
            find_value = matcher.find_longest_match(0, eoa, 0, len(choices["input_ids"]))
            golden_value_index = list(range(find_value.b, find_value.b+find_value.size))
            golden_value_index += [0] * (max_length - len(golden_value_index))
        except:
            golden_value_index = [0] * max_length
        '''
        attention_mask = (
            inputs["attention_mask"] if "attention_mask" in inputs else None
        )
        decoder_attention_mask = (
            outputs["attention_mask"] if "attention_mask" in outputs else None
        )
        value_candidate_mask = (
            choices["attention_mask"] if "attention_mask" in choices else None
        )
        
        num_all_statements += 1
        # num_exceed_max += sum([int(x["input_ids"][0][-1])!=int(x["input_ids"][0][-2]) for x in choices_inputs])
        features.append(
            InputFeatures(
                input_ids=inputs["input_ids"] + choices["input_ids"],
                decoder_input_ids=outputs["input_ids"],
                value_candidate_embedding=choices["input_ids"],
                attention_mask=attention_mask + value_candidate_mask,
                decoder_attention_mask=decoder_attention_mask,
                value_candidate_mask=value_candidate_mask,
                labels=outputs["input_ids"],
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)
    logger.info("num_all_statements: %d" % num_all_statements)
    return features

class DSTDataset(Dataset):

    bundledfeatures: List[BundledFeatures]

    def __init__(
        self,
        cached_data_dir: str,
        tokenizer: PreTrainedTokenizer,
        task: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Split = Split.train,
        history_turn=None,
        neg_num=None,
        neg_context_ratio=None,
        only_domain="none",
        except_domain="none",
        train_file=None,
        validation_file=None,
        test_file=None,
        ontology_file=None,
        description_file=None,
        percentage=None,
        domain = None,
    ):
        processor = processors[task]()
        PREFIX = "bundles_domain_" + domain
        #PREFIX = "QA_test" #test QA data
        cached_example_file = os.path.join(
            cached_data_dir,
            "cached_{}_{}_{}".format(
                PREFIX,
                tokenizer.__class__.__name__,
                str(max_seq_length),
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_example_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_example_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_example_file}")
                self.bundledfeatures = torch.load(cached_example_file)
            else:
                logger.info(f"Creating features from dataset file at {cached_data_dir}")
                #answer_list = processor.get_answers()
                if ontology_file is not None:
                    with open(ontology_file, "r", encoding="utf-8") as fin:
                        ontology = json.load(fin)
                if description_file is not None:
                    with open(description_file, "r", encoding="utf-8") as fd:
                        description = json.load(fd)
                all_slots = None if not ontology_file else get_slot_information(ontology)
                slot_config = {
                    "all_slots": all_slots, 
                    "only_domain": only_domain,
                    "except_domain": except_domain,
                    } if all_slots else None
                neg_sampling = {
                    "neg_num": neg_num,
                    "neg_context_ratio": neg_context_ratio,
                }


                slot_config["only_domain"] = domain
                examples = processor.get_test_examples(test_file, tokenizer, neg_sampling, slot_config, description, history_turn)
                #examples = processor.get_val_examples(validation_file, tokenizer, neg_sampling, slot_config, percentage=percentage) #test QA data
                logger.info("Num examples: %s", len(examples))
                features = convert_examples_to_features(
                    examples,
                    max_seq_length,
                    tokenizer,
                )
                self.bundledfeatures = []
                for exp, fes in zip(examples, features):
                    self.bundledfeatures.append(
                        BundledFeatures(
                            intput_text=exp.intput_text,
                            output_text=exp.output_text,
                            choices=exp.choices,
                            ID=exp.ID,
                            domains=exp.domains,
                            turn_id=exp.turn_id,
                            dialog_history=exp.dialog_history,
                            turn_belief=exp.turn_belief,
                            slot_text=exp.slot_text,
                            value_text=exp.value_text,
                            question_type=exp.question_type,
                            input_ids=fes.input_ids + fes.value_candidate_embedding,
                            decoder_input_ids=fes.decoder_input_ids,
                            value_candidate_embedding=fes.value_candidate_embedding,
                            labels=fes.labels,
                            attention_mask=fes.attention_mask + fes.value_candidate_mask,
                            decoder_attention_mask=fes.decoder_attention_mask,
                            value_candidate_mask=fes.value_candidate_mask,
                        )
                    )
                logger.info("Saving examples into cached file %s", cached_example_file)
                torch.save(self.bundledfeatures, cached_example_file)

    def __len__(self):
        return len(self.bundledfeatures )

    def __getitem__(self, i) -> InputExample:
        return self.bundledfeatures[i]

processors = {"vadst": QAProcessor}