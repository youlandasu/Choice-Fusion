""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""
import os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import argparse
from config import get_args
from evaluate import evaluate_metrics, feature2dict_tensor, example2dict, multirc_f1_over_all_answers, f1_score_with_invalid, bundlefeature2dict
import json
from copy import deepcopy
import numpy as np
from collections import Counter
import difflib
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional
import pickle 
import tqdm
#from sacrebleu.metrics import BLEU
#from bert_score import BERTScorer


import numpy as np

from transformers.tokenization_utils_base import TruncationStrategy
from datasets import load_metric
from transformers.optimization import Adafactor, AdafactorSchedule
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    T5Config, 
    T5Tokenizer, 
    T5ForConditionalGeneration,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    get_constant_schedule_with_warmup,
    AdamW,
)
from utils.data_utils import normalize_ontology, get_slot_information
from data_loader_fusion import QADataset, DSTDataset, Split, processors, EXPERIMENT_DOMAINS
from value_generation_fusion import DSTGeneration

logger = logging.getLogger(__name__)
torch.cuda.empty_cache() 

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    mode: str = field(default="pretrain", metadata={"help": "Should select either \"pretrain\" or \"xcsqa-finetune\"."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    cached_data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    train_file: str = field(metadata={"help": "Should contain the data files for the task."})
    validation_file: str = field(metadata={"help": "Should contain the data files for the task."})
    test_file: str = field(metadata={"help": "Should contain the data files for the task."})
    prediction_output: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    percentage: int = field(default=100, metadata={"help": "Should contain the data files for the task."})
    except_domain: str = field(default="none", metadata={"help":"Training with all other domains: hotel, train, restaurant, attraction, taxi"})
    only_domain: str = field(default="none", metadata={"help":"Training with only one domain: hotel, train, restaurant, attraction, taxi"})
    neg_num: float = field(default=0.3, metadata={"help":"Negative samples for qa training."})
    neg_context_ratio: float = field(default=0.05, metadata={"help":"Negative context for qa training."})
    value_distribution : bool = field(default=True, metadata={"help":"Calculate prior and post value distribution"})
    knowledge_fusion: str = field(default="initDecoder", metadata={"help":"Decoder initialized based on the fused knowledge."})
    word_bow_loss: float = field(default=0.5, metadata={"help":"The weight of word bow loss"})
    history_turn: int = field(default=2, metadata={"help":"The max number of history truns of dialogue."})
    ontology_file: Optional[str] = field(default=None, metadata={"help":"Path to the ontology file."})
    description_file: Optional[str] = field(default=None, metadata={"help":"Path to the description file."})
    test_type: Optional[str] = field(default="dst", metadata={"help":"Path to the description file."})
    

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        processor = processors[data_args.task_name]()
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    
    '''
    model = T5ForConditionalGeneration.from_pretrained( 
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    '''

    model = DSTGeneration(
        model_path=model_args.model_name_or_path,
        pretrain_config=config,
        knowledge_fusion=data_args.knowledge_fusion,
        word_bow_loss=data_args.word_bow_loss,
        max_len=data_args.max_seq_length,
        cache_dir=model_args.cache_dir,
        training_mode = model_args.mode,
        )
    
    
    # Get datasets
    train_dataset = (
        QADataset(
            cached_data_dir=data_args.cached_data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
            train_file=data_args.train_file,
            validation_file=data_args.validation_file,
            test_file=data_args.test_file,
            percentage=data_args.percentage,
            neg_num=data_args.neg_num,
            neg_context_ratio=data_args.neg_context_ratio,
            only_domain=data_args.only_domain,#
            except_domain=data_args.except_domain,#
            ontology_file=data_args.ontology_file,#
            description_file=data_args.description_file,#
            history_turn = data_args.history_turn,#
        )
        if training_args.do_train
        else None
    )

    
    val_dataset = (
        QADataset(
            cached_data_dir=data_args.cached_data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.val,
            train_file=data_args.train_file,
            validation_file=data_args.validation_file,
            test_file=data_args.test_file,
            percentage=data_args.percentage,
            neg_num=data_args.neg_num,
            neg_context_ratio=data_args.neg_context_ratio,
            only_domain=data_args.only_domain,#
            except_domain=data_args.except_domain,#
            ontology_file=data_args.ontology_file,#
            description_file=data_args.description_file,#
            history_turn = data_args.history_turn,#
        )
        if training_args.do_eval
        else None
    )

    if data_args.test_type == "qa":
        test_dataset = (
            DSTDataset(
                cached_data_dir=data_args.cached_data_dir,
                tokenizer=tokenizer,
                task=data_args.task_name,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                validation_file=data_args.validation_file,
                percentage=data_args.percentage,
                neg_num=data_args.neg_num,
                neg_context_ratio=data_args.neg_context_ratio,
            )
            if training_args.do_predict
            else None
        )

    if data_args.test_type == "dst":
        domain_dataset = {} #dataset of domain features

        for domain in EXPERIMENT_DOMAINS:
            domain_dataset[domain] = (
                DSTDataset(
                    cached_data_dir=data_args.cached_data_dir,
                    tokenizer=tokenizer,
                    task=data_args.task_name,
                    max_seq_length=data_args.max_seq_length,
                    overwrite_cache=data_args.overwrite_cache,
                    test_file=data_args.test_file,
                    neg_num=data_args.neg_num,
                    neg_context_ratio=data_args.neg_context_ratio,
                    ontology_file=data_args.ontology_file,
                    description_file=data_args.description_file,
                    history_turn = data_args.history_turn,
                    domain=domain,
                )
                if training_args.do_predict
                else None
            )
    
    optimizer = Adafactor(
        model.parameters(), 
        scale_parameter=True, 
        relative_step=True, 
        warmup_init=True, 
        lr=None)
    
    scheduler = AdafactorSchedule(optimizer)
    
    optimizers = (optimizer, scheduler)
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=optimizers,
    )

    # Training
    if training_args.do_train:
        if model_args.mode == "pretrain":
            trainer.train(
                resume_from_checkpoint=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None,
                ignore_keys_for_eval=["past_key_values","decoder_hidden_states", "decoder_attentions", "cross_attentions", 
                                        "encoder_last_hidden_state", "encoder_hidden_states", "encoder_attentions"],
            )

            trainer.save_model()
            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            if trainer.is_world_process_zero():
                tokenizer.save_pretrained(training_args.output_dir)
        
    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate (Validation) ***")
        
        result = trainer.evaluate(
                ignore_keys=["past_key_values","decoder_hidden_states", "decoder_attentions", "cross_attentions", 
                                        "encoder_last_hidden_state", "encoder_hidden_states", "encoder_attentions"],
        )
        print(result)
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results for %s *****" % output_eval_file)
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

                results.update(result)

        return results

    if training_args.do_predict:
        model.to(trainer.args.device)
        model.load_state_dict(torch.load(model_args.model_name_or_path+"/pytorch_model.bin"))

        logger.info("*** Evaluate (Prediction) ***")
        trainer.compute_metrics = None  # Prediciton no need for eval.
        if data_args.test_type == "qa":
            bleu = load_metric("bleu")
            qa_exps = example2dict(test_dataset)
            qa_data = feature2dict_tensor(test_dataset)
            test_loader = DataLoader(qa_data, batch_size=training_args.per_device_eval_batch_size, shuffle=False)
            test_loader_for_examples = DataLoader(qa_exps, batch_size=training_args.per_device_eval_batch_size, shuffle=False, collate_fn=lambda x: x )
            value_all = []
            decoder_input_ids = torch.tensor(0).to(trainer.args.device) #initial bos token id as decoder input

            for batch in tqdm.tqdm(test_loader, desc="Generating qa test data"):

                eval_inputs = {"input_ids": batch["input_ids"].to(trainer.args.device)}
                with torch.no_grad():
                    pred_result = model.generate(**eval_inputs, max_length=200) ####

                value_batch = tokenizer.batch_decode(pred_result.to(trainer.args.device), skip_special_tokens=True)
                value_all.append(value_batch)
                #for item in value_batch:
                #    if item != "":
                #        print(item)
            
            predictions = []
            references = []
            for batch,value_batch in tqdm.tqdm(zip(test_loader_for_examples,value_all), desc="Calculating qa metrics in test dataset"):
                for idx, value in enumerate(value_batch):
                    value = "none" if value=="" else value # if greedy search generation is empty
                    ref = "none" if batch[idx]["output_text"]=="" else batch[idx]["output_text"]

                    predictions.append(value)
                    references.append([ref])

            print(predictions[:10])
            print(references[:10])
            
            sacrebleu = load_metric("sacrebleu")
            results = sacrebleu.compute(predictions=predictions, references=references)   
            print(f"Prediction QA BLEU:", results["score"])  
            scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            P, R, F1 = scorer.score(predictions, references)
            print(P.mean())
            print(R.mean())
            print(F1.mean())
            

        if data_args.test_type == "dst":
            ontology = normalize_ontology(json.load(open("data/mwz2.1/ontology.json", 'r')))
            slots = get_slot_information(json.load(open(data_args.ontology_file, 'r')))
            for domain in domain_dataset.keys():
                domain_exps = example2dict(domain_dataset[domain])
                domain_data = feature2dict_tensor(domain_dataset[domain])
                domain_slots = [k for k in slots if domain in k]
                test_loader = DataLoader(domain_data, batch_size=training_args.per_device_eval_batch_size, shuffle=False)
                test_loader_for_examples = DataLoader(domain_exps, batch_size=training_args.per_device_eval_batch_size, shuffle=False, collate_fn=lambda x: x )
                #Initialize counters
                slot_logger = {slot_name:[0,0,0] for slot_name in domain_slots}
                slot_logger["slot_gate"] = [0,0,0]
                value_all = []
                predictions={}
                multi_choices_collection = [] 

                
                for batch in tqdm.tqdm(test_loader, desc="Generating domain_{} test data".format(domain)):
                    eval_inputs = {"input_ids": batch["input_ids"].to(trainer.args.device)} 
                    with torch.no_grad():
                        pred_result = model.generate(**eval_inputs, max_length=200) ####

                    value_batch = tokenizer.batch_decode(pred_result.to(trainer.args.device), skip_special_tokens=True)
                    value_all.append(value_batch)
                    #for item in value_batch:
                    #    if item != "":
                    #        print(item)
                for batch,value_batch in tqdm.tqdm(zip(test_loader_for_examples,value_all), desc="Calculating domain_{} metrics in test dataset".format(domain)):
                    for idx, value in enumerate(value_batch):

                        value = "none" if value=="" else value # if greedy search generation is empty
                        dial_id = batch[idx]["ID"]
                        if dial_id not in predictions:
                            predictions[dial_id] = {}
                            predictions[dial_id]["domain"] = batch[idx]["domains"][0]
                            predictions[dial_id]["turns"] = {}

                        if batch[idx]["turn_id"] not in predictions[dial_id]["turns"]:
                            predictions[dial_id]["turns"][batch[idx]["turn_id"]] = {"turn_belief":batch[idx]["turn_belief"], "pred_belief":[]}

                        # add the active slots into the collection
                        if batch[idx]["question_type"]=="extractive" and value!="none":

                            value = difflib.get_close_matches(value, ontology[batch[idx]["slot_text"]], n=1)
                            if len(value)>0:
                                predictions[dial_id]["turns"][batch[idx]["turn_id"]]["pred_belief"].append(str(batch[idx]["slot_text"])+'-'+str(value[0]))
                                value = value[0]
                            else:
                                value="none"
                        # analyze none acc:
                        if batch[idx]["question_type"]=="extractive":
                            if value=="none" and batch[idx]["value_text"]=="none":
                                slot_logger["slot_gate"][1]+=1 # hit
                            if value!="none" and batch[idx]["value_text"]!="none":
                                slot_logger["slot_gate"][1]+=1 # hit
                            slot_logger["slot_gate"][0]+=1 # total

                        # collect multi-choice answers
                        if batch[idx]["question_type"]=="multi-choice":
                            value = difflib.get_close_matches(value, ontology[batch[idx]["slot_text"]], n=1)
                            if len(value)>0 and value!="":
                                value = value[0]
                            else:
                                value="none"
                            multi_choices_collection.append({"dial_id":batch[idx]["ID"], "turn_id":batch[idx]["turn_id"], "slot_text":batch[idx]["slot_text"], "value":value})
                        # analyze slot acc:
                        if (batch[idx]["value_text"]!="none"):
                            if str(value)==str(batch[idx]["value_text"]):
                                slot_logger[str(batch[idx]["slot_text"])][1]+=1 # hit
                            slot_logger[str(batch[idx]["slot_text"])][0]+=1 # total

                for example in multi_choices_collection:
                    dial_id = example["dial_id"]
                    turn_id = example["turn_id"]
                    extractive_value = ""
                    # check active slot
                    for kv in predictions[dial_id]["turns"][turn_id]["pred_belief"]:
                        if example["slot_text"] in kv:
                            extractive_value = kv
                    # if slot is not active
                    if extractive_value=="":
                        continue
                    # replace extrative slot with multi-choice
                    predictions[dial_id]["turns"][turn_id]["pred_belief"].remove(extractive_value)
                    predictions[dial_id]["turns"][turn_id]["pred_belief"].append(str(example["slot_text"])+'-'+str(example["value"]))


                for slot_log in slot_logger.values():
                    if slot_log[0] > 0:
                        slot_log[2] = slot_log[1]/slot_log[0]
                    else:
                        slot_log[2] = 0

                # save results to prediction_output
                prediction_dir = os.path.join(data_args.prediction_output, domain)
                os.makedirs(prediction_dir, exist_ok=True)
                with open(os.path.join(prediction_dir, f"slot_acc.json"), 'w') as f:
                    json.dump(slot_logger,f, indent=4)

                with open(os.path.join(prediction_dir, f"prediction.json"), 'w') as f:
                    try:
                        print(predictions['SNG0797.json'])
                    except:
                        pass
                    json.dump(predictions,f, indent=4)

                joint_acc_score, F1_score, turn_acc_score= evaluate_metrics(predictions, domain_slots)

                evaluation_metrics = {"Joint Acc":joint_acc_score, "Turn Acc":turn_acc_score, "Joint F1":F1_score}
                print(f"Prediction {domain}:",evaluation_metrics)

                with open(os.path.join(prediction_dir, f"result.json"), 'w') as f:
                    json.dump(evaluation_metrics,f, indent=4)

        
    return predictions


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
