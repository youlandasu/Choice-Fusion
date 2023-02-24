'''
    Adding  value appreciation to T5
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from logging import raiseExceptions

import os
import numpy as np
import copy
import math
import random
from typing import Optional, Tuple
from dataclasses import dataclass, field
from itertools import chain

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    T5Model,
    T5EncoderModel,
    )
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.modeling_outputs import Seq2SeqLMOutput,BaseModelOutput
from torch.optim import Optimizer
from utils.model_utils import Projection, LinearELU, SequenceMask

import torch.optim as optimizer_module

from transformers import logging
logging.set_verbosity_error()

SIM_DIM = 64
BOW_DIM = 1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class T5Pretrained(PreTrainedModel):
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization

        if isinstance(module, (T5Model, T5ForConditionalGeneration, T5EncoderModel)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)

class DSTGeneration(T5Pretrained):
    def __init__(self, model_path, pretrain_config, knowledge_fusion, word_bow_loss, max_len, cache_dir, training_mode="pretrain"):
        
        super(DSTGeneration, self).__init__(pretrain_config) 
        self.config=pretrain_config
        self.model = T5ForConditionalGeneration.from_pretrained(
                    model_path,
                    from_tf=bool(".ckpt" in model_path),
                    config=pretrain_config,
                    cache_dir=cache_dir,
                    )

        self.fact_layer = Projection(self.config.d_model, SIM_DIM)
        self.prior_layer = Projection(self.config.d_model, SIM_DIM)
        self.post_layer = Projection(self.config.d_model*2, SIM_DIM)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        self.training_mode = training_mode
        self.shared = nn.Embedding(self.config.vocab_size, self.config.d_model)
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.model.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.model.encoder.block))
        self.model.encoder.parallelize(self.device_map)
        self.model.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.model.decoder.first_device)
        self.model_parallel = True


    def deparallelize(self):
        self.model.encoder.deparallelize()
        self.model.decoder.deparallelize()
        self.model.encoder = self.model.encoder.to("cpu")
        self.model.decoder = self.model.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()
    
    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1, device=self.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = (row_vector < matrix).to(dtype=dtype, device=self.device)

        return mask
    
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.model.config.decoder_start_token_id
        pad_token_id = self.model.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids

    def _safe_log(self,y):
        return torch.log(torch.clamp(y, 1e-9))

    def forward(
        self,
        input_ids=None,
        decoder_input_ids=None,
        value_candidate_embedding=None,
        attention_mask=None,
        decoder_attention_mask=None,
        value_candidate_mask=None,
        labels=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        max_candidate_num = value_candidate_embedding.size(1) if value_candidate_embedding is not None else None
        
        if encoder_outputs is None:
            value_candidate_outputs = self.model.encoder(
                                        input_ids=value_candidate_embedding,
                                        attention_mask=value_candidate_mask,
                                        inputs_embeds=inputs_embeds,
                                        head_mask=head_mask,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict,
                                        )
            value_candidate_states = value_candidate_outputs[0]
            fact_projection = self.fact_layer(value_candidate_states)
        
        # hidden_states of encoder outputs
            encoder_outputs = self.model.encoder(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                return_dict=return_dict,
                                )

            encoder_states = encoder_outputs[0][:,0,:]
            # hidden_states of golden decoder outputs in training
            golden_encoder_outputs = self.model.encoder(
                                input_ids=decoder_input_ids,
                                attention_mask=decoder_attention_mask,
                                return_dict=return_dict,
                                )
            golden_encoder_states = golden_encoder_outputs[0][:,0,:]
            
            # prior value distribution 
            prior_projection = self.prior_layer(encoder_states)
            prior_projection = prior_projection.unsqueeze(1).repeat(1, max_candidate_num, 1)
            prior_scores = torch.sum(prior_projection * fact_projection, -1)
            fact_seq_mask = ~value_candidate_mask #.float() #[batch,value_seq_len]
            unk_mask = self.sequence_mask(torch.ones(value_candidate_states.size(0),dtype=torch.float32, device=self.device), maxlen=max_candidate_num, dtype=torch.float32) #[batch,value_seq_len]
            fact_mask = fact_seq_mask.masked_fill_(fact_seq_mask==1, -1e-10) + unk_mask.masked_fill_(unk_mask==1, -1e-10)
            prior_scores += fact_mask
            prior_distribution = F.softmax(prior_scores, dim=1)
            
            # post value distribution
            post_inputs = torch.cat((golden_encoder_states,encoder_states), -1)
            
            post_projection = self.post_layer(post_inputs)
            post_projection = post_projection.unsqueeze(1).repeat(1, max_candidate_num, 1)
            post_scores = torch.sum(post_projection * fact_projection, -1)
            post_scores += fact_mask
            post_distribution = F.softmax(post_scores, dim=1)
        
            # calc KLD and bag-of-word loss
            post_prior_ratio = torch.div(post_distribution, torch.clamp(prior_distribution, 1e-9, 1.0))
            kld_loss = post_distribution * self._safe_log(post_prior_ratio) 
            kld_loss = torch.mean(kld_loss, -1)
            kld_loss = torch.sum(kld_loss) / value_candidate_states.size(0)

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        
        encoder_hidden_states = encoder_outputs[0] 

        if self.model_parallel:
            torch.cuda.set_device(self.model.decoder.first_device)

        if labels is not None:# and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.model.decoder.first_device)
            hidden_states = hidden_states.to(self.model.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.model.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.model.decoder.first_device)

        decoder_outputs = self.model.decoder(input_ids=decoder_input_ids,
                            attention_mask=decoder_attention_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=attention_mask,
                            inputs_embeds=decoder_inputs_embeds,
                            past_key_values=past_key_values,
                            head_mask=decoder_head_mask,
                            cross_attn_head_mask=cross_attn_head_mask,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                            )
        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.model.encoder.first_device)
            self.lm_head = self.lm_head.to(self.model.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(sequence_output)

        loss = None 
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = kld_loss + loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))


        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels):
        return self._shift_right(labels)
