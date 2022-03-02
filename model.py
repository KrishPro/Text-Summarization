"""
Written by KrishPro @ KP
"""

import torch
from data import Dataset
from pytorch_lightning import LightningModule
from transformers.models.bart.modeling_bart import BartForConditionalGeneration


class Model(LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    @staticmethod
    def shift_right(tensor: torch.Tensor, start_token: int):
        shifted = torch.zeros_like(tensor)

        shifted[:, 1:] = tensor[:, :-1].clone()
        shifted[:, 0] = start_token
        return shifted
        
    def forward(self, batch: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]):
        (input_ids, attention_mask), (decoder_ids, decoder_attention_mask) = batch
        # decoder_ids.shape: (batch_size, seq_len)
        decoder_inps = self.shift_right(decoder_ids, Dataset.tokenizer.bos_token_id)
        
        decoder_attention_mask = self.shift_right(decoder_attention_mask, 1)
        logits: torch.Tensor = self.model(input_ids, attention_mask, decoder_inps, decoder_attention_mask).logits
        # logits.shape: (batch_size, seq_len, vocab_size)
        # logits obviously means no activation
        return logits