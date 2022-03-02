"""
Written by KrishPro @ KP
"""

from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

from model import Model


class CustomCrossEntropy(nn.Module):
    """
    CustomCrossEntropy Loss

    Due to torch_xla not working on pytorch=1.10, I had to down-grade to 1.9.
    But there is feat. added to CrossEntropy in 1.10, Called label_smothing

    I needed it, So i coded up a the feat. myself.
    Now, I'm neither using 1.10's CrossEntropy nor 1.9's CrossEntropy. I'm using mine
    """
    def __init__(self, label_smoothing=0.0, ignore_index=None):
        super(CustomCrossEntropy, self).__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
    
    def smooth_label(self, one_hot: torch.Tensor):
        one_hot = (1 - self.label_smoothing) * one_hot + self.label_smoothing / one_hot.size(1)
        return one_hot
    
    def zero_pad_prob(self, smoothed_labels: torch.Tensor):
        K = smoothed_labels.size(1)
        smoothed_labels = smoothed_labels + (smoothed_labels[:, self.ignore_index] / (K - 1)).view(smoothed_labels.size(0), 1)
        smoothed_labels[:, self.ignore_index] = 0.0
        return smoothed_labels
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        #inputs.shape: (num_samples, num_classes)
        #targets.shape: (num_samples)
        num_classes = inputs.size(1)
        
        one_hot = F.one_hot(targets, num_classes=num_classes)
        smoothed_labels = self.smooth_label(one_hot)
        smoothed_labels = self.zero_pad_prob(smoothed_labels) if self.ignore_index else smoothed_labels
        loss = F.cross_entropy(inputs, smoothed_labels, reduction="none")
        loss = loss[targets != self.ignore_index] if self.ignore_index else loss
        return torch.mean(loss)

class TrainModel(Model):
    def __init__(self, learning_rate: float, ultimate_batch_size: int, epochs: int, label_smoothing=0.0, ignore_index=None):
        super(TrainModel, self).__init__()
        self.learning_rate = learning_rate

        self.criterion = CustomCrossEntropy(label_smoothing, ignore_index)

        steps_per_iter = 215344 # len(train_dataset) + len(val_dataset)
        self.num_training_steps = (steps_per_iter // ultimate_batch_size) * epochs
        self.num_warmup_steps = int(self.num_training_steps * 0.1)


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.num_warmup_steps, self.num_training_steps)

        return [optimizer], [scheduler]

    @staticmethod
    @torch.no_grad()
    def calculate_accuracy(logits: torch.Tensor, target: torch.Tensor):
        # logits.shape: (batch_size, seq_len, vocab_size)
        # decoder_ids.shape: (batch_size, seq_len)
        predictions: torch.Tensor = F.softmax(logits, dim=2).argmax(axis=2)
        # predictions.shape: (batch_size, seq_len)
        accuracy = (target.view(-1) == predictions.view(-1)).sum() / (predictions.size(0) * predictions.size(1))
        return accuracy

    def forward_step(self, batch: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]):
        _, (decoder_ids, _) = batch
        # decoder_ids.shape: (batch_size, seq_len)

        logits = self.forward(batch)
        # logits.shape: (batch_size, seq_len, vocab_size)
        loss: torch.Tensor = self.criterion(logits.reshape(logits.size(0) * logits.size(1), logits.size(2)), decoder_ids.reshape(decoder_ids.size(0) * decoder_ids.size(1)))
        accu = self.calculate_accuracy(logits, decoder_ids)
        return loss.item(), accu.item()

    def training_step(self, batch: tuple, batch_idx: int):
        loss, accu = self.forward_step(batch)
        self.log("loss", loss)
        self.log("accu", accu, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        loss, accu = self.forward_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accu", accu, prog_bar=True)
        return loss
    
    def test_step(self, batch: tuple, batch_idx: int):
        loss, accu = self.forward_step(batch)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accu", accu, prog_bar=True)
        return loss

