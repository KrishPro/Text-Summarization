"""
Written by KrishPro @ KP
"""

from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

from model import Model

# !pip install -q tf-estimator-nightly==2.8.0.dev2021122109 earthengine-api==0.1.238 folium==0.2.1
# !pip install -q torchtext==0.11.0 torchaudio==0.10.0 torchvision==0.11.1 torch==1.10
# !pip install -q cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/colab/1.10/torch_xla-1.10-cp37-cp37m-linux_x86_64.whl
# !pip install -q transformers pytorch_lightning datasets pyngrok

# !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.tgz
# !tar zxvf ngrok-stable-linux-amd64.tgz
# !./ngrok authtoken 1y2MQMr0xLh05Dvbb0dABiNQpAY_3bqEfwwEtM7duDaqwrN93

# drive.mount('/content/drive')

# pwd = "/content/drive/MyDrive/Models/text-summarization/bart"

# def start_tensorboard(log_dir: str):
#     tb = program.TensorBoard()
#     tb.configure(argv=[None, '--logdir', log_dir])
#     url = tb.launch()
#     print(f"Tensorflow listening on {url}")
#     port = int(url.split(":")[-1][:-1])
#     print(ngrok.connect(port))

class TrainModel(Model):
    def __init__(self, learning_rate: float, ultimate_batch_size: int, epochs: int, label_smoothing=0.0, ignore_index=None):
        super(TrainModel, self).__init__()
        self.learning_rate = learning_rate

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=ignore_index)

        steps_per_iter = 215344 # len(train_dataset) + len(val_dataset)
        self.num_training_steps = (steps_per_iter // ultimate_batch_size) * epochs
        self.num_warmup_steps = int(self.num_training_steps * 0.1)


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.num_warmup_steps, self.num_training_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
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
        self.log("lr", self.lr_schedulers().get_last_lr(), prog_bar=True)
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

