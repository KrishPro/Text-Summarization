"""
Written by KrishPro @ KP
"""


from pytorch_lightning import LightningDataModule
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast
from datasets import load_dataset
import torch.utils.data as data
import pandas as pd
import os


class Dataset(data.Dataset):
    tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

    def __init__(self, split: str = "train"):
     
        self.data = pd.read_csv(f"Data/{split}.csv").drop("id", axis=1)
        self.data = self.data.to_numpy()

    def __getitem__(self, idx):
        document, summary = self.data[idx]
        input_ids, attention_mask =  tuple(self.tokenizer(document, padding="max_length", truncation=True, return_tensors="pt").values())
        decoder_input_ids, decoder_attention_mask =  tuple(self.tokenizer(summary, padding="max_length", truncation=True, return_tensors="pt").values())
        return (input_ids.squeeze(0), attention_mask.squeeze(0)), (decoder_input_ids.squeeze(0), decoder_attention_mask.squeeze(0))

    def __len__(self):
        return len(self.data)
        
class DataModule(LightningDataModule):
    def __init__(self, batch_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

    def prepare_data(self) -> None:
        if not os.path.exists("Data"):
            os.mkdir("Data")
            datasets = load_dataset("xsum", name="bart-base")

            for split in ("train", "test", "validation"):
                datasets[split].to_csv(f"Data/{split}.csv", index=False)
                pd.read_csv(f"Data/{split}.csv").dropna().to_csv(f"Data/{split}.csv", index=False)


    def setup(self, stage: str = None) -> None:
        if (stage == "fit" or stage == None) and ((not self.train_dataset) or (not self.val_dataset)):
            self.train_dataset = Dataset(split="train")
            self.val_dataset = Dataset(split="validation")
       
        if (stage == "test" or stage == None) and (not self.test_dataset):
            self.test_dataset = Dataset(split="test")

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    datamodule = DataModule(64)
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    (input_ids, attention_mask), (decoder_input_ids, decoder_attention_mask) = next(iter(train_dataloader))
    
    assert input_ids.shape == attention_mask.shape == decoder_input_ids.shape == decoder_attention_mask.shape
    assert input_ids.dtype == attention_mask.dtype == decoder_input_ids.dtype == decoder_attention_mask.dtype

    print(input_ids.shape)
    