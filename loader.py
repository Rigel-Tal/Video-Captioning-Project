import os  
import pandas as pd 
import spacy  
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Download with: python -m spacy download en
spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class MSVDataset(Dataset):
    def __init__(self, root_dir, captions_file, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)

        # Get video_features, caption columns
        self.video_feat = self.df["video_id"]
        self.captions = self.df["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        video_id = self.video_feat[index]

        video_feat = torch.load(self.root_dir+ "/" +video_id[0:-4]+".pt")

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return video_feat, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        vid = [item[0].unsqueeze(0) for item in batch]
        vid = torch.cat(vid, dim=0)
        vid = vid.permute(1, 0, 2)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return vid, targets


def get_loader(
    root_folder,
    annotation_file,
    batch_size=32,
    num_workers=1,
    shuffle=True,
    # pin_memory=True,
):
    dataset = MSVDataset(root_folder, annotation_file)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        # pin_memory=pin_memory,
        collate_fn = MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


if __name__ == "__main__":
    loader, dataset = get_loader(
        "./DATA/train_feat", "./DATA/train_data.csv"
    )

    for idx, (vid, captions) in enumerate(loader):
        print(vid.shape)
        print(captions.shape)
