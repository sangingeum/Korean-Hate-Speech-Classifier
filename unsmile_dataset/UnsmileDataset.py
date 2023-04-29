import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label

def load_unsmile_data(train=True, return_label_features=False):
    if train:
        split_name = "train"
    else:
        split_name = "valid"
    datasets = load_dataset('smilegate-ai/kor_unsmile', split=split_name)
    label_features = ['여성/가족', '남성', '성소수자', '인종/국적', '연령', '지역', '종교', '기타 혐오', '악플/욕설', 'clean', '개인지칭']
    texts = datasets['문장']
    temp = [np.reshape(datasets[feature], (-1, 1)) for feature in label_features]
    labels = np.hstack(temp)
    labels = torch.reshape(torch.tensor(labels, dtype=torch.float32, requires_grad=False), (-1, 11))
    torch_dataset = TextDataset(texts, labels)
    if return_label_features:
        return torch_dataset, label_features
    return torch_dataset
