import torch

from Model import *
from unsmile_dataset.UnsmileDataset import *
from utils import *
from torch.utils.data import Dataset
import sys
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


if __name__ == '__main__':

    # Load model
    device = get_device_name_agnostic()
    model = ClassificationModel(input_size=768, hidden_size=1024, num_layers=1,
                                      mlp_units=[256, 128, 64, 11], bidirectional=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)

    args = sys.argv
    save_path = "model.pt"
    # Set save path
    if len(args) >= 2:
        save_path = args[1]
    # Resume from a checkpoint if given
    if len(args) >= 3:
        load_path = args[2]
        load_model(load_path, model, optimizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00000001)
    # Load unsmile dataset
    train_texts, train_labels = load_unsmile_data(train=True, return_column_name=False, return_torch=True)
    test_texts, test_labels = load_unsmile_data(train=False, return_column_name=False, return_torch=True)
    # Make text dataset
    train_data_set = TextDataset(train_texts, train_labels)
    test_data_set = TextDataset(test_texts, test_labels)
    # Set hyper parameters
    loss_function = torch.nn.CrossEntropyLoss()
    epochs = 50
    print_interval = 1
    batch_size = 32
    # Train
    train_loop(train_data_set=train_data_set, test_data_set=test_data_set, epochs=epochs, model=model, device=device,
               batch_size=batch_size, loss_function=loss_function, optimizer=optimizer, print_interval=print_interval,
               accuracy_function=calculate_accuracy_multi_class, X_on_the_fly_function=model.embed_texts, test_first=True)
    save_model(save_path, model, optimizer)


