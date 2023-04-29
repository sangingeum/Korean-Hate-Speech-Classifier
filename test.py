import torch
from Model import *
from utils import *
from unsmile_dataset.UnsmileDataset import *
import sys

if __name__ == '__main__':
    args = sys.argv
    if len(args) <= 1:
        print("Error: Please provide an argument.")
        sys.exit(1)
    # Load model
    device = get_device_name_agnostic()
    model = ClassificationModel(input_size=768, hidden_size=1024, num_layers=1,
                                      mlp_units=[256, 128, 64, 11], bidirectional=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    model_path = args[1]
    load_model(model_path, model, optimizer)
    #
