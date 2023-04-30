from sklearn.metrics import classification_report
from model import *
from utils import *
from unsmile_dataset.UnsmileDataset import *
from torch.utils.data import DataLoader
from collections import deque
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
    # Load test dataset
    test_dataset, label_features = load_unsmile_data(train=False, return_label_features=True)
    # Make dataloader
    batch_size = 32
    data_loader = DataLoader(test_dataset, batch_size=batch_size)
    y_preds = deque()
    ys = deque()
    # Predict label
    model.eval()
    with torch.inference_mode():
        for (X, y) in data_loader:
            X = model.embed_texts(X)
            y_pred = torch.round(model(X))
            y_preds.append(y_pred)
            ys.append(y)

        y_preds = torch.cat(list(y_preds), dim=0).cpu()
        ys = torch.cat(list(ys), dim=0).cpu()
    # Convert
    print(classification_report(ys, y_preds, target_names=label_features))

