from model import *
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
    # Load test dataset
    _, label_features = load_unsmile_data(train=False, return_label_features=True)
    model.eval()
    with torch.inference_mode():
        while True:
            print("테스트하고 싶은 텍스트 입력")
            input_text = input()
            embedding = model.embed_texts([input_text])
            output = torch.squeeze(torch.round(model(embedding)))
            predicted_categories = []
            for i, prediction in enumerate(output):
                if prediction > 0.5:
                    predicted_categories.append(label_features[i])
            print("predicted_categories:")
            print(predicted_categories)
            print("\n")