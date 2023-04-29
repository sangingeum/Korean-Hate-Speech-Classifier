from model import *
from unsmile_dataset.UnsmileDataset import *
from utils import *
import sys

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

    # Load unsmile dataset
    train_data_set = load_unsmile_data(train=True)
    test_data_set = load_unsmile_data(train=False)
    # Set hyper parameters
    loss_function = torch.nn.CrossEntropyLoss()
    epochs = 800
    print_interval = 1
    batch_size = 32
    # Train
    train_loop(train_data_set=train_data_set, test_data_set=test_data_set, epochs=epochs, model=model, device=device,
               batch_size=batch_size, loss_function=loss_function, optimizer=optimizer, print_interval=print_interval,
               accuracy_function=calculate_accuracy_multi_class, X_on_the_fly_function=model.embed_texts, test_first=True)
    save_model(save_path, model, optimizer)


