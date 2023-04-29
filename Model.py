import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

# "klue/bert-base", lstm -> 100 epochs, 65.5
# "beomi/KcELECTRA-base-v2022", lstm -> 135 epochs, 72
# "beomi/KcELECTRA-base-v2022", lstm -> 0.000001 lr, 777 epochs, 77
class ClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, mlp_units=[256, 128, 64, 11], bidirectional=False):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.checkpoint = "beomi/KcELECTRA-base-v2022"
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModel.from_pretrained(self.checkpoint).to(self.device)
        self.dropout = 0
        if num_layers > 1:
            self.dropout = 0.2
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=self.dropout, bidirectional=bidirectional).to(self.device)
        mlp_layers = []
        if bidirectional:
            mlp_layers.append(nn.Linear(in_features=self.hidden_size*2, out_features=mlp_units[0]))
        else:
            mlp_layers.append(nn.Linear(in_features=self.hidden_size, out_features=mlp_units[0]))
        for i in range(len(mlp_units)-1):
            mlp_layers.append(nn.Dropout(p=0.2))
            mlp_layers.append(nn.Linear(in_features=mlp_units[i], out_features=mlp_units[i+1]))
        self.mlp_layer = nn.Sequential(*mlp_layers).to(self.device)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X: torch.Tensor):
        d = 2 if self.bidirectional else 1
        N = X.shape[0]
        h_0 = torch.zeros((self.num_layers * d, N, self.hidden_size)).to(self.device)
        c_0 = torch.zeros((self.num_layers * d, N, self.hidden_size)).to(self.device)
        y, (_, _) = self.lstm(X, (h_0, c_0))
        y = self.softmax(self.mlp_layer(y[:, -1, :]))
        return y

    def embed_texts(self, texts: list):
        tokenized_texts = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        model_output = self.model(input_ids=tokenized_texts['input_ids'],
                                  attention_mask=tokenized_texts['attention_mask'])
        embeddings = model_output["last_hidden_state"]
        return embeddings

