import torch.nn as nn
import torch
import torch.nn.functional as F
from config import get_arguments
from load_data_new import get_loader_arrf_train_test


class Classifier(nn.Module):
    """
    input_embeding_units: numbers of input embeding 
    inchannel：numbers of input feature
    """

    def __init__(self, args):
        super().__init__()
        self.input_dim = args.n_hidden*args.n_units[1]#args.n_hidden*args.n_units[0]
        self.hidden_dim = args.classfier_hidden_dim
        self.output_dim = 1

        self.fc1 = nn.Linear(self.input_dim,self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim,self.output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        return x







    

