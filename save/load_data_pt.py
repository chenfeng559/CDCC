import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np



class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()
        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 2:  # make sure the seq in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]

 
    def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def get_loader_pt(data_path, args, clamp_len=0, pad_len=0, label_modify=0):

    train_data = torch.load(os.path.join(data_path, "train.pt")) #(sample,feature,seq)
    valid_data = torch.load(os.path.join(data_path, "val.pt"))
    test_data = torch.load(os.path.join(data_path, "test.pt"))

    # clamp seq
    if clamp_len!=0:
        train_data['samples']= train_data['samples'].index_select(dim=2, index=torch.arange(clamp_len))
        valid_data['samples']= valid_data['samples'].index_select(dim=2, index=torch.arange(clamp_len))
        test_data['samples']= test_data['samples'].index_select(dim=2, index=torch.arange(clamp_len))
    # pad seq
    # modify label
    if label_modify!=0:
        train_data['labels'] = train_data['labels']+label_modify
        valid_data['labels'] = valid_data['labels']+label_modify
        test_data['labels'] = test_data['labels']+label_modify


    print(np.unique(train_data['labels'].numpy()))
    #print(list(train_data['labels']).count(0))

    train_dataset = Load_Dataset(train_data)
    valid_dataset = Load_Dataset(valid_data)
    test_dataset = Load_Dataset(test_data)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                               shuffle=False, drop_last=args.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size,
                                               shuffle=False, drop_last=args.drop_last,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    
    X_scaled = train_data['samples'] 
    X_scaled = X_scaled.permute(0,2,1)
    X_scaled = X_scaled.numpy()

    return train_loader, X_scaled


if __name__ == "__main__":
    from config import get_arguments
    parser = get_arguments()
    args = parser.parse_args()
    args.drop_last = True
    args.batch_size = 20
    trainloader, X_scaled = get_loader_pt('data/HAR',args,label_modify=0)
    #for data,label in trainloader:
        #print(data.shape,label.shape)
    
    
    #print(len(trainloader))
    print(X_scaled.shape)