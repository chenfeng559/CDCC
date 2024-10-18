from os import fwalk
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



def load_data(filename):
    data_label = np.loadtxt(filename, delimiter=',')
    data = data_label[:, 1:]
    label = data_label[:, 0].astype(np.int32)
    return data, label

def get_fake_sample(data):
    sample_nums = data.shape[0]
    series_len = data.shape[1]
    mask = np.ones(shape=[sample_nums, series_len])
    rand_list = np.zeros(shape=[sample_nums, series_len])

    fake_position_nums = int(series_len * 0.2)
    fake_position = np.random.randint(low=0, high=series_len, size=[sample_nums, fake_position_nums])

    for i in range(fake_position.shape[0]):
        for j in range(fake_position.shape[1]):
            mask[i, fake_position[i, j]] = 0

    for i in range(rand_list.shape[0]):
        count = 0
        for j in range(rand_list.shape[1]):
            if j in fake_position[i]:
                rand_list[i, j] = data[i, fake_position[i, count]]
                count += 1
    fake_data = data * mask + rand_list * (1 - mask)
    real_fake_labels = np.zeros(shape=[sample_nums * 2, 2])
    for i in range(sample_nums * 2):
        if i < sample_nums:
            real_fake_labels[i, 0] = 1
        else:
            real_fake_labels[i, 1] = 1
    return fake_data, real_fake_labels

def get_dataloader(filename='DTCR/Coffee/Coffee_TRAIN', batch_size=16, shuffle=True, num_workers=2, drop_last=True):
    data, label = load_data(filename)
    trainset = CustomDataset_Anomaly(data, label)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,drop_last=drop_last
    )
    return trainloader

class CustomDataset_Anomaly(Dataset):
    def __init__(self, inputs, labels):
        """
        This class creates a torch dataset.

        """
        self.real_inputs = inputs
        self.fake_inputs, self.real_fake_labels = get_fake_sample(inputs)
        self.inputs = np.concatenate([self.real_inputs,self.fake_inputs],axis=0)
        self.labels = np.concatenate([labels,labels])


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input = torch.tensor(self.inputs[idx]).unsqueeze(1)
        label = torch.tensor(self.labels[idx]).unsqueeze(0)
        real_fake_label = torch.tensor(self.real_fake_labels[idx])
        
        return (input, label, real_fake_label)


if __name__ == "__main__":
    print('end')



