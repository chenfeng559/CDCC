import torch
import torch.nn as nn
from drnn import multi_dRNN_with_dilations
import torch.nn.functional as F
import numpy as np
import random
import os
from load_data_txt import get_dataloader
from sklearn.cluster import KMeans
from utils_DTCR import evaluation

class Input(nn.Module):
    """
    input_embeding_units: numbers of input embeding 
    inchannel：numbers of input feature
    """

    def __init__(self, input_embeding_units, inchannels):
        super().__init__()
        self.input_embeding_units = input_embeding_units
        self.in_channels = inchannels
        self.input_embeding = nn.Linear(self.in_channels,self.input_embeding_units)
    def forward(self, x):
        x = self.input_embeding(x)
        return x

class Output(nn.Module):
    """
    input_embeding_units: numbers of input embeding 
    inchannel：numbers of input feature
    """

    def __init__(self, input_embeding_units, inchannels):
        super().__init__()
        self.input_embeding_units = input_embeding_units
        self.in_channels = inchannels
        self.output_embeding = nn.Linear(self.input_embeding_units,self.in_channels)
    def forward(self, x):
        x = self.output_embeding(x)
        return x




class Encoder(nn.Module):
    """
    """

    def __init__(self, hidden_structs, dilations, input_dims, cell_type, batch_first, device):
        super().__init__()
        self.multi_dRNN_with_dilations = multi_dRNN_with_dilations(hidden_structs, dilations, input_dims, cell_type, batch_first=batch_first, device=device).to(device)
        
    def forward(self, inputs):
        outputs, hiddens=self.multi_dRNN_with_dilations(inputs)
        return outputs, hiddens


class Deocder(nn.Module):
    def __init__(self, hidden_structs, input_dims, cell_type, batch_first, device):
        super().__init__()
        self.hidden_dim = 2*sum(hidden_structs)
        self.input_dims = input_dims
        self.cell_type = cell_type
        self.batch_first = batch_first
        if self.cell_type == "GRU":
            self.cell = nn.GRU(self.input_dims,self.hidden_dim)
        elif self.cell_type == "RNN":
            self.cell = nn.RNN(self.hidden_dim,self.hidden_dim)
        elif self.cell_type == "LSTM":
            self.cell = nn.LSTM(self.hidden_dim,self.hidden_dim)
    
        self.linear = nn.Linear(self.hidden_dim,input_dims)
        
    def forward(self, inputs, hidden):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)     #(B,1,D)->(1,B,D)
        outputs, hiddens = self.cell(inputs,hidden)
        outputs = self.linear(outputs)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)     #(1,B,D)->(B,1,D)
        return outputs, hiddens


class Classifier(nn.Module):
    """
    input_embeding_units: numbers of input embeding 
    inchannel：numbers of input feature
    """

    def __init__(self, encoder_state_dim):
        super().__init__()
        self.input_dim = encoder_state_dim
        self.hidden_dim = 128
        self.output_dim = 2

        self.fc1 = nn.Linear(self.input_dim,self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim,self.output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class DTCR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_structs = args.hidden_structs
        self.dilations = args.dilations
        self.input_dims = args.input_dims
        self.cell_type = args.cell_type
        self.batch_size = args.batch_size
        self.cluster_num = args.cluster_num
        self.batch_first = args.batch_first
        self.device = args.device
        self.train_decoder = args.train_decoder

        self.Encoder = Encoder(self.hidden_structs, self.dilations, self.input_dims,self.cell_type, batch_first=self.batch_first, device=self.device)
        self.Decoder = Deocder(self.hidden_structs, self.input_dims, self.cell_type, batch_first=self.batch_first, device=self.device)
        self.F = nn.Parameter(torch.eye(self.batch_size, self.cluster_num), requires_grad=False)
        self.Classifier = Classifier(2*sum(self.hidden_structs))
        '''
        # Input embeding、Output embeding
        self.Input_embeding_list = nn.ModuleList()
        self.Output_embeding_list = nn.ModuleList()
        
        for i in range(args.dataset_num):
            self.Input_embeding_list.append(Input(args.input_embeding_units,args.inchannels[i]))
            self.Output_embeding_list.append(Output(args.input_embeding_units,args.inchannels[i]))
        '''
    def forward(self, inputs, targets):
        #inputs = self.Input_embeding_list[datasetid](inputs)
        if self.batch_first:
            time_length = targets.shape[1]
            input_reversed = torch.flip(inputs,[1])
        else:
            time_length = targets.shape[0]
            input_reversed = torch.flip(inputs,[0])
        encoder_outputs_fw, encoder_hiddens_fw = self.Encoder(inputs)
        encoder_outputs_bw, encoder_hiddens_bw = self.Encoder(input_reversed)
        hiddens_fw = torch.cat(encoder_hiddens_fw,dim=2)
        hiddens_bw = torch.cat(encoder_hiddens_bw,dim=2)
        encoder_state = torch.cat([hiddens_fw,hiddens_bw],dim=2)# [1,batch,2*sum(hidden_structs)]

        final_outputs = []
        for i in range(time_length):
            if i==0:
                hiddens = encoder_state
                input = targets[:,0,:].unsqueeze(1)
            if self.train_decoder:
                outputs, hiddens = self.Decoder(input,hiddens)
                input = targets[:,i,:].unsqueeze(1)
            else:
                outputs, hiddens = self.Decoder(input,hiddens)
                input = outputs
            final_outputs.append(outputs)
        final_outputs = torch.cat(final_outputs,dim=1)
        #final_outputs = self.Input_embeding_list[datasetid](final_outputs)

        W = encoder_state.squeeze()
        WT = torch.transpose(W,0,1)
        WTW = torch.matmul(W,WT)
        FTWTWF = torch.matmul(torch.matmul(torch.transpose(self.F,0,1), WTW),self.F)
        pred_pro = self.Classifier(encoder_state.squeeze())

        return final_outputs,encoder_state.squeeze(), WTW, FTWTWF, pred_pro


def train(args, model, train_loader, epoch, optimizer, max_patience):
    loss_mse = nn.MSELoss()
    loss_ce = nn.CrossEntropyLoss()
    min_loss = 1000000
    patience = 0
    for i in range(epoch):
        train_loss = 0
        train_constr = 0
        train_kmeans_loss = 0
        train_cls_loss = 0
        for batch_idx, (inputs, label, real_fake_label) in enumerate(train_loader):
            inputs = inputs.type(torch.FloatTensor).to(args.device)
            real_fake_label = real_fake_label.type(torch.FloatTensor).to(args.device)
            #real_index = torch.where(real_fake_label == 0)
            real_index = torch.where(real_fake_label[:, 0] == 1)

            optimizer.zero_grad()
            inputs_reconstr, encoder_state, WTW, FTWTWF, real_fake_pred_pro = model(inputs,inputs)
            # 重构 loss
            loss_constr = loss_mse(inputs[real_index], inputs_reconstr[real_index])
            # kmeans loss
            loss_Kmeans = torch.trace(WTW)-torch.trace(FTWTWF)
            # classification loss
            loss_cls = loss_ce(real_fake_pred_pro,real_fake_label) 
            loss_total = loss_constr + args.lambda1*loss_Kmeans + loss_cls
            loss_total.backward()
            optimizer.step()
            
            # 每10个epoch进行一次F矩阵更新
            if i%10==0 and i!=0:
                part_hidden_val = np.array(encoder_state.cpu().detach()).reshape(-1, np.sum(args.hidden_structs) * 2)
                W = part_hidden_val.T
                U, sigma, VT = np.linalg.svd(W)
                sorted_indices = np.argsort(sigma)
                topk_evecs = VT[sorted_indices[:-args.cluster_num - 1:-1], :]
                F_new = topk_evecs.T
                model.F = nn.Parameter(torch.tensor(F_new).to(args.device), requires_grad=False)
           
            train_loss += loss_total.item()
            train_constr += loss_constr.item()
            train_kmeans_loss += loss_Kmeans.item()
            train_cls_loss += loss_cls.item()

        print(
            "For epoch：{} ".format(i),
            " Loss is : %.3f" % (train_loss / (batch_idx + 1)),
            "mse loss:{}".format(train_constr / (batch_idx + 1)),
            "kmeans loss:{}".format(train_kmeans_loss / (batch_idx + 1)),
            "cls loss:{}".format(train_cls_loss / (batch_idx + 1))
        )
        if train_loss<min_loss:
            min_loss = train_loss
            torch.save(model.state_dict(), 'DTCR/model_weights_DTCR/model.pth')
        else:
            patience+=1
            if patience>max_patience:
                break
        


def test(args,model,test_loader):
    encoder_state_list = []
    label_list = []
    model.eval()
    for batch_idx, (inputs, label, real_fake_label) in enumerate(test_loader):
        inputs = inputs.type(torch.FloatTensor).to(args.device)
        real_fake_label = real_fake_label.type(torch.FloatTensor).to(args.device)
        real_index = torch.where(real_fake_label[:, 0] == 1)
        inputs_reconstr, encoder_state, WTW, FTWTWF, real_fake_pred_pro = model(inputs,inputs)
        encoder_state_list.extend(encoder_state[real_index].cpu().detach().numpy())
        label_list.extend(label[real_index])
    # 聚类并评估
    test_hidden_val = np.array(encoder_state_list)
    label = np.array(label_list,dtype=int)

    km = KMeans(n_clusters=args.cluster_num)
    km_idx = km.fit_predict(test_hidden_val)
    evaluation(prediction=km_idx, label=label)




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现



if __name__ == "__main__":
    '''
    hidden_structs = [100,50,50]
    dilations = [1,2,4]
    input_dims = 55
    cell_type = 'GRU'
    model = Encoder(hidden_structs,dilations,input_dims,cell_type,batch_first=True,device='cpu')
    x = torch.randn((64,25,55))
    inputs, outputs=model(x)
    hidden = torch.cat(outputs,dim=2)
    print(inputs.shape)
    print(hidden.shape)

    model_de = Deocder(hidden_structs,input_dims,cell_type,batch_first=True,device='cpu')

    final_outputs = []
    for i in range(x.shape[1]):
        if i==0:
            hiddens=hidden
        outputs, hiddens = model_de(x[:,i,:].unsqueeze(1),hiddens)
        final_outputs.append(outputs.squeeze())
    print(torch.stack(final_outputs,dim=1).shape)
    '''

    set_seed(42)
    from config_DTCR import get_arguments
    parser = get_arguments()
    args = parser.parse_args()
    args.hidden_structs = [100,50,50]
    args.dilations = [1,2,4]
    args.input_dims = 1
    args.cell_type = 'GRU'
    args.batch_size = 16
    args.lambda1 =1e-2
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #x = torch.randn((64,25,55))
    
    model = DTCR(args)
    model.load_state_dict(torch.load('DTCR/model_weights_DTCR/model.pth'))
    model.to(args.device)
    '''
    trainloader =get_dataloader(batch_size=args.batch_size)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-4, momentum=args.momentum
    )
    import datetime
    starttime = datetime.datetime.now()
    train(args, model, trainloader, 300, optimizer, 300)
    endtime = datetime.datetime.now()
    print (endtime - starttime)
    '''
    testloader =get_dataloader(filename='DTCR/Coffee/Coffee_TEST',batch_size=args.batch_size)
    test(args,model,testloader)
    


   
