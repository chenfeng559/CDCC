import torch
import torch.nn as nn
from DTCR.drnn import multi_dRNN_with_dilations
import torch.nn.functional as F

class Input(nn.Module):
    def __init__(self, input_embeding_units, inchannels):
        """
        input_embeding_units: numbers of input embeding 
        inchannel：numbers of input feature
        """
        super().__init__()
        self.input_embeding_units = input_embeding_units
        self.in_channels = inchannels
        self.input_embeding = nn.Linear(self.in_channels, self.input_embeding_units)
    def forward(self, x):
        x = self.input_embeding(x)
        return x

class Output(nn.Module):
    def __init__(self, input_embeding_units, inchannels):
        """
        input_embeding_units: numbers of input embeding 
        inchannel：numbers of input feature
        """
        super().__init__()
        self.input_embeding_units = input_embeding_units
        self.in_channels = inchannels
        self.output_embeding = nn.Linear(self.input_embeding_units,self.in_channels)
    def forward(self, x):
        x = self.output_embeding(x)
        return x

class Encoder(nn.Module):
    def __init__(self, hidden_structs, dilations, input_dims, cell_type, batch_first, device):
        """
        Args:
            hidden_structs (list): a list, each element indicates the hidden node dimension of each layer.
            dilations (list): a list, each element indicates the dilated rate of each layer.
            input_dims (int): input feature dim
            cell_type (str): RNN cell type
            batch_first (bool)
            device (str)
        """        
        super().__init__()
        self.multi_dRNN_with_dilations = multi_dRNN_with_dilations(hidden_structs, dilations, input_dims, cell_type, batch_first=batch_first, device=device).to(device)
        
    def forward(self, inputs):
        outputs, hiddens=self.multi_dRNN_with_dilations(inputs)
        return outputs, hiddens


class Deocder(nn.Module):
    def __init__(self, hidden_structs, input_dims, cell_type, batch_first, device):
        """
        Args:
            hidden_structs (list): a list, each element indicates the hidden node dimension of each layer.
            input_dims (int): input feature dim
            cell_type (str): RNN cell type
            batch_first (bool)
            device (str)
        """        
        super().__init__()
        # Encoder  hidden
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
    # seq2seq架构下，decoder每次只接受一个时间步的输入，也只给出一个时间步测输出
    
    def forward(self, inputs, hidden):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)     #(B,1,D)->(1,B,D)
        outputs, hiddens = self.cell(inputs,hidden)
        outputs = self.linear(outputs)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)     #(1,B,D)->(B,1,D)
        return outputs, hiddens


class Classifier(nn.Module):
    def __init__(self, encoder_state_dim):
        """
        Args:
            encoder_state_dim (int)): Encoder_state dim
        """        
        super().__init__()
        self.input_dim = encoder_state_dim
        self.hidden_dim = 128
        self.output_dim = 1

        self.fc1 = nn.Linear(self.input_dim,self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim,self.output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = torch.sigmoid(self.output(x))
        return output


class DTCR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_structs = args.hidden_structs
        self.dilations = args.dilations
        self.encoder_input_dims = args.input_embeding_units
        self.cell_type = args.cell_type
        self.batch_size = args.batch_size
        self.cluster_num = args.cluster_num
        self.batch_first = args.batch_first
        self.device = args.device
        self.train_decoder = args.train_decoder

        self.Encoder = Encoder(self.hidden_structs, self.dilations, self.encoder_input_dims,self.cell_type, batch_first=self.batch_first, device=self.device)
        self.Decoder = Deocder(self.hidden_structs, self.encoder_input_dims, self.cell_type, batch_first=self.batch_first, device=self.device)
        # initialize F matrix
        self.F = nn.Parameter(torch.eye(self.batch_size, self.cluster_num), requires_grad=False)
        self.F = nn.init.orthogonal(self.F, gain=1)

        self.Classifier = Classifier(2*sum(self.hidden_structs))

        # Input embeding Layer、Output embeding Layer
        self.Input_embeding_list = nn.ModuleList()
        self.Output_embeding_list = nn.ModuleList()
        for i in range(args.dataset_num):
            self.Input_embeding_list.append(Input(args.input_embeding_units, args.inchannels[i]))
            self.Output_embeding_list.append(Output(args.input_embeding_units, args.inchannels[i]))

    def forward(self, inputs, datasetid):
        """
        Args:
            inputs (tensor): shape:[Batch,Length,Dim] 
            datasetid (int): id number of dataset
        Returns:
            _type_: final_outputs:重构结果, encoder_state.squeeze():Encoder隐藏状态, WTW, FTWTWF, pred_pro：real-fake分类预测值
        """        
        inputs = self.Input_embeding_list[datasetid](inputs)
        if self.batch_first:
            time_length = inputs.shape[1]
            input_reversed = torch.flip(inputs,[1])
        else:
            time_length = inputs.shape[0]
            input_reversed = torch.flip(inputs,[0])
        # get Encoder state
        encoder_outputs_fw, encoder_hiddens_fw = self.Encoder(inputs)
        encoder_outputs_bw, encoder_hiddens_bw = self.Encoder(input_reversed)
        hiddens_fw = torch.cat(encoder_hiddens_fw, dim=2)
        hiddens_bw = torch.cat(encoder_hiddens_bw, dim=2)
        encoder_state = torch.cat([hiddens_fw, hiddens_bw], dim=2) # [1,batch,2*sum(hidden_structs)]
        # get Decoder output
        final_outputs = []
        for i in range(time_length):
            if i==0:
                # use Encoder state initialize encoder_state
                hiddens = encoder_state
                input = inputs[:,0,:].unsqueeze(1)
            if self.train_decoder:
                # if train, use real data as decoder input
                outputs, hiddens = self.Decoder(input, hiddens)
                input = inputs[:,i,:].unsqueeze(1)
            else:
                outputs, hiddens = self.Decoder(input, hiddens)
                input = outputs
            final_outputs.append(outputs)
        final_outputs = torch.cat(final_outputs, dim=1)
        final_outputs = self.Output_embeding_list[datasetid](final_outputs)

        W = encoder_state.squeeze()
        WT = torch.transpose(W,0,1)
        WTW = torch.matmul(W,WT)
        FTWTWF = torch.matmul(torch.matmul(torch.transpose(self.F,0,1), WTW),self.F)
        pred_pro = self.Classifier(encoder_state.squeeze())

        return final_outputs,encoder_state.squeeze(), WTW, FTWTWF, pred_pro



        

'''
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
'''


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
    '''
    from load_data_txt import get_dataloader
    set_seed(42)
    from config_DTCR import get_arguments
    parser = get_arguments()
    args = parser.parse_args()
    args.hidden_structs = [100,50,50]
    args.dilations = [1,2,4]
    args.inchannels = [1]
    args.cell_type = 'GRU'
    args.batch_size = 16
    args.lambda1 =1e-2
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #x = torch.randn((64,25,55))
    
    model = DTCR(args)
    #model.load_state_dict(torch.load('DTCR/model_weights_DTCR/model.pth'))
    model.to(args.device)
    
    trainloader =get_dataloader(batch_size=args.batch_size)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-4, momentum=args.momentum
    )
    import datetime
    starttime = datetime.datetime.now()
    train(args, model, trainloader, 300, optimizer, 300)
    endtime = datetime.datetime.now()
    print (endtime - starttime)
    
    testloader =get_dataloader(filename='DTCR/Coffee/Coffee_TEST',batch_size=args.batch_size)
    test(args,model,testloader)
    '''
    

