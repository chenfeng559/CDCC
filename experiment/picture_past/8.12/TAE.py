import torch.nn as nn
import torch
from sklearn.cluster import AgglomerativeClustering
import gc

from torch.nn.modules import pooling

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



        


class TAE_encoder(nn.Module):
    """
    Class for temporal autoencoder encoder.
    n_filters: number of filters in convolutional layer
    kernel_size: size of kernel in convolutional layer
    strides: strides in convolutional layer
    pool_size: pooling size in max pooling layer, must divide time series length
    n_units: numbers of units in the two BiLSTM layers
    n_hidden: size of z
    input_linear_n_units: numbers of input embeding 
    
    """

    def __init__(self, args):
        super().__init__()
        self.n_filters = args.n_filters
        self.kernel_size = args.kernel_size
        self.strides = args.strides
        self.pool_size = args.pool_size
        self.hidden_lstm_1 = args.n_units[0]
        self.hidden_lstm_2 = args.n_units[1]

        self.in_channels = args.in_channels
        
        ## CNN PART
        ### output shape (batch_size, 50 , n_hidden = 64)
        self.conv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.n_filters,
                kernel_size=self.kernel_size,
                stride=self.strides,
                padding='same'
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.pool_size),
        )

        ## LSTM PART
        ### output shape (batch_size , n_hidden = 64 , 50)
        self.lstm_1 = nn.LSTM(
            input_size=self.n_filters,
            hidden_size=self.hidden_lstm_1,
            batch_first=True,
            bidirectional=True,
        )

        ### output shape (batch_size , n_hidden = 64 , 1)
        self.lstm_2 = nn.LSTM(
            input_size=self.hidden_lstm_1,
            hidden_size=self.hidden_lstm_2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):

        ## encoder
        out_cnn = self.conv_layer(x)
        out_cnn = out_cnn.permute((0, 2, 1))#batch_size, seq, in_channels
        out_lstm1, _ = self.lstm_1(out_cnn)
        out_lstm1 = torch.sum(
            out_lstm1.view(
                out_lstm1.shape[0], out_lstm1.shape[1], 2, self.hidden_lstm_1 #batch_size, seq, direction, hindden
            ),
            dim=2,
        )

        features, _ = self.lstm_2(out_lstm1)
        features = torch.sum(
            features.view(
                features.shape[0], features.shape[1], 2, self.hidden_lstm_2
            ),
            dim=2,
        )  ## (batch_size , seq ,out_lstm2)

        return features

# ConvTranspose1dSame padding='same'
class ConvTranspose1dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvTranspose1dSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        input_length = x.size(2)

        # 计算输出的长度
        output_length = (input_length - 1) * self.stride + self.kernel_size

        # 计算padding的数量
        padding_needed = max(0, output_length - input_length)
        padding_left = padding_needed // 2
        padding_right = padding_needed - padding_left

        # 前向传播
        x = self.conv_transpose(x)

        # 修剪或填充结果以匹配所需的长度
        if padding_left > 0 or padding_right > 0:
            x = x[:, :, padding_left:-padding_right]

        return x


class TAE_decoder(nn.Module):
    """
    Class for temporal autoencoder decoder.
    pool_size: pooling size in max pooling layer, must divide time series length
    n_hidden : hidden size of the encoder, time_steps/pool_size
    """

    def __init__(self, args):
        super().__init__()

        self.pool_size = args.pool_size
        self.n_hidden = args.n_hidden
        self.kernel_size = args.kernel_size
        self.strides = args.strides
        self.output_channel = args.input_embeding_units
        self.hidden_lstm_2 = args.n_units[1]

        # upsample
        self.up_layer = nn.Upsample(scale_factor=self.pool_size)
        self.deconv_layer = nn.ConvTranspose1d(in_channels=self.hidden_lstm_2,
            out_channels=self.output_channel,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(self.kernel_size-1)//2)
        '''
        ConvTranspose1dSame(
            in_channels=self.hidden_lstm_2,
            out_channels=self.output_channel,
            kernel_size=self.kernel_size,
            stride=self.strides,
        )
        '''
    def forward(self, features):
        features = features.permute(0,2,1) # (batch,seq,feature)->(batch,feature,seq); 
        upsampled = self.up_layer(features)
        out_deconv = self.deconv_layer(upsampled)
        out_deconv = out_deconv.permute(0,2,1)#(batch,feature,seq)->(batch,seq,feature)

        return out_deconv


class TAE(nn.Module):
    """
    # Arguments
    timesteps: number of timesteps (can be None for variable length sequences)
    n_filters: number of filters in convolutional layer
    kernel_size: size of kernel in convolutional layer
    strides: strides in convolutional layer
    pool_size:  pooling size in max pooling layer, must divide time series length
    n_units: numbers of units in the two BiLSTM layers

    dataset_num: number of dataset
    """
    def __init__(self, args):
        super().__init__()
        args.n_hidden = int(args.timesteps/args.pool_size)
        # Input embeding、Output embeding
        self.Input_embeding_list = nn.ModuleList()
        self.Output_embeding_list = nn.ModuleList()
        for i in range(args.dataset_num):
            self.Input_embeding_list.append(Input(args.input_embeding_units,args.inchannels[i]))
            self.Output_embeding_list.append(Output(args.input_embeding_units,args.inchannels[i]))
        # set Encoder input dim
        args.in_channels = args.input_embeding_units
        #Encoder
        self.tae_encoder = TAE_encoder(args)
        #Decoder
        self.tae_decoder = TAE_decoder(args)


    def forward(self, x, datasetid):
        # Input embeding
        x = self.Input_embeding_list[datasetid](x)
        #print('================='+str(datasetid)+"================")
        #print(torch.mean(x))
        x = x.permute(0,2,1) #[batch,len,feature]->[batch,feature,len]
        # encoder
        features = self.tae_encoder(x)
        # decoder
        out_deconv = self.tae_decoder(features)
        # Output embeding
        out_deconv = self.Output_embeding_list[datasetid](out_deconv)
        return features, out_deconv


if __name__ == "__main__":
    test_a = torch.randn((10,80,15))#(bathc_size, seq_len, feature)
    test_b = torch.randn((10,80,20))#(bathc_size, seq_len, feature)
    from config import get_arguments
    parser = get_arguments()
    args = parser.parse_args()
    args.timesteps = 80
    #args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tae = TAE(args)
    #tae = tae.to(args.device)
    #test_x = test_x.to(args.device)
    z, out = tae(test_a,0)
    print(z.shape)
    print(out.shape)
    z_b, out_b = tae(test_b,1)
    print(z_b.shape)
    print(out_b.shape)
    