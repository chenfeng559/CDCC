import torch
import torch.nn as nn

class multi_dRNN_with_dilations(nn.Module):
    def __init__(self, hidden_structs, dilations, input_dims, cell_type, batch_first=False, device='cpu'):
        """
        Args:
            hidden_structs (list): a list, each element indicates the hidden node dimension of each layer.
            dilations (list): a list, each element indicates the dilated rate of each layer.
            input_dims (int): input feature dim
            cell_type (str): RNN cell type
            batch_first (bool)
            device (str)
        """        
        super(multi_dRNN_with_dilations, self).__init__()
        self.hidden_structs = hidden_structs
        self.dilations = dilations
        self.input_dims = input_dims
        self.cell_type = cell_type
        self.batch_first = batch_first
        self.device = device
        
        # define cells
        self.cells = nn.Sequential(*self.construct_cells(self.hidden_structs, self.cell_type, self.input_dims))

    def forward(self, inputs):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)     #(B,L,D)->(L,B,D)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            inputs, hidden = self.drnn_layer(cell, inputs, dilation)
            # 取最后一个时间步的输出作为hidden
            outputs.append(inputs[-1:])
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        return inputs, outputs

    '''应用drnn进行预测'''
    def drnn_layer(self, cell, inputs, rate):
        """
        Args:
            cell (str): RNN cell type
            inputs (tensor)
            rate (int): dilated rate
        Returns:
            outputs， hidden
        """        
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size
        # 填充inputs，使得其长度能够被rate整除
        inputs, _ = self._pad_inputs(inputs, n_steps, rate)
        # 压缩inputs，实现dilated机制(L,B,D)->(L/rate,B*rate,D)
        dilated_inputs = self._prepare_inputs(inputs, rate)
        # cell推理得出结果
        dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        # 反压缩，(L/rate,B*rate,D)->(L,B,D)
        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        # 删除padding
        outputs = self._unpad_outputs(splitted_outputs, n_steps)
        return outputs, hidden

    '''用于填充inputs'''
    def _pad_inputs(self, inputs, n_steps, rate):
        """
        Args:
            inputs (tensor)
            n_steps (int): inputs seq Length
            rate (int): dilated rate
        Returns:
            _type_: _description_
        """        
        #长度不能被rate整除时，需要进行填充
        if (n_steps % rate) != 0:
            dilated_steps = n_steps // rate + 1
            zeros_ = torch.zeros(dilated_steps * rate - inputs.shape[0],inputs.shape[1],inputs.shape[2])
            zeros_ = zeros_.to(self.device)
            inputs = torch.cat((inputs, zeros_))

        else:
            dilated_steps = n_steps // rate
        return inputs, dilated_steps

    '''对inputs进行压缩，实现dilated机制'''
    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs

    '''应用cell推理'''
    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size):
        if self.cell_type == 'LSTM':
            c, m = self.init_hidden(batch_size * rate, hidden_size)
            hidden = (c.unsqueeze(0), m.unsqueeze(0))
        else:
            hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    '''删除padding'''
    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    '''对压缩后inputs的输出结果，进行反压缩'''
    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    '''构建cell'''
    def construct_cells(self, hidden_structs, cell_type, input_dim):
        """
        Constructs a list of RNN cells based on the given structure and cell type.
        """
        if cell_type not in ["RNN", "LSTM", "GRU"]:
            raise ValueError("The cell type is not currently supported.")

        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError

        for i in range(len(hidden_structs)):
            if i == 0:
                c = cell(input_dim, hidden_structs[i])
            else:
                c = cell(hidden_structs[i-1], hidden_structs[i])
            layers.append(c)
        return layers

    '''初始化cell状态'''
    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(batch_size, hidden_dim)
        hidden = hidden.to(self.device)
        if self.cell_type == "LSTM":
            memory = torch.zeros(batch_size, hidden_dim)
            memory = memory.to(self.device)
            return (hidden, memory)
        else:
            return hidden