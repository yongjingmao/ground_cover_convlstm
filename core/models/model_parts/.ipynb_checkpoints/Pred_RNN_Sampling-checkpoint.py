import torch.nn as nn
import torch
import torch.nn.functional as F

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, input_dim, h_channels, kernel_size, img_width, img_height, layer_norm_flag):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.h_channels = h_channels
        self.kernel_size = kernel_size
        self.layer_norm_flag = layer_norm_flag
        self.img_width = img_width
        self.img_height = img_height
        
        self.padding = kernel_size // 2
        self.stride = 1
        self._forget_bias = 1.0
        
        if self.layer_norm_flag:
            self.conv_x = nn.Sequential(
                nn.Conv2d(self.input_dim, self.h_channels * 7, 
                          kernel_size=kernel_size, stride=self.stride, padding=self.padding, bias=False),
                nn.LayerNorm([self.h_channels * 7, self.img_width, self.img_height])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(self.h_channels, self.h_channels * 4, 
                          kernel_size=kernel_size, stride=self.stride, padding=self.padding, bias=False),
                nn.LayerNorm([self.h_channels * 4, self.img_width, self.img_height])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(self.h_channels, self.h_channels * 3, 
                          kernel_size=kernel_size, stride=self.stride, padding=self.padding, bias=False),
                nn.LayerNorm([self.h_channels * 3, self.img_width, self.img_height])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(self.h_channels * 2, self.h_channels, 
                          kernel_size=kernel_size, stride=self.stride, padding=self.padding, bias=False),
                nn.LayerNorm([self.h_channels, self.img_width, self.img_height])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(self.input_dim, self.h_channels * 7, 
                          kernel_size=kernel_size, stride=self.stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(self.h_channels, self.h_channels * 4, 
                          kernel_size=kernel_size, stride=self.stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(self.h_channels, self.h_channels * 3, 
                          kernel_size=kernel_size, stride=self.stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(self.h_channels * 2, self.h_channels, 
                          kernel_size=kernel_size, stride=self.stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(self.h_channels * 2, self.h_channels, 
                                   kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.h_channels, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.h_channels, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.h_channels, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m

    
class Pred_RNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, kernel_size, img_width, img_height, 
                 num_layers, layer_norm_flag=True, baseline="last_frame", mask_channel=1):
        """
        Can be initialized both with and without peephole connections. This Conv_LSTM works in a delta prediction 
        fashion, i.e. predicts the deviation to a given baseline.

        Parameters:
            input_dim: Number of channels in input
            output_dim: Number of channels in the output
            hidden_dim: Number of channels in the hidden outputs (should be a number or a list of num_layers - 1)
            kernel_size: Size of kernel in convolutions (Note: Will do same padding)
            img_width: Width of the image in pixels
            img_height: Height of the image in pixels
            layer_norm_flag: Whether to perform layer normalization
            num_layers: Number of PredRNN layers stacked on each other
            baseline: Used for quicker convergence for non-sampling methods, not necessary in this model
            mask_channel: Index of mask channel
        Input:
            A tensor of shape (b, c, w, h, t), step 1~t
        Output:
            A tensor of shape (b, c, w, h, t), step 2~t+1
        """
        super(Pred_RNN, self).__init__()

        self.num_layers = num_layers
        self.h_channels = self._extend_for_multilayer(hidden_dims, num_layers)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.baseline = baseline
        self.mask_channel = mask_channel
              
        cell_list = []

        for i in range(num_layers):
            in_channel = self.input_dim if i == 0 else self.h_channels[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, self.h_channels[i], kernel_size, img_width, img_height, layer_norm_flag)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.h_channels[num_layers - 1], self.output_dim, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        # shared adapter
        adapter_num_hidden = self.h_channels[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)

    def forward(self, input_tensor, input_flag, context_count=8, future_count=4, sampling=None):
        """
        Parameters
        ----------
        input_tensor:
            (b - batch_size, h - height, w - width, c - channel, t - time)
            5-D Tensor either of shape (b, c, w, h, t)
        non_pred_feat:
            non-predictive features for future frames
        baseline:
            baseline computed on the input variables. Only needed for prediction_count > 1.
        Returns
        -------
        prediction
        """
        batch, _, width, height, T = input_tensor.size()
        input_flag = torch.from_numpy(input_flag[:batch,...]).to(self._get_device(), dtype=torch.float)
        
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        
        non_pred_feat = input_tensor[:, self.mask_channel + 1:, :, :, :]
        non_pred_feat = torch.cat((torch.zeros((batch, 1, width, height, T), device=input_tensor.device),
                          non_pred_feat), dim = 1)

        for i in range(self.num_layers):
            initial = torch.zeros([batch, self.h_channels[i], height, width]).to(self._get_device())
            h_t.append(initial)
            c_t.append(initial)
            delta_c_list.append(initial)
            delta_m_list.append(initial)

        memory = torch.zeros([batch, self.h_channels[0], height, width]).to(self._get_device())
        
        preds = torch.zeros((batch, self.output_dim, height, width, context_count + future_count), device = self._get_device())
        
        for t in range(context_count + future_count):
            
            if sampling == 'rs':
                # reverse schedule sampling
                if t == 0:
                    flag_tensor = input_tensor[..., t]
                elif t <= T-2:
                    flag_tensor = input_flag[..., t - 1] * input_tensor[..., t] + (1 - input_flag[..., t - 1]) * prev
                else:
                    flag_tensor = prev
                    
            elif sampling == 'ss':
                # schedule sampling
                if t < context_count:
                    flag_tensor = input_tensor[..., t]
                elif t <= T-2:
                    flag_tensor = input_flag[..., t - context_count] * input_tensor[..., t] + \
                          (1 - input_flag[..., t - context_count]) * prev
                else:
                    flag_tensor = prev
            else:
                if t < context_count:
                    flag_tensor = input_tensor[..., t]
                else:
                    flag_tensor = prev
                                                    
            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](flag_tensor, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
                
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            preds[..., t] = x_gen
            if t < context_count + future_count-1:
                prev = torch.cat((preds[..., t], non_pred_feat[..., t+1]), axis=1)
                       
        return preds

    def _get_device(self):
        return self.cell_list[0].conv_last.weight.device

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                isinstance(kernel_size, int) or
                # lists are currently not supported for Peephole_Conv_LSTM
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, rep):
        if not isinstance(param, list):
            if rep > 0:
                param = [param] * rep
            else:
                return []
        return param