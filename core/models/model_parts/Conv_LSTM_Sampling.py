import torch.nn as nn
import torch

class Conv_LSTM_Cell(nn.Module):
    def __init__(self, input_dim, h_channels, big_mem, kernel_size, memory_kernel_size, dilation_rate, layer_norm_flag, img_width, img_height, mask_channel, peephole):
        
        super(Conv_LSTM_Cell, self).__init__()

        self.input_dim = input_dim
        self.h_channels = h_channels
        self.c_channels = h_channels if big_mem else 1
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.layer_norm_flag = layer_norm_flag
        self.img_width = img_width
        self.mask_channel = mask_channel
        self.img_height = img_height
        self.peephole = peephole

        self.conv_cc = nn.Conv2d(self.input_dim + self.h_channels, self.h_channels + 3*self.c_channels, dilation=dilation_rate, kernel_size=kernel_size,
                                     bias=True, padding='same', padding_mode='reflect')
        if self.peephole:
            self.conv_ll = nn.Conv2d(self.c_channels, self.h_channels + 2*self.c_channels, dilation=dilation_rate, kernel_size=memory_kernel_size,
                                        bias=False, padding='same', padding_mode='reflect')
        
        if self.layer_norm_flag:
            self.layer_norm = nn.InstanceNorm2d(self.input_dim + self.h_channels, affine=True)
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        # apply layer normalization
        if self.layer_norm_flag:
            combined = self.layer_norm(combined)

        combined_conv = self.conv_cc(combined) # h_channel + 3 * c_channel 

        cc_i, cc_f, cc_g, cc_o = torch.split(combined_conv, [self.c_channels, self.c_channels, self.c_channels, self.h_channels], dim=1)
        if self.peephole:
            combined_memory = self.conv_ll(c_cur)  # h_channel + 2 * c_channel  # NO BIAS HERE
            ll_i, ll_f, ll_o = torch.split(combined_memory, [self.c_channels, self.c_channels, self.h_channels], dim=1)

            i = torch.sigmoid(cc_i + ll_i)
            f = torch.sigmoid(cc_f + ll_f)
            o = torch.sigmoid(cc_o + ll_o)
        else:
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o) 

        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        if self.h_channels == self.c_channels:
            h_next = o * torch.tanh(c_next)
        elif self.c_channels == 1:
            h_next = o * torch.tanh(c_next).repeat([1, self.h_channels, 1, 1])

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.h_channels, height, width, device=self.conv_cc.weight.device),  
                torch.zeros(batch_size, self.c_channels, height, width, device=self.conv_cc.weight.device))


class Conv_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, big_mem, kernel_size, memory_kernel_size, dilation_rate, 
                 img_width, img_height, layer_norm_flag=True, num_layers=1, baseline="last_frame", mask_channel = 1,
                 peephole=True):
        """
        Parameters:
            input_dim: Number of channels in input
            output_dim: Number of channels in the output
            hidden_dim: Number of channels in the hidden outputs (should be a number or a list of num_layers - 1)
            kernel_size: Size of kernel in convolutions (Note: Will do same padding)
            memory_kernel_size: Size of kernel in convolutions when the memory influences the output
            dilation_rate: Size of holes in convolutions
            img_width: Width of the image in pixels
            img_height: Height of the image in pixels
            layer_norm_flag: Whether to perform layer normalization
            num_layers: Number of LSTM layers stacked on each other
            baseline: Used for quicker convergence for non-sampling methods, not necessary in this model
            mask_channel: Index of mask channel
            peephole: Whether to include peephole connections or not
        Input:
            A tensor of shape (b, c, w, h, t), step 1~t
        Output:
            A tensor of shape (b, c, w, h, t), step 2~t+1
        """
        super(Conv_LSTM, self).__init__()
        self._check_kernel_size_consistency(kernel_size)

        self.input_dim = input_dim                                                  # n of channels in input pics
        self.output_dim = output_dim                                                # n of channels in output pics
        self.h_channels = self._extend_for_multilayer(hidden_dims, num_layers - 1)  # n of hidden channels   
        self.h_channels.append(output_dim)                                          # n of channels in output pics
        self.big_mem = big_mem                                                      # true means c = h, false c = 1. 
        self.num_layers = num_layers                                                # n of channels that go through hidden layers
        self.kernel_size = kernel_size     
        self.memory_kernel_size = memory_kernel_size                                # n kernel size (no magic here)
        self.dilation_rate = dilation_rate
        self.layer_norm_flag = layer_norm_flag
        self.img_width = img_width
        self.img_height = img_height
        self.mask_channel = mask_channel
        self.peephole = peephole

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.h_channels[i - 1]
            cur_layer_norm_flag = self.layer_norm_flag if i != 0 else False

            cell_list.append(Conv_LSTM_Cell(input_dim=cur_input_dim,
                                            h_channels=self.h_channels[i],
                                            big_mem=self.big_mem,
                                            layer_norm_flag=cur_layer_norm_flag,
                                            img_width=self.img_width,
                                            img_height=self.img_height,
                                            kernel_size=self.kernel_size,
                                            memory_kernel_size=self.memory_kernel_size,
                                            dilation_rate=dilation_rate,
                                            mask_channel = self.mask_channel,
                                            peephole=self.peephole))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, input_flag, context_count=8, future_count=4, sampling=None):
        """
        Parameters
        ----------
        input_tensor:
            (b - batch_size, h - height, w - width, c - channel, t - time)
            5-D Tensor either of shape (b, c, w, h, t)
        non_pred_feat:
            non-predictive features for future frames
        Returns
        -------
        prediction
        """
        batch, _, width, height, T = input_tensor.size()
        input_flag = torch.from_numpy(input_flag[:batch,...]).to(self._get_device(), dtype=torch.float)
        hs = []
        cs = []
        
        non_pred_feat = input_tensor[:, self.mask_channel + 1:, :, :, :]
        non_pred_feat = torch.cat((torch.zeros((batch, 1, width, height, T), device=input_tensor.device),
                          non_pred_feat), dim = 1)

        for i in range(self.num_layers):
            h, c = self.cell_list[i].init_hidden(batch,(height,width))
            hs.append(h)
            cs.append(c)

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
        

            # glue together with non_pred_data
            hs[0], cs[0] = self.cell_list[0](input_tensor=flag_tensor, cur_state=[hs[0], cs[0]])
            for i in range(1, self.num_layers):
                hs[i], cs[i] = self.cell_list[i](input_tensor=hs[i-1], cur_state=[hs[i], cs[i]])
            
            x_gen = hs[-1]
            preds[..., t] = x_gen
            if t < context_count + future_count-1:
                prev = torch.cat((preds[..., t], non_pred_feat[..., t+1]), axis=1)

        return preds

    def _get_device(self):
        return self.cell_list[0].conv_cc.weight.device

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