import numpy as np
import torch
import torch.nn as nn
import torch.fft

class GRUModel(nn.Module):
    """GRU模型定义"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# TCN残差块
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=(kernel_size - 1) * dilation // 2, 
                               dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation // 2, 
                               dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)

# TCN模型
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, 
                              dilation=dilation, dropout=dropout)]
            
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        x = x.transpose(1, 2)  # [batch, channels, sequence]
        out = self.network(x)
        out = out[:, :, -1]  # 取最后一个时间步
        return self.linear(out)
    

# 注意力机制
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(seq_len, 1, 1).permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

# 编码器
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x):
        outputs, hidden = self.gru(x)
        return outputs, hidden

# 解码器
class Decoder(nn.Module):
    def __init__(self, input_size,output_size, hidden_size, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(self.input_size + hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x, hidden, encoder_outputs):
        # 注意力计算
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        # print(f"context shape: {context.shape}")  # [batch_size, seq_len, input_size]
        # print(f"x0 shape: {x.shape}")  # [batch_size, seq_len, input_size]
        # 解码步骤
        x = torch.cat((x, context.unsqueeze(1)), dim=2)
        # print(f"x shape: {x.shape}")  # [batch_size, seq_len, input_size]
        output, hidden = self.gru(x, hidden)
        output = torch.cat((output.squeeze(1), context), dim=1)
        prediction = self.fc(output)
        return prediction, hidden

# Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self,input_size,hidden_dim,output_size):
        super().__init__()
        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_dim, num_layers=2)
        self.decoder = Decoder(input_size=input_size,output_size=output_size, hidden_size=hidden_dim, num_layers=2)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # print(f"src shape: {src.shape}")  # [batch_size, seq_len, input_size]
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        outputs = torch.zeros(batch_size, trg_len, 1).to(self.device)
        
        # 编码
        encoder_outputs, hidden = self.encoder(src)
        # print(f"trg shape: {trg.shape}")  # [batch_size, seq_len, input_size]
        # 初始解码器输入
        decoder_input = src[:, -1, :].unsqueeze(1)
        # print(f"decoder_input shape: {decoder_input.shape}")  # [batch_size, seq_len, input_size]
        
        for t in range(trg_len):
            decoder_output, hidden = self.decoder(
                decoder_input, hidden, encoder_outputs
            )
            outputs[:, t, :] = decoder_output
            # 使用教师强制或预测值作为下一个输入
            use_teacher_forcing = np.random.random() < teacher_forcing_ratio
            decoder_input = trg[:, :].unsqueeze(1) if use_teacher_forcing else decoder_output.unsqueeze(1)
            
        return outputs.squeeze(2)  # [batch_size, trg_len, 1] -> [batch_size, trg_len]
    

# 周期检测与2D转换模块
class TimesBlock(nn.Module):
    def __init__(self, d_model, num_kernels=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_kernels = num_kernels
        
        # 1D -> 2D转换参数
        self.conv2d = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
                nn.ReLU(),
                nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1)
            ) for _ in range(num_kernels)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        B, T, N = x.shape
        
        # FFT检测周期
        x_freq = torch.fft.rfft(x, dim=1)
        freqs = torch.tensor(np.fft.rfftfreq(T))
        amplitudes = torch.abs(x_freq)
        top_k = 3  # 选择前3个显著周期
        
        # 选择显著周期
        _, top_indices = torch.topk(amplitudes.mean(dim=(0,2)), top_k)
        selected_periods = (1 / freqs[top_indices]).long()
        
        # 多周期处理
        representations = []
        for period in selected_periods:
            if period <= 1:
                continue
                
            # 序列重塑为2D张量
            if T % period.item() != 0:
                length = (T // period.item()) * period.item()
                x_pad = x[:, :length]
            else:
                x_pad = x
                
            x_2d = x_pad.reshape(B, period, -1, N).permute(0, 3, 1, 2)
            
            # 2D卷积处理
            conv_out = 0
            for conv in self.conv2d:
                conv_out += conv(x_2d)
            conv_out = conv_out / self.num_kernels
            
            # 恢复1D序列
            x_1d = conv_out.permute(0, 2, 3, 1).reshape(B, -1, N)
            
            # 对齐长度
            if x_1d.shape[1] < T:
                x_1d = torch.cat([x_1d, x[:, x_1d.shape[1]:]], dim=1)
                
            representations.append(x_1d)
        
        # 动态融合
        rep_stack = torch.stack(representations, dim=-1)
        fusion_weights = torch.softmax(torch.mean(rep_stack, dim=(1,2)), dim=-1)
        fused = torch.einsum('btnk,bk->btn', rep_stack, fusion_weights)
        
        # 残差连接
        output = self.norm(x + self.dropout(fused))
        return output

# TimesNet模型架构
class TimesNet(nn.Module):
    def __init__(self, pred_len, d_model=64,input_features=6, num_layers=3, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        
        # 输入嵌入
        self.embed = nn.Linear(input_features, d_model)
        
        # 多层TimesBlock
        self.model = nn.ModuleList([
            TimesBlock(d_model, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 预测头
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, pred_len)
        )
        
    def forward(self, x):
        # 输入形状: [B, L, 1]
        x = self.embed(x)  # [B, L, d_model]
        
        # 多层处理
        for layer in self.model:
            x = layer(x)
            
        # 取最后时间步预测
        last = x[:, -1]
        output = self.head(last)
        return output



