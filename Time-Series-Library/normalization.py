# x_enc: [B, T, C] 随机数字
import torch


x_enc = torch.rand(2,3,4)  # 假设有32个样本，时间序列长度为96，特征维度为16
means = x_enc.mean(1, keepdim=True).detach()
x_enc = x_enc - means
stdev = torch.sqrt(
    torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)

print("means:", means)
print("stdev:", stdev)
x_enc /= stdev  # 归一化处理            
print("x_enc after normalization:", x_enc)