import torch
import torch.nn.functional as F
# 定义两个矩阵
x = torch.randn((4, 5))
y = torch.randn((4, 5))
# 因为要用y指导x,所以求x的对数概率，y的概率
logp_x = F.log_softmax(x, dim=-1)
p_y = F.softmax(y, dim=-1)
 
 
kl_sum = F.kl_div(logp_x, p_y, reduction='sum')
kl_mean = F.kl_div(logp_x, p_y, reduction='mean')
kl_batchmean = F.kl_div(logp_x, p_y, reduction='batchmean')
 
print(kl_sum, kl_mean,kl_batchmean)

a = torch.tensor([[0.,0,0.000001,0.0000002,0,0.0000007,0]])
log_a =F.log_softmax(a,dim=-1)
print("log_a:",log_a)

b = torch.tensor([[0.,0,0.000001,0.0000002,0,0.0000007,0]])
softmax_b =F.softmax(b,dim=-1)
print("softmax_b:",softmax_b)

kl_mean = F.kl_div(log_a, softmax_b,reduction='mean')
print("kl_mean:",kl_mean)