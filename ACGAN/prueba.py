import torch.nn as nn
import numpy as np
import torch

auxiliary_loss = nn.CrossEntropyLoss().cuda()

a = np.zeros(10)
a.fill(1)
b = np.zeros(10)
c = [0,0,0,0,0,1,1,1,1,1]
n1 = torch.FloatTensor(10, 2).cuda()
n2 = torch.LongTensor(10).cuda()
n3 = torch.LongTensor(10).cuda()

p1 = np.zeros((10, 2))
p1[np.arange(10), a.astype(int)] = 1
p1 = torch.from_numpy(p1)

p2 = np.zeros((10, 2))
p2[np.arange(10), b.astype(int)] = 1

a1 = torch.from_numpy(a)
b1 = torch.from_numpy(b)
n1.data.copy_(p1.view(10,2))
n2.data.copy_(a1.view(10))
n3.data.copy_(b1.view(10))
c1 = torch.cuda.LongTensor(c)

e1 = auxiliary_loss(n1,n2)
e2 = auxiliary_loss(n1,n3)
e3 = auxiliary_loss(n1,c1)

print('e1 = ' +  str(e1.item()))
print('e2 = ' +  str(e2.item()))
print('e3 = ' + str(e3.item()))
