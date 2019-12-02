import torch
from UDA.model_UDA import Model
from DANN.model_DANN import CNNModel
from DANN.test_DANN import test as testD
from UDA.test_UDA import test as testU
from UDA.parser_2 import arg_parse

if __name__ == '__main__':
    #load args
    args = arg_parse()

    m_UDA = Model().cuda()
    m_DANN = CNNModel().cuda()
    m_DANN.eval()
    m_UDA.eval()

    m1u = 'model_UDA_mnistm-svhn.pth.tar'
    m2u = 'model_UDA_svhn-mnistm.pth.tar'
    m1d = 'model_DANN_mnistm-svhn.pth.tar'
    m2d = 'model_DANN_svhn-mnistm.pth.tar'

    m_UDA.load_state_dict(torch.load(m2u))
    m_DANN.load_state_dict(torch.load(m2d))

    p1 = testU('mnistm', 1, args, m_UDA)
    p2 = testD('mnistm', 1, args, m_DANN)

