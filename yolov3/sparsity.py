# additional subgradient descent on the sparsity-induced penalty term
# x_{k+1} = x_{k} - \alpha_{k} * g^{k}
def updateBN(scale, model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(scale*torch.sign(m.weight.data))  # L1
