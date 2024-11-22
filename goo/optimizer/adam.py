from torch.optim import AdamW

def PytorchAdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, **kwargs):
	return AdamW(params, lr, betas, eps, weight_decay)
	