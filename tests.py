import torch
from optim import EMLYN

def rosenbrock(tensor):
	x, y = tensor
	return (1-x) ** 2 + 1 * (y - x ** 2) ** 2

def rastrigin(tensor):
	x, y = tensor
	tau = 2 * 3.1415926535897932
	return (x**2 - 10 * torch.cos(tau * x)) + \
		   (y**2 - 10 * torch.cos(tau * y)) + 20

def optimize_emlyn(optim):
	lr = 1e-3
	state = (2.0, 2.0)
	loc_min = (1, 1)
	x = torch.Tensor(state).requires_grad_(True)

	optim = optim([x])
	for _ in range(800):
		optim.zero_grad()
		y = rosenbrock(x)
		y.backward(retain_graph=True)
		optim.step(y)
		print(y.clone().detach().numpy())

def optimize_other(optim):
	lr = 1e-3
	state = (2.0, 2.0)
	loc_min = (1, 1)
	x = torch.Tensor(state).requires_grad_(True)

	optim = optim([x])
	for _ in range(800):
		optim.zero_grad()
		y = rosenbrock(x)
		y.backward(retain_graph=True)
		optim.step()
		print(y.clone().detach().numpy())


optimize_other(torch.optim.Adam)
