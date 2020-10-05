import torch
import torch.nn as nn
from torch.optim import Optimizer

class OptimActivation(nn.Module):
	def __init__(self):
		super(OptimActivation, self).__init__()
		self.params = nn.Parameter(torch.zeros(5))
	def forward(self, x):
		params_softmax = torch.softmax(self.params, dim=-1)

		x_swish = (torch.sigmoid(x) * x) * params_softmax[0]
		x_sine = torch.sin(x) * params_softmax[1]
		x_linear = x * params_softmax[2]
		x_sigmoid = (torch.sigmoid(x) * params_softmax[3])
		x_tanh = (torch.tanh(x)) * params_softmax[4]

		return x_swish + x_sine + x_linear

class OptimAttn(nn.Module):
	def __init__(self, size, out_size):
		super(OptimAttn, self).__init__()

		self.query_linear = nn.Linear(size, size, bias=False)
		self.key_linear = nn.Linear(size, size, bias=False)
		self.value_linear = nn.Linear(size, size, bias=False)

		self.activation = OptimActivation()

		self.fc_linear = nn.Linear(size, size)
		self.norm = nn.LayerNorm((size,))

		self.fc_linear2 = nn.Linear(size, out_size)
		self.norm2 = nn.LayerNorm((out_size,))
	def forward(self, x):
		query = self.query_linear(x).unsqueeze(-1)
		key = self.key_linear(x).unsqueeze(-1)
		value = self.value_linear(x).unsqueeze(-1)
		
		attended = torch.matmul(query, key.transpose(1,2))
		attended = torch.softmax(torch.flatten(attended, start_dim=1), dim=-1).view_as(attended)
		valued = torch.matmul(attended, value)
		valued = valued.squeeze(-1)

		fc = self.activation(valued)

		fc = self.fc_linear(fc)
		fc = self.norm(fc)
		fc = self.fc_linear2(fc)
		fc = self.norm2(fc)
		return fc

class OptimNet(torch.nn.Module):
	def __init__(self, params):
		super(OptimNet, self).__init__()
		self.params = params.size()[-1]

		self.linear_h = OptimAttn(self.params*2, self.params*2)
		self.linear_h2 = OptimAttn(self.params*2, self.params*2)
		self.linear_c = OptimAttn(self.params*2, 1)
		self.linear_a = OptimAttn(self.params*2, self.params)

		self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

		self.hidden = torch.zeros((1,self.params*2))


	def forward(self, p_grad, is_forget=True):
		z = self.hidden
		h = self.linear_h(p_grad)
		h = self.linear_h2(h + z)

		c = self.linear_c(h)
		a = self.linear_a(h)

		if is_forget:
			a = torch.sigmoid(a)

		return a, c, h

	def optimize(self, loss, p_grad, is_forget=True):
		self.optimizer.zero_grad()
		p_grad = p_grad.detach()
		a, c, h = self.forward(p_grad, is_forget)

		self.hidden = h.detach()

		loss_new = (c) + torch.nn.functional.smooth_l1_loss(c, loss)
		loss_new.backward(retain_graph=True)

		loss_new = loss_new + loss
		self.optimizer.step()


class EMLYN(Optimizer):
	def __init__(self, params, lr=1e-2, betas=0.99):
		defaults = dict(lr=lr, betas=betas)
		super(EMLYN, self).__init__(params, defaults)
	def step(self, loss):
		for group in self.param_groups:
			for p in group['params']:

				grad_original_shape = p.grad.data.shape

				grad = p.grad.data.flatten().unsqueeze(0)
				parameters = p.data.flatten().unsqueeze(0)
				p_g = torch.cat([grad, parameters], dim=-1)

				state = self.state[p]
				if len(state) == 0:
					state['forgnet'] = OptimNet(params=parameters)
					state['net'] = OptimNet(params=parameters)
					state['neta'] = OptimNet(params=parameters)

					state['momentum_buffer'] = torch.zeros_like(parameters)

				net = state['net']
				forgnet = state['forgnet']
				neta = state['neta']
				momentum_buffer = state['momentum_buffer']

				with torch.enable_grad():
					new_grad = net(p_g, False)[0]
					new_forget = forgnet(p_g)[0]
					new_neta = neta(p_g)[0]

					net.optimize(loss, p_g)
					forgnet.optimize(loss, p_g)
					neta.optimize(loss, p_g)

				grad = new_grad + grad
				momentum_buffer = new_neta * momentum_buffer + (1-new_neta) * grad

				grad = momentum_buffer

				grad *= new_forget
				grad *= group['lr']
				grad = grad.view(grad_original_shape)

				p.data = p.data - grad