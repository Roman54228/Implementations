import torchvision
import torchv

def denorm_dict(net_dict, val):
	denormal = []
	for k, v in net_dict.items():
		denorms = (v != 0) & (v.abs() < val)
		if denorms.sum() > 0:
			denormal.append(k)

			v[v.abs() < val] = 0
			net_dict[k] = val
	print(f"{len(denormal)} / {len(net_dict.values())} denormal weights")
	return net_dict

net = torchvision.models.resnet18(pretrained=False)
denorm_dict(net.state_dict(), 1)
