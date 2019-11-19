import torch
import torchvision
resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.eval()
example_inputs = torch.rand(1, 3, 224, 224)
resnet18_traced = torch.jit.trace(resnet18, example_inputs = example_inputs)
resnet18_traced.save("resnet18_traced.pt")