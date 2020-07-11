import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy
class Denormalization:
    def __init__(self):
        self.mean_3 = torch.tensor([-1.0, -1.0, -1.0]).to(device)
        self.std_3 = torch.tensor([1 / 0.5, 1 / 0.5, 1 / 0.5]).to(device)
        self.mean_1 = torch.tensor([-1.0]).to(device)
        self.std_1 = torch.tensor([1 / 0.5]).to(device)

        self.mean_norm = torch.tensor([0.485,0.456,0.406]).to(device)
        self.std_norm = torch.tensor([0.229,0.224,0.225]).to(device)
    def __call__(self, inp_tensor):
        #print(inp_tensor)
        if inp_tensor.shape[1] == 3:
            #inp_tensor.sub_(self.mean_3[None, :, None, None]).div_(self.std_3[None, :, None, None])
            inp_tensor_denorm = (inp_tensor - self.mean_3[None, :, None, None])/(self.std_3[None, :, None, None])
        elif inp_tensor.shape[1] == 1:
            #inp_tensor.sub_(self.mean_1[None, :, None, None]).div_(self.std_1[None, :, None, None])
            inp_tensor_denorm = (inp_tensor - self.mean_1[None, :, None, None]) / (self.std_1[None, :, None, None])
        #print(inp_tensor)
        #inp_tensor.sub_(self.mean_norm[None, :, None, None]).div_(self.std_norm[None, :, None, None])
        inp_tensor_renorm = (inp_tensor_denorm - self.mean_norm[None, :, None, None])/(self.std_norm[None, :, None, None])
        return inp_tensor_renorm
