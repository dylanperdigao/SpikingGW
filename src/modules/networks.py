import snntorch as snn
import torch
from torch import nn
from snntorch import surrogate

class FFSNN(nn.Module):
    """
    Feed-Forward Spiking Neural Network (SNN) without Global Workspace Theory (GWT) integration.
    """
    def __init__(self, input_size: int, betas: list, thresholds: list, slopes: list, learn_betas: bool = False, learn_thresholds: bool = False, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.num_classes = 2
        
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.lif1 = snn.Leaky(beta=betas[0], threshold=thresholds[0], spike_grad=surrogate.fast_sigmoid(slope=slopes[0]), learn_beta=learn_betas, learn_threshold=learn_thresholds)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.lif2 = snn.Leaky(beta=betas[1], threshold=thresholds[1], spike_grad=surrogate.fast_sigmoid(slope=slopes[1]), learn_beta=learn_betas, learn_threshold=learn_thresholds)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.lif3 = snn.Leaky(beta=betas[2], threshold=thresholds[2], spike_grad=surrogate.fast_sigmoid(slope=slopes[2]), learn_beta=learn_betas, learn_threshold=learn_thresholds)

        self.fc4 = nn.Linear(32, self.num_classes)

        self.lif4 = snn.Leaky(beta=betas[3], threshold=thresholds[3], spike_grad=surrogate.fast_sigmoid(slope=slopes[3]), learn_beta=learn_betas, learn_threshold=False)

    def forward(self, x, steps: int):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk4_rec = []
        mem4_rec = []

        for _ in range(steps):
            cur1 = self.bn1(self.fc1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.bn2(self.fc2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.bn3(self.fc3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            
            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            
            spk4_rec.append(spk4)
            mem4_rec.append(mem4)

        return torch.stack(spk4_rec, dim=0), torch.stack(mem4_rec, dim=0)

class FFSNN_GWT(nn.Module):
    """
    Feed-Forward Spiking Neural Network (SNN) with Global Workspace Theory (GWT) integration.
    """
    def __init__(self, input_size: int, betas: list, thresholds: list, slopes: list, learn_betas: bool = False, learn_thresholds: bool = False, device=torch.device('cpu'), gw_threshold=1.5):
        super().__init__()
        self.device = device
        self.num_classes = 2
        
        self.idx_A = [0, 2, 3, 4, 6, 7, 14, 15, 17, 18, 19, 20, 21, 22] # 14 Features (Perfil)
        self.idx_B = [1, 5, 8, 9, 10, 11, 12, 13, 16, 23, 24, 25, 26, 27, 28, 29, 30] # 17 Features (Comportamento)
        
        # --- MODULE A (Unconscious Audience A) ---
        self.fcA1 = nn.Linear(14, 64)
        self.bnA1 = nn.BatchNorm1d(64)
        self.lifA1 = snn.Leaky(beta=betas[0], threshold=thresholds[0], spike_grad=surrogate.fast_sigmoid(slope=slopes[0]), learn_beta=learn_betas, learn_threshold=learn_thresholds)
        
        self.fcA2 = nn.Linear(64, 32)
        self.bnA2 = nn.BatchNorm1d(32)
        self.lifA2 = snn.Leaky(beta=betas[1], threshold=thresholds[1], spike_grad=surrogate.fast_sigmoid(slope=slopes[1]), learn_beta=learn_betas, learn_threshold=learn_thresholds)

        # --- MODULE B (Unconscious Audience B) ---
        self.fcB1 = nn.Linear(17, 64)
        self.bnB1 = nn.BatchNorm1d(64)
        self.lifB1 = snn.Leaky(beta=betas[0], threshold=thresholds[0], spike_grad=surrogate.fast_sigmoid(slope=slopes[0]), learn_beta=learn_betas, learn_threshold=learn_thresholds)
        
        self.fcB2 = nn.Linear(64, 32)
        self.bnB2 = nn.BatchNorm1d(32)
        self.lifB2 = snn.Leaky(beta=betas[1], threshold=thresholds[1], spike_grad=surrogate.fast_sigmoid(slope=slopes[1]), learn_beta=learn_betas, learn_threshold=learn_thresholds)

        # --- GLOBAL WORKSPACE ---
        self.fc_gw = nn.Linear(32 + 32, 64)
        self.bn_gw = nn.BatchNorm1d(64)
        self.lif_gw = snn.Leaky(beta=betas[2], threshold=gw_threshold, spike_grad=surrogate.fast_sigmoid(slope=slopes[2]), learn_beta=False, learn_threshold=False)
        
        # --- OUTPUT ---
        self.fc_out = nn.Linear(64, self.num_classes)
        self.lif_out = snn.Leaky(beta=betas[3], threshold=thresholds[3], spike_grad=surrogate.fast_sigmoid(slope=slopes[3]), learn_beta=False, learn_threshold=False)

    def forward(self, x, steps: int):
        memA1, memA2 = self.lifA1.init_leaky(), self.lifA2.init_leaky()
        memB1, memB2 = self.lifB1.init_leaky(), self.lifB2.init_leaky()
        mem_gw = self.lif_gw.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec, mem_out_rec = [], []

        for _ in range(steps):
            xA, xB = x[:, self.idx_A], x[:, self.idx_B]
            
            spkA1, memA1 = self.lifA1(self.bnA1(self.fcA1(xA)), memA1)
            spkA2, memA2 = self.lifA2(self.bnA2(self.fcA2(spkA1)), memA2)
            
            spkB1, memB1 = self.lifB1(self.bnB1(self.fcB1(xB)), memB1)
            spkB2, memB2 = self.lifB2(self.bnB2(self.fcB2(spkB1)), memB2)
            
            gw_in = torch.cat((spkA2, spkB2), dim=1)
            spk_gw, mem_gw = self.lif_gw(self.bn_gw(self.fc_gw(gw_in)), mem_gw)
            
            spk_out, mem_out = self.lif_out(self.fc_out(spk_gw), mem_out)
            
            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_out)

        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)


class CSNN(nn.Module):
    """
    Convolutional Spiking Neural Network (SNN) without Global Workspace Theory (GWT) integration.
    """
    def __init__(self, input_size: int, betas: list, thresholds: list, slopes: list, learn_betas: bool = False, learn_thresholds: bool = False, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.num_classes = 2
        
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)
        self.lif1 = snn.Leaky(beta=betas[0], threshold=thresholds[0], spike_grad=surrogate.fast_sigmoid(slope=slopes[0]), learn_beta=learn_betas, learn_threshold=learn_thresholds)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2) 
        self.lif2 = snn.Leaky(beta=betas[1], threshold=thresholds[1], spike_grad=surrogate.fast_sigmoid(slope=slopes[1]), learn_beta=learn_betas, learn_threshold=learn_thresholds)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        self.lif3 = snn.Leaky(beta=betas[2], threshold=thresholds[2], spike_grad=surrogate.fast_sigmoid(slope=slopes[2]), learn_beta=learn_betas, learn_threshold=learn_thresholds)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(64 * 3 if input_size == 31 else 64 * 4, 64)
        self.lif4 = snn.Leaky(beta=betas[3], threshold=thresholds[3], spike_grad=surrogate.fast_sigmoid(slope=slopes[3]), learn_beta=learn_betas, learn_threshold=learn_thresholds)

        self.fc2 = nn.Linear(64, self.num_classes)
        self.lif5 = snn.Leaky(beta=betas[4] if len(betas)>4 else betas[-1], threshold=thresholds[4] if len(thresholds)>4 else thresholds[-1], spike_grad=surrogate.fast_sigmoid(slope=slopes[4] if len(slopes)>4 else slopes[-1]), learn_beta=False,  learn_threshold=False)

    def forward(self, x, steps: int):
        mem1, mem2, mem3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        mem4, mem5 = self.lif4.init_leaky(), self.lif5.init_leaky()

        spk_out_rec, mem_out_rec = [], []

        for _ in range(steps):
            x_seq = x.view(-1, 1, x.shape[-1])
            
            spk1, mem1 = self.lif1(self.pool1(self.bn1(self.conv1(x_seq))), mem1)
            spk2, mem2 = self.lif2(self.pool2(self.bn2(self.conv2(spk1))), mem2)
            spk3, mem3 = self.lif3(self.pool3(self.bn3(self.conv3(spk2))), mem3)
            
            flat = self.flatten(spk3)
            
            spk4, mem4 = self.lif4(self.fc1(flat), mem4)
            spk5, mem5 = self.lif5(self.fc2(spk4), mem5)
            
            spk_out_rec.append(spk5)
            mem_out_rec.append(mem5)

        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)

class CSNN_GWT(nn.Module):
    """
    Convolutional Spiking Neural Network (SNN) with Global Workspace Theory (GWT) integration.
    """
    def __init__(self, input_size: int, betas: list, thresholds: list, slopes: list, learn_betas: bool = False, learn_thresholds: bool = False, device=torch.device('cpu'), gw_threshold=1.5):
        super().__init__()
        self.device = device
        self.num_classes = 2
        
        self.idx_A = [0, 2, 3, 4, 6, 7, 14, 15, 17, 18, 19, 20, 21, 22] # 14 Features
        self.idx_B = [1, 5, 8, 9, 10, 11, 12, 13, 16, 23, 24, 25, 26, 27, 28, 29, 30] # 17 Features
        
        # --- MODULE A (Unconscious Audience A) ---
        self.convA1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bnA1 = nn.BatchNorm1d(16)
        self.poolA1 = nn.MaxPool1d(2) 
        self.lifA1 = snn.Leaky(beta=betas[0], threshold=thresholds[0], spike_grad=surrogate.fast_sigmoid(slope=slopes[0]), learn_beta=learn_betas, learn_threshold=learn_thresholds)
        
        self.convA2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bnA2 = nn.BatchNorm1d(32)
        self.poolA2 = nn.MaxPool1d(2) 
        self.lifA2 = snn.Leaky(beta=betas[1], threshold=thresholds[1], spike_grad=surrogate.fast_sigmoid(slope=slopes[1]), learn_beta=learn_betas, learn_threshold=learn_thresholds)

        # --- MODULE B (Unconscious Audience B) ---
        self.convB1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bnB1 = nn.BatchNorm1d(16)
        self.poolB1 = nn.MaxPool1d(2) 
        self.lifB1 = snn.Leaky(beta=betas[0], threshold=thresholds[0], spike_grad=surrogate.fast_sigmoid(slope=slopes[0]), learn_beta=learn_betas, learn_threshold=learn_thresholds)
        
        self.convB2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bnB2 = nn.BatchNorm1d(32)
        self.poolB2 = nn.MaxPool1d(2) 
        self.lifB2 = snn.Leaky(beta=betas[1], threshold=thresholds[1], spike_grad=surrogate.fast_sigmoid(slope=slopes[1]), learn_beta=learn_betas, learn_threshold=learn_thresholds)

        self.flatten = nn.Flatten()
        
        # --- GLOBAL WORKSPACE ---
        self.fc_gw = nn.Linear(224, 64)
        self.bn_gw = nn.BatchNorm1d(64)
        self.lif_gw = snn.Leaky(beta=betas[2], threshold=gw_threshold, spike_grad=surrogate.fast_sigmoid(slope=slopes[2]), learn_beta=False, learn_threshold=False)
        
        # --- OUTPUT ---
        self.fc_out = nn.Linear(64, self.num_classes)
        self.lif_out = snn.Leaky(beta=betas[3], threshold=thresholds[3], spike_grad=surrogate.fast_sigmoid(slope=slopes[3]), learn_beta=False,  learn_threshold=False)

    def forward(self, x, steps: int):
        memA1, memA2 = self.lifA1.init_leaky(), self.lifA2.init_leaky()
        memB1, memB2 = self.lifB1.init_leaky(), self.lifB2.init_leaky()
        mem_gw = self.lif_gw.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec, mem_out_rec = [], []

        for _ in range(steps):
            xA = x[:, self.idx_A].unsqueeze(1)
            xB = x[:, self.idx_B].unsqueeze(1)
            
            spkA1, memA1 = self.lifA1(self.poolA1(self.bnA1(self.convA1(xA))), memA1)
            spkA2, memA2 = self.lifA2(self.poolA2(self.bnA2(self.convA2(spkA1))), memA2)
            
            spkB1, memB1 = self.lifB1(self.poolB1(self.bnB1(self.convB1(xB))), memB1)
            spkB2, memB2 = self.lifB2(self.poolB2(self.bnB2(self.convB2(spkB1))), memB2)
            
            gw_in = torch.cat((self.flatten(spkA2), self.flatten(spkB2)), dim=1)
            spk_gw, mem_gw = self.lif_gw(self.bn_gw(self.fc_gw(gw_in)), mem_gw)
            
            spk_out, mem_out = self.lif_out(self.fc_out(spk_gw), mem_out)
            
            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_out)

        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)