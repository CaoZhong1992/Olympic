import torch
import torch.nn as nn
import numpy as np
import math



class SelfAttentionLayer(nn.Module):
    """
    Self-attention layer. no scale_factor d_k
    """

    def __init__(self, in_channels, global_graph_width):
        super(SelfAttentionLayer, self).__init__()
        self.in_channels = in_channels
        
        self.q_lin = nn.Linear(in_channels, global_graph_width)
        self.k_lin = nn.Linear(in_channels, global_graph_width)
        self.v_lin = nn.Linear(in_channels, global_graph_width)

        self._norm_fact = 1 / math.sqrt(in_channels)


    def forward(self, x):

        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)
        scores = torch.bmm(query, key.transpose(1, 2)) * self._norm_fact
        scores = nn.functional.softmax(scores, dim=-1)
        
        return torch.bmm(scores,value)
    
def solve():
    # vehicle x,y,vx,vy
    # ego_vehicle, front_vehicle, three closest vehicles
    x = np.array([[1,2,3,4],[-1,-1,-1,-2],[0,0,0,0]])
    x = torch.as_tensor(x).float().unsqueeze(0)
    print(x)

    model = SelfAttentionLayer(4, 6)
    y = model(x)
    print(y)

    
if __name__ == '__main__':
    
    solve()