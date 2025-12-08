import torch.nn as nn
from model.model_hyena import HyenaDNAModel
import json


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        config = json.load(open("./model_save/config.json"))
        self.model = HyenaDNAModel(**config, use_head=False)
    def forward(self, dna):
        dna_embed = self.get_dna(dna)
        return dna_embed
    def get_dna(self,dna_inputs):
        dna_embed = self.model(dna_inputs)[:,-1,:]
        return dna_embed
    def get_dna2(self,dna):
        dna2 = torch.flip(dna,dims = [1])
        dna2[:,0] = dna[:,0]
        dna2[:,-1] = dna[:,-1]
        dna_embed1 = self.model(dna)[:,-1,:]
        dna_embed2 = self.model(dna2)[:,-1,:]
        dna_embed = (dna_embed1 + dna_embed2) / 2
        return dna_embed
    

        
        
