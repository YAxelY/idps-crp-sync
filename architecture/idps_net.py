import torch
import torch.nn as nn
from .resnet import resnet18
from .transformer import Transformer
from .dps_module import PerturbedTopK

class IDPSNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.device = conf.device
        self.M = conf.M  # Number of Scouted patches (Candidates for Pass 2)
        self.K = conf.K  # Number of Final patches selected by DPS (K < M)
        self.D = conf.D 
        self.n_class = conf.n_class
        
        # Shared Backbone
        self.encoder = resnet18(pretrained=conf.pretrained, num_channels=conf.n_chan_in, flatten=True)
        self.projector = nn.Linear(512, self.D)

        # Scorer / Aggregator
        # Used for computing "Importance Scores" in Pass 2
        self.scorer_head = nn.Linear(self.D, 1) 
        
        # DPS Selector
        self.dps_topk = PerturbedTopK(k=self.K, num_samples=500, sigma=0.05)
        
        # Aggregator (Transformer) runs on the K selected patches
        self.transf = Transformer(conf.n_token, conf.H, conf.D, conf.D_k, conf.D_v, conf.D_inner, conf.attn_dropout, conf.dropout)
        
        # Head
        self.head = nn.Linear(self.D, self.n_class)

    def _get_embeddings(self, patches):
        B_dim = -1
        if patches.dim() == 5:
            B, N, C, H, W = patches.shape
            B_dim = B
            patches = patches.reshape(-1, C, H, W)
        
        feats = self.encoder(patches) 
        embeddings = self.projector(feats) 
        
        if B_dim != -1:
            embeddings = embeddings.view(B_dim, N, -1)
        
        return embeddings

    @torch.no_grad()
    def scouting_pass(self, wsi_patches):
        """
        Pass 1: The Scout (Cheap/Heuristic)
        Output: top_m_indices (B, M) - Candidates for Pass 2
        """
        self.eval()
        embeddings = self._get_embeddings(wsi_patches)
        # Simple attention or scoring
        scores = self.scorer_head(embeddings).squeeze(-1) # (B, N)
        top_m_indices = torch.topk(scores, self.M, dim=-1)[1]
        return top_m_indices

    def training_pass(self, candidate_patches):
        """
        Pass 2: The Learner (with DPS)
        Input: candidate_patches (B, M, C, H, W) - The pool of likely candidates
        Process:
           1. Re-Embed M candidates (Grad ON).
           2. Compute scores from embeddings.
           3. DPS Select K from M (Differentiable).
           4. Aggregate K patches.
        """
        self.train() 
        B, M, C, H, W = candidate_patches.shape
        
        # 1. Embed Candidates
        embeddings = self._get_embeddings(candidate_patches) # (B, M, D)
        
        # 2. Score Candidates (Differentiable)
        scores = self.scorer_head(embeddings).squeeze(-1) # (B, M)
        
        # 3. DPS Selection (Soft Top-K Indicators)
        # indicators: (B, K, M) - One-hot-ish matrix pointing to selected K
        indicators = self.dps_topk(scores) 
        
        # 4. Soft Gather: Weighted Sum to get K "Selected" Embeddings
        # (B, K, M) @ (B, M, D) -> (B, K, D)
        selected_embeddings = torch.matmul(indicators, embeddings)
        
        # 5. Aggregate (Transformer)
        # Output is (B, n_token, D). with n_token=1, it is (B, 1, D)
        slide_embedding = self.transf(selected_embeddings).squeeze(1) # (B, D)
        
        # 6. Classify
        logits = self.head(slide_embedding)
        
        return logits

    def forward(self, x, mode='train'):
        """
        Convenience wrapper.
        Real training loop will call scouting_pass and training_pass explicitly.
        """
        if mode == 'scout':
            return self.scouting_pass(x)
        elif mode == 'learn':
            return self.training_pass(x)
        else:
            raise ValueError("Use explicit scouting_pass or training_pass")
