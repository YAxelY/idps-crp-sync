import math
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

from architecture.resnet import resnet18 as custom_resnet18
from architecture.transformer import Transformer, pos_enc_1d
from architecture.dps_module import PerturbedTopK

class IDPSNet(nn.Module):
    """
    Hybrid IDPS-CRP Network:
    - Pass 1 (Scout): Iterative Patch Selection applied on downsampled patches.
    - Pass 2 (Learner): Differentiable Patch Selection (DPS) + High-Res extraction.
    """

    def get_conv_patch_enc(self, enc_type, pretrained, n_chan_in, n_res_blocks):
        # Using built-in resnets for the lightweight scout
        if enc_type == 'resnet18': 
            res_net_fn = resnet18
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        elif enc_type == 'resnet50':
            res_net_fn = resnet50
            weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None        
        else:
            raise ValueError()

        res_net = res_net_fn(weights=weights)

        if n_chan_in == 1:
            res_net.conv1 = nn.Conv2d(n_chan_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        layer_ls = [
            res_net.conv1,
            res_net.bn1,
            res_net.relu,
            res_net.maxpool,
            res_net.layer1,
            res_net.layer2
        ]

        if n_res_blocks == 4:
            layer_ls.extend([res_net.layer3, res_net.layer4])
        
        layer_ls.append(res_net.avgpool)
        return nn.Sequential(*layer_ls)

    def __init__(self, conf):
        super().__init__()

        self.n_class = conf.n_class
        self.M = conf.M # Scout memory limit
        self.K = conf.K # Learner selection limit
        self.I = conf.I # Scout Iteration Chunk Size
        self.D = conf.D 
        self.use_pos = conf.use_pos

        self.dps_sigma = conf.sigma
        self.num_samples = conf.num_samples

        # ---- Shared Scout Components ----
        # Lightweight encoder for downsampled patches
        self.encoder_low = self.get_conv_patch_enc(conf.enc_type, conf.pretrained, conf.n_chan_in, conf.n_res_blocks)
        
        # Transformer Multi-Head Cross-Attention Scorer (Shared between IPS Pass 1 and DPS Pass 2)
        # We reuse the same transformer architecture as IPS for scoring.
        self.scorer = Transformer(conf.n_token, conf.H, conf.D, conf.D_k, conf.D_v, conf.D_inner, conf.attn_dropout, conf.dropout)

        # ---- Learner Specific Components ----
        # DPS Soft Selection Module
        self.dps_topk = PerturbedTopK(k=self.K, num_samples=self.num_samples, sigma=self.dps_sigma)
        
        # Heavy encoder for the K High-Res patches
        # Here we use the custom ResNet implementation from the original code for accurate 1-channel support
        self.encoder_high = custom_resnet18(num_channels=conf.n_chan_in, pretrained=conf.pretrained, flatten=True)
        
        # Aggregator for High-Res features (Can reuse self.scorer or build a new one)
        self.aggregator = Transformer(conf.n_token, conf.H, conf.D, conf.D_k, conf.D_v, conf.D_inner, conf.attn_dropout, conf.dropout)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(conf.D, conf.n_class),
            # Do not use Softmax if using CrossEntropyLoss later! 
            # (Assuming BCELoss/BCEWithLogitsLoss or CrossEntropy depending on dataset)
            # Will output logits.
        )
        
        if self.use_pos:
            self.pos_enc = pos_enc_1d(conf.D, conf.N).unsqueeze(0)
        else:
            self.pos_enc = None


    def _score_and_select_m(self, emb, M, idx):
        """ Scoring mechanism for the IPS loop """
        # Obtain scores from transformer
        attn = self.scorer.get_scores(emb) # (B, M+I)
        # Hard Top-M Selection
        top_idx = torch.topk(attn, M, dim=-1)[1] # (B, M)
        
        D_emb = emb.shape[-1]
        mem_emb = torch.gather(emb, 1, top_idx.unsqueeze(-1).expand(-1, -1, D_emb))
        mem_idx = torch.gather(idx, 1, top_idx)
        return mem_emb, mem_idx

    @torch.no_grad()
    def scouting_pass(self, patches_low):
        """
        Pass 1: Iterative Patch Selection on Downsampled Patches.
        Runs in strictly no-gradient mode to save VRAM.
        matches `ips(self, patches)` exactly.
        
        patches_low: (B, N, C, H_low, W_low)
        Returns:
            global_indices: (B, M) -> The absolute indices of the best downsampled patches.
        """
        if self.training:
            self.encoder_low.eval()
            self.scorer.eval()

        device = patches_low.device
        B, N = patches_low.shape[:2]
        M = self.M
        I = self.I

        if M >= N:
            # Trivial case
            return torch.arange(N, device=device).unsqueeze(0).expand(B, N)

        patch_shape = patches_low.shape
        D = self.D
        
        # Init buffer (first M patches)
        init_patch = patches_low[:, :M]
        mem_emb = self.encoder_low(init_patch.reshape(-1, *patch_shape[2:]))
        mem_emb = mem_emb.view(B, M, -1)
        
        global_idx = torch.arange(N, dtype=torch.int64, device=device).unsqueeze(0).expand(B, -1)
        mem_idx = global_idx[:, :M]

        n_iter = math.ceil((N - M) / I)
        for i in range(n_iter):
            start_idx = i * I + M
            end_idx = min(start_idx + I, N)

            iter_patch = patches_low[:, start_idx:end_idx]
            iter_idx = global_idx[:, start_idx:end_idx]

            iter_emb = self.encoder_low(iter_patch.reshape(-1, *patch_shape[2:]))
            iter_emb = iter_emb.view(B, -1, D)
            
            all_emb = torch.cat((mem_emb, iter_emb), dim=1)
            all_idx = torch.cat((mem_idx, iter_idx), dim=1)

            mem_emb, mem_idx = self._score_and_select_m(all_emb, M, all_idx)

        # Restore modes
        if self.training:
            self.encoder_low.train()
            self.scorer.train()

        return mem_idx

    def training_pass(self, patches_low_m, patches_high_m):
        """
        Pass 2: Differentiable Learner.
        Takes the M selected patches in BOTH downsampled and high-res forms.
        
        patches_low_m: (B, M, C, H_low, W_low) -> Same content selected in pass 1.
        patches_high_m: (B, M, C, H_high, W_high)
        """
        B, M = patches_low_m.shape[:2]
        device = patches_low_m.device
        patch_shape_high = patches_high_m.shape

        # 1. Re-Embed downsampled patches to enable gradients
        embs_low = self.encoder_low(patches_low_m.reshape(-1, *patches_low_m.shape[2:]))
        embs_low = embs_low.view(B, M, -1)

        # 2. Score with Transformer
        scores = self.scorer.get_scores(embs_low) # (B, M)
        
        # 0-1 Normalization for DPS stability
        scores_min = scores.min(dim=-1, keepdim=True)[0]
        scores_max = scores.max(dim=-1, keepdim=True)[0]
        scores = (scores - scores_min) / (scores_max - scores_min + 1e-5)

        # 3. DPS Soft Selection (Perturbed Top-K)
        # indicators: (B, K, M)
        indicators = self.dps_topk(scores)

        # 4. Memory Trick: Soft Gather High-Res Patches
        # patches_high_m: (B, M, C, H_high, W_high)
        # We want (B, K, C, H_high, W_high).
        # Einsum: bkm (indicators), bmchw (high-res patches) -> bkchw
        selected_high_res = torch.einsum('bkm,bmchw->bkchw', indicators, patches_high_m)

        # 5. Heavy Feature Extraction on K high-res patches
        K, C, H, W = selected_high_res.shape[1:]
        embs_high = self.encoder_high(selected_high_res.reshape(-1, C, H, W))
        embs_high = embs_high.view(B, K, -1)

        # 6. Aggregation & Classification
        agg_feats = self.aggregator(embs_high) # (B, D) depending on Transformer logic
        # Usually transformer returns (B, D) if taking CLS token, or (B, K, D). Assuming (B, D) from ips design task
        
        # If transformer returns sequence, pool it. Assume IDPS transformer returns (B, D) from prototype
        if agg_feats.dim() == 3:
            # if (B, K, D), mean pool
            agg_feats = agg_feats.mean(dim=1)
            
        preds = self.classifier(agg_feats)

        return preds
