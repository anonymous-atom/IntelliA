import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config


base_model_name = "openai/gpt-oss-20b"


quantization_config = Mxfp4Config(dequantize=True)


class TransformerResidualHead(nn.Module):
    """
    Transformer-based residual head for flat feature vectors.
    """
    def __init__(
        self,
        in_dim=2880,
        num_chunks=8,
        d_model=512,
        num_layers=2,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.2,
        num_cls=2,
        pool_dropout=0.15,
        eps=1e-4
    ):
        super().__init__()
        assert in_dim % num_chunks == 0
        self.in_dim = in_dim
        self.num_chunks = num_chunks
        self.chunk_dim = in_dim // num_chunks
        self.d_model = d_model

        # Per-chunk projection into d_model
        self.chunk_proj = nn.Linear(self.chunk_dim, d_model)

        # Transformer encoder (small)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False  # encoder expects (S, B, E)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling: learnable query
        self.pool_q = nn.Parameter(torch.randn(1, 1, d_model))
        self.pool_dropout = nn.Dropout(pool_dropout)
        self.pool_norm = nn.LayerNorm(d_model, eps=eps)

        # Residual MLP block applied after pooling
        self.residual_fc1 = nn.Linear(d_model, dim_feedforward)
        self.residual_act = nn.GELU()
        self.residual_fc2 = nn.Linear(dim_feedforward, d_model)
        self.residual_norm = nn.LayerNorm(d_model, eps=eps)
        self.residual_dropout = nn.Dropout(dropout)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_cls)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, in_dim) float / bfloat16 tensor
        returns logits: (B, num_cls)
        """
        B = x.shape[0]
        # break into chunks
        x = x.view(B, self.num_chunks, self.chunk_dim)  # (B, S, chunk_dim)
        # project each chunk
        x = self.chunk_proj(x)                           # (B, S, d_model)
        # prepare for transformer: (S, B, d_model)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)                              # (S, B, d_model)
        # back to (B, S, d_model)
        x = x.permute(1, 0, 2)

        # Attention pooling
        q = self.pool_q.expand(B, -1, -1)                # (B, 1, d_model)
        att_scores = torch.matmul(q, x.transpose(1, 2))  # (B,1,S)
        att_weights = torch.softmax(att_scores, dim=-1)  # (B,1,S)
        pooled = torch.matmul(att_weights, x)            # (B,1,d_model)
        pooled = pooled.squeeze(1)                       # (B, d_model)
        pooled = self.pool_norm(pooled)
        pooled = self.pool_dropout(pooled)

        # Residual MLP block
        res = pooled
        x2 = self.residual_fc1(pooled)
        x2 = self.residual_act(x2)
        x2 = self.residual_dropout(x2)
        x2 = self.residual_fc2(x2)
        x2 = self.residual_dropout(x2)
        x2 = self.residual_norm(x2 + res)

        logits = self.classifier(x2)
        return logits

class ChatMultiObjClassifier(nn.Module):
    def __init__(self, num_classes_per_head=[2, 2, 2]):
        super().__init__()
        
        self.heads = nn.ModuleList([
            TransformerResidualHead(
                in_dim=2880,
                num_chunks=16,     # tune 8/16/32
                d_model=512,       # tune 128/256/384
                num_layers=4,      # 1-4
                nhead=8,
                dim_feedforward=1024,
                dropout=0.15,
                num_cls=num_cls
            )
            for num_cls in num_classes_per_head
        ])




    def get_representation(self, input_ids, attention_mask):
        """
        Extract the final hidden representation (last token) from the base model.
        This method is used for precomputing representations.
        
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            
        Returns:
            rep: [B, H] - Final hidden representation
        """
        base_model_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        last_hidden = base_model_output.hidden_states[-1]  # [B, L, H]
        rep = last_hidden[:, -1, :]  # [B, H] - take last token representation
        return rep

    def forward(self, input_ids=None, attention_mask=None, rep=None, labels=None):
        """
        input_ids: [B, L]
        attention_mask: [B, L]
        labels: [B, k] (optional, k = number of objectives)
        """
        # Get representation using the helper method
        if rep is None:
            rep = self.get_representation(input_ids, attention_mask)

        # Compute logits from each head: List of [B, C_j]
        logits_list = [head(rep.float()) for head in self.heads]
        logits = torch.stack(logits_list, dim=1)  # [B, k, 2]

        output = {"logits": logits}

        if labels is not None:
            loss = 0
            for j, head_logits in enumerate(logits_list):  # head_logits: [B, 2]
                target = labels[:, j]                      # [B]
                loss += nn.CrossEntropyLoss()(head_logits, target)
            output["loss"] = loss

        return output
    
    def compute_scalar_reward(self, logits, weights=None):
        # Compute positive-class probabilities for each head
        probs = torch.softmax(logits, dim=-1)[:, :, 1]  # shape: [B, k]
        if weights is None:
            weights = torch.full((logits.size(1),), 1/logits.size(1), device=probs.device)
        return (probs * weights).sum(dim=1)  # scalar per example
