from __future__ import annotations

import torch
from torch import nn


class TimePrior(nn.Module):
    def __init__(self, num_stages: int = 5, num_tasks: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.stage_emb = nn.Embedding(num_stages, hidden_dim)
        self.task_emb = nn.Embedding(num_tasks, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(1 + hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        primitive_time: torch.Tensor,
        stage_id: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        primitive_time = primitive_time.reshape(-1, 1).float()
        x = torch.cat([primitive_time, self.stage_emb(stage_id), self.task_emb(task_id)], dim=-1)
        return self.net(x)


class MLPCritic(nn.Module):
    def __init__(
        self,
        feature_dim: int = 768,
        proprio_dim: int = 14,
        action_dim: int = 14,
        horizon: int = 8,
        num_stages: int = 5,
        num_tasks: int = 16,
    ) -> None:
        super().__init__()
        self.obs_proj = nn.Sequential(nn.Linear(feature_dim, 256), nn.GELU())
        self.proprio_proj = nn.Sequential(nn.Linear(proprio_dim, 128), nn.GELU())
        self.action_proj = nn.Sequential(nn.Linear(horizon * action_dim, 256), nn.GELU())
        self.stage_emb = nn.Embedding(num_stages, 64)
        self.task_emb = nn.Embedding(num_tasks, 64)
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128 + 256 + 64 + 64, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        obs_features: torch.Tensor,
        proprio: torch.Tensor,
        action_chunk: torch.Tensor,
        stage_id: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        obs = obs_features.float().mean(dim=(1, 2))
        action = action_chunk.float().flatten(start_dim=1)
        x = torch.cat(
            [
                self.obs_proj(obs),
                self.proprio_proj(proprio.float()),
                self.action_proj(action),
                self.stage_emb(stage_id),
                self.task_emb(task_id),
            ],
            dim=-1,
        )
        return self.fusion(x)


class StageFiLMTransformerCritic(nn.Module):
    def __init__(
        self,
        feature_dim: int = 768,
        proprio_dim: int = 14,
        action_dim: int = 14,
        num_stages: int = 5,
        num_tasks: int = 16,
        hidden_dim: int = 256,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_proj = nn.Linear(feature_dim, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.stage_emb = nn.Embedding(num_stages, hidden_dim)
        self.task_emb = nn.Embedding(num_tasks, hidden_dim)
        self.stage_film = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        obs_features: torch.Tensor,
        proprio: torch.Tensor,
        action_chunk: torch.Tensor,
        stage_id: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = obs_features.shape[0]

        obs_tokens = self.obs_proj(obs_features.float().mean(dim=2))
        proprio_token = self.proprio_proj(proprio.float()).unsqueeze(1)
        stage_token = self.stage_emb(stage_id).unsqueeze(1)
        task_token = self.task_emb(task_id).unsqueeze(1)

        action_tokens = self.action_proj(action_chunk.float())
        film = self.stage_film(stage_token.squeeze(1))
        gamma, beta = film.chunk(2, dim=-1)
        action_tokens = action_tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        cls = self.cls.expand(batch_size, -1, -1)
        tokens = torch.cat(
            [cls, obs_tokens, proprio_token, stage_token, task_token, action_tokens],
            dim=1,
        )
        fused = self.fusion(tokens)
        return self.head(fused[:, 0])


class PromptFiLMTransformerCritic(nn.Module):
    def __init__(
        self,
        feature_dim: int = 768,
        proprio_dim: int = 14,
        action_dim: int = 14,
        prompt_dim: int = 512,
        hidden_dim: int = 256,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
        use_action: bool = True,
    ) -> None:
        super().__init__()
        self.use_action = use_action
        self.obs_proj = nn.Linear(feature_dim, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(prompt_dim),
            nn.Linear(prompt_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.prompt_film = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        obs_features: torch.Tensor,
        proprio: torch.Tensor,
        action_chunk: torch.Tensor,
        prompt_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = obs_features.shape[0]

        obs_tokens = self.obs_proj(obs_features.float().mean(dim=2))
        proprio_token = self.proprio_proj(proprio.float()).unsqueeze(1)
        prompt_token = self.prompt_proj(prompt_features.float()).unsqueeze(1)

        tokens = [self.cls.expand(batch_size, -1, -1), obs_tokens, proprio_token, prompt_token]
        if self.use_action:
            action_tokens = self.action_proj(action_chunk.float())
            film = self.prompt_film(prompt_token.squeeze(1))
            gamma, beta = film.chunk(2, dim=-1)
            action_tokens = action_tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            tokens.append(action_tokens)

        fused = self.fusion(torch.cat(tokens, dim=1))
        return self.head(fused[:, 0])
