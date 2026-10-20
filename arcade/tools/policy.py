"""Small vision behaviour-cloning policy: two cameras + robot state + skill id -> next K commands."""
import numpy as np
import torch
from torch import nn


class CameraEncoder(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 5, 2, 2), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, 2, 1), nn.GroupNorm(8, width * 2), nn.SiLU(),
            nn.Conv2d(width * 2, width * 4, 3, 2, 1), nn.GroupNorm(8, width * 4), nn.SiLU(),
            nn.Conv2d(width * 4, width * 4, 3, 2, 1), nn.GroupNorm(8, width * 4), nn.SiLU(),
        )
        # Spatial softmax keypoints keep "where" information that global pooling throws away.
        self.keypoints = nn.Conv2d(width * 4, 32, 1)
        self.out_dim = 64

    def forward(self, x):
        features = self.keypoints(self.net(x))
        b, k, h, w = features.shape
        attention = torch.softmax(features.flatten(2), -1).view(b, k, h, w)
        ys = torch.linspace(-1, 1, h, device=x.device)
        xs = torch.linspace(-1, 1, w, device=x.device)
        ky = (attention.sum(3) * ys).sum(2)
        kx = (attention.sum(2) * xs).sum(2)
        return torch.cat([kx, ky], 1)


class BCPolicy(nn.Module):
    def __init__(self, skills, horizon=8, cameras=2, state_dim=22):
        super().__init__()
        self.skills = list(skills)
        self.horizon = horizon
        self.encoders = nn.ModuleList([CameraEncoder() for _ in range(cameras)])
        self.skill_embedding = nn.Embedding(max(1, len(self.skills)), 16)
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, 64), nn.SiLU())
        width = 64 * cameras + 64 + 16
        self.head = nn.Sequential(nn.Linear(width, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(),
                                  nn.Linear(512, horizon * 7))
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))

    def forward(self, images, state, skill_index):
        # images: (B, cams, 3, H, W) float in [0, 1]
        features = [encoder(images[:, i]) for i, encoder in enumerate(self.encoders)]
        state = (state - self.state_mean) / self.state_std
        features += [self.state_mlp(state), self.skill_embedding(skill_index)]
        out = self.head(torch.cat(features, 1)).view(-1, self.horizon, 7)
        return out  # dims 0-5: pre-tanh motion, dim 6: grip logit

    @staticmethod
    def to_command(out):
        motion = torch.tanh(out[..., :6])
        grip = torch.where(out[..., 6:] > 0, 1.0, -1.0)
        return torch.cat([motion, grip], -1)


def images_to_tensor(images, device):
    """(B, cams, H, W, 3) uint8 numpy -> (B, cams, 3, H, W) float tensor."""
    tensor = torch.as_tensor(images, device=device).permute(0, 1, 4, 2, 3).float() / 255.0
    return tensor


def save_policy(path, policy, extra):
    torch.save({"state_dict": policy.state_dict(), "skills": policy.skills, "horizon": policy.horizon,
                **extra}, path)


def load_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    policy = BCPolicy(checkpoint["skills"], checkpoint["horizon"])
    policy.load_state_dict(checkpoint["state_dict"])
    policy.to(device).eval()
    return policy, checkpoint


def default_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def numpy_state(robot_state):
    from arcade.core.dataset import STATE_KEYS
    return np.concatenate([np.asarray(robot_state[k], np.float32).ravel() for k in STATE_KEYS])
