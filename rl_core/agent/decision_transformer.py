from typing import io

import numpy as np
import torch
from torch import nn
from torchinfo import summary

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)

    def forward(self, x, pos_embedding):
        output = self.embedding(x)
        return self.embedding(x) + pos_embedding


class DecisionTransformer(nn.Module):
    metadata = {'name': 'DecisionTransformer'}

    def __init__(self, state_dim, action_dim, max_length, embed_dim, num_heads, num_layers, dim_feedforward=2048, device=DEVICE):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.device = device

        self.embed_s = EmbeddingLayer(state_dim, embed_dim)
        self.embed_a = EmbeddingLayer(action_dim, embed_dim)
        self.embed_R = EmbeddingLayer(1, embed_dim)
        self.embed_t = nn.Embedding(max_length, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)

        self.transformer = nn.Transformer(
            d_model=embed_dim, nhead=num_heads,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward, batch_first=True
        )

        self.pred_a = nn.Linear(embed_dim, action_dim)
        self.max_length = max_length

    def forward(self, R, s, a, t, mask=None, a_mask=None):
        pos_embedding = self.embed_t(t)
        s_embedding = self.embed_s(s, pos_embedding)
        a_embedding = self.embed_a(a, pos_embedding)
        R_embedding = self.embed_R(R, pos_embedding)

        input_embeds = torch.stack((R_embedding, s_embedding, a_embedding), dim=1).permute(0, 2, 1, 3)
        input_embeds = input_embeds.reshape(s.size(0), 3*s.size(1), self.embed_dim)
        input_embeds = self.embed_ln(input_embeds)

        mask_size = s.size(1) * 3

        if mask is not None:
            mask = torch.stack((mask, mask, mask), dim=1).permute(0, 2, 1).reshape(s.size(0), mask_size)
            mask = mask.bool()
        else:
            mask = torch.zeros((s.size(0), mask_size)).bool().to(DEVICE)

        attn_mask = self.transformer.generate_square_subsequent_mask(sz=mask_size).to(DEVICE)
        attn_mask = torch.isfinite(attn_mask)
        attn_mask = ~attn_mask

        hidden_states = self.transformer(
            input_embeds,
            input_embeds,
            src_key_padding_mask=mask,
            tgt_key_padding_mask=mask,
            memory_key_padding_mask=mask,
            memory_is_causal=True,
            src_is_causal=True,
            tgt_is_causal=True,
            src_mask=attn_mask,
            tgt_mask=attn_mask,
            memory_mask=attn_mask
        )

        hidden_states = hidden_states.reshape(s.size(0), s.size(1), 3, self.embed_dim).permute(0, 2, 1, 3)

        a_hidden = hidden_states[:, 1, :]

        # q_values = self.pred_a(a_hidden).squeeze(0)[-1]
        # mask_tensor = torch.tensor(a_mask, dtype=torch.bool).to(self.device)
        # m_q_values = torch.where(mask_tensor, q_values, -torch.inf)
        # argmax_action = torch.argmax(m_q_values)
        #
        # # 0.1 - eps regularization
        # probs = 0.1 * np.ones(self.action_dim) / sum(a_mask)
        # m_probs = np.where(a_mask, probs, 0)
        # m_probs[argmax_action] += 1 - 0.1
        # action = np.random.choice(np.arange(self.action_dim), p=m_probs)

        return self.pred_a(a_hidden)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    def create_log_report(self, log_dir):
        with io.open(f'{log_dir}/params.log', 'w', encoding='utf-8') as file:
            file.write('-- PARAMS --\n')
            file.write(f'state_dim {self.state_dim}\n')
            file.write(f'action_dim {self.action_dim}\n')
            file.write(f'max_length {self.max_length}')
            file.write(f'embed_dim {self.embed_dim}')
            file.write(f'num_heads {self.num_heads}')
            file.write(f'num_layers {self.num_layers}')
            file.write(f'dim_feedforward {self.dim_feedforward}')
            file.write('--\n')
            file.write(f'device {self.device}\n')
            file.write('--\n')
            file.write('\n-- ARCHITECTURE --\n')
            file.write('- PI MODEL -\n')
            embed_s = str(summary(self.embed_s, (1, self.embed_dim), verbose=0))
            embed_a = str(summary(self.embed_a, (1, self.embed_dim), verbose=0))
            embed_R = str(summary(self.embed_R, (1, self.embed_dim), verbose=0))
            embed_t = str(summary(self.embed_t, (1, self.embed_dim), verbose=0))
            embed_ln = str(summary(self.embed_ln, (1, self.embed_dim), verbose=0))
            transformer = str(summary(self.transformer, (1, self.embed_dim), verbose=0))
            file.write(f'{embed_s}')
            file.write(f'{embed_a}')
            file.write(f'{embed_R}')
            file.write(f'{embed_t}')
            file.write(f'{embed_ln}')
            file.write(f'{transformer}')
