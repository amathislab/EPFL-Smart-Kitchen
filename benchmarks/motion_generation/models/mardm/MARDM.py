#
# Copyright 2025-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under the MIT License.
# A copy is included in LICENSE.
#
# Incorporates code adapted from MARDM by neu-vi
# Original work Copyright (c) 2024 by neu-vi
# Source: https://github.com/neu-vi/MARDM
# Originally licensed under MIT License
#
import math
from functools import partial

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mardm.DiffMLPs import DiffMLPs_models
from models.mask_transformer.tools import *
from timm.models.vision_transformer import Mlp


#################################################################################
#                                      MARDM                                    #
#################################################################################
class MARDM(nn.Module):
    def __init__(
        self,
        ae_dim,
        cond_mode,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.2,
        clip_dim=512,
        diffmlps_batch_mul=4,
        diffmlps_model="SiT-XL",
        cond_drop_prob=0.1,
        clip_version="ViT-B/32",
        **kargs
    ):
        super(MARDM, self).__init__()

        self.ae_dim = ae_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout

        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == "action":
            assert "num_actions" in kargs
            self.num_actions = kargs.get("num_actions", 1)
            self.encode_action = partial(F.one_hot, num_classes=self.num_actions)
        # --------------------------------------------------------------------------
        # MAR Tranformer
        print("Loading MARTransformer...")
        self.input_process = InputProcess(self.ae_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        self.MARTransformer = nn.ModuleList(
            [
                MARTransBlock(
                    self.latent_dim, num_heads, mlp_size=ff_size, drop_out=self.dropout
                )
                for _ in range(num_layers)
            ]
        )

        if self.cond_mode == "text":
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == "action":
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == "uncond":
            self.cond_emb = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")

        self.mask_latent = nn.Parameter(torch.zeros(1, 1, self.ae_dim))

        self.apply(self.__init_weights)
        for block in self.MARTransformer:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        if self.cond_mode == "text":
            print("Loading CLIP...")
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

        # --------------------------------------------------------------------------
        # DiffMLPs
        print("Loading DiffMLPs...")
        self.DiffMLPs = DiffMLPs_models[diffmlps_model](
            target_channels=self.ae_dim, z_channels=self.latent_dim
        )
        self.diffmlps_batch_mul = diffmlps_batch_mul

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device="cpu", jit=False)
        assert torch.cuda.is_available()
        clip.model.convert_weights(clip_model)

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def encode_image(self, raw_image):
        image = self.clip_model.encode_image(raw_image).float()
        return image

    def mask_cond(self, cond, force_mask=False):
        bs, modal_num, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_drop_prob
            ).view(bs, 1, 1)
            return cond * (1.0 - mask)
        else:
            return cond

    def forward(self, latents, cond, padding_mask, force_mask=False, mask=None):
        cond = self.mask_cond(cond, force_mask=force_mask)
        x = self.input_process(latents)
        cond = self.cond_emb(cond).reshape(cond.shape[0], 2 * self.latent_dim)
        x = self.position_enc(x)
        x = x.permute(1, 0, 2)
        if (
            mask is not None
        ):  # hard pseudo reorder, in practice, after pe, for transformer encoder architecture should be near same without hard because of bidirectional attention.
            sort_indices = torch.argsort(mask.to(torch.float), dim=1)
            x = torch.gather(
                x, dim=1, index=sort_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
            )
            inverse_indices = torch.argsort(sort_indices, dim=1)
            padding_mask = torch.gather(padding_mask, dim=1, index=sort_indices)

        for block in self.MARTransformer:
            x = block(x, cond, padding_mask)
        if mask is not None:
            x = torch.gather(
                x, dim=1, index=inverse_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
            )
        return x

    def forward_loss(self, latents, y, m_lens, image):
        latents = latents.permute(0, 2, 1)
        b, l, d = latents.shape
        device = latents.device

        non_pad_mask = lengths_to_mask(m_lens, l)
        latents = torch.where(
            non_pad_mask.unsqueeze(-1), latents, torch.zeros_like(latents)
        )

        target = latents.clone().detach()
        input = latents.clone()

        force_mask = False
        if self.cond_mode == "text":
            with torch.no_grad():
                text_cond_vector = self.encode_text(y)
                image_cond_vector = self.encode_image(image)
                cond_vector = torch.stack([text_cond_vector, image_cond_vector], dim=1)
        elif self.cond_mode == "action":
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == "uncond":
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        rand_time = uniform((b,), device=device)
        rand_mask_probs = cosine_schedule(rand_time)
        num_masked = (l * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((b, l), device=device).argsort(dim=-1)
        mask = batch_randperm < num_masked.unsqueeze(-1)
        mask &= non_pad_mask
        mask_rlatents = get_mask_subset_prob(mask, 0.1)
        rand_latents = torch.randn_like(input)
        input = torch.where(mask_rlatents.unsqueeze(-1), rand_latents, input)
        mask_mlatents = get_mask_subset_prob(mask & ~mask_rlatents, 0.88)
        input = torch.where(
            mask_mlatents.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), input
        )

        z = self.forward(input, cond_vector, ~non_pad_mask, force_mask)
        target = target.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        z = z.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        mask = mask.reshape(b * l).repeat(self.diffmlps_batch_mul)
        target = target[mask]
        z = z[mask]
        loss = self.DiffMLPs(z=z, target=target)

        return loss

    def forward_with_CFG(
        self,
        latents,
        cond_vector,
        padding_mask,
        cfg=3,
        mask=None,
        force_mask=False,
        hard_pseudo_reorder=False,
    ):
        if hard_pseudo_reorder:
            reorder_mask = mask.clone()
        else:
            reorder_mask = None
        if force_mask:
            return self.forward(
                latents, cond_vector, padding_mask, force_mask=True, mask=reorder_mask
            )

        logits = self.forward(latents, cond_vector, padding_mask, mask=reorder_mask)
        if cfg != 1:
            aux_logits = self.forward(
                latents, cond_vector, padding_mask, force_mask=True, mask=reorder_mask
            )
            mixed_logits = torch.cat([logits, aux_logits], dim=0)
        else:
            mixed_logits = logits
        b, l, d = mixed_logits.size()
        if mask is not None:
            mask2 = torch.cat([mask, mask], dim=0).reshape(b * l)
            mixed_logits = (mixed_logits.reshape(b * l, d))[mask2]
        else:
            mixed_logits = mixed_logits.reshape(b * l, d)
        output = self.DiffMLPs.sample(mixed_logits, 1, cfg)
        if cfg != 1:
            scaled_logits, _ = output.chunk(2, dim=0)
        else:
            scaled_logits = output
        if mask is not None:
            latents = latents.reshape(b // 2 * l, self.ae_dim)
            latents[mask.reshape(b // 2 * l)] = scaled_logits
            scaled_logits = latents.reshape(b // 2, l, self.ae_dim)

        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        conds,
        image,
        m_lens,
        timesteps: int,
        cond_scale: int,
        temperature=1,
        force_mask=False,
        hard_pseudo_reorder=False,
    ):
        device = next(self.parameters()).device
        l = max(m_lens)
        b = len(m_lens)

        if self.cond_mode == "text":
            with torch.no_grad():
                # cond_vector = self.encode_text(conds)
                text_cond_vector = self.encode_text(conds)
                image_cond_vector = self.encode_image(image)
                cond_vector = torch.stack([text_cond_vector, image_cond_vector], dim=1)
        elif self.cond_mode == "action":
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == "uncond":
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, l)

        latents = torch.where(
            padding_mask.unsqueeze(-1),
            torch.zeros(b, l, self.ae_dim).to(device),
            self.mask_latent.repeat(b, l, 1),
        )
        masked_rand_schedule = torch.where(
            padding_mask, 1e5, torch.rand_like(padding_mask, dtype=torch.float)
        )

        for timestep, steps_until_x0 in zip(
            torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))
        ):
            rand_mask_prob = cosine_schedule(timestep)
            num_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)
            sorted_indices = masked_rand_schedule.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = ranks < num_masked.unsqueeze(-1)

            latents = torch.where(
                is_mask.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), latents
            )
            logits = self.forward_with_CFG(
                latents,
                cond_vector=cond_vector,
                padding_mask=padding_mask,
                cfg=cond_scale,
                mask=is_mask,
                force_mask=force_mask,
                hard_pseudo_reorder=hard_pseudo_reorder,
            )
            latents = torch.where(is_mask.unsqueeze(-1), logits, latents)

            masked_rand_schedule = masked_rand_schedule.masked_fill(~is_mask, 1e5)

        latents = torch.where(
            padding_mask.unsqueeze(-1), torch.zeros_like(latents), latents
        )
        return latents.permute(0, 2, 1)

    @torch.no_grad()
    @eval_decorator
    def edit(
        self,
        conds,
        latents,
        m_lens,
        timesteps: int,
        cond_scale: int,
        temperature=1,
        force_mask=False,
        edit_mask=None,
        padding_mask=None,
        hard_pseudo_reorder=False,
    ):

        device = next(self.parameters()).device
        l = latents.shape[-1]

        if self.cond_mode == "text":
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == "action":
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == "uncond":
            cond_vector = torch.zeros(1, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        if padding_mask == None:
            padding_mask = ~lengths_to_mask(m_lens, l)

        if edit_mask == None:
            mask_free = True
            latents = torch.where(
                padding_mask.unsqueeze(-1),
                torch.zeros(latents.shape[0], l, self.ae_dim).to(device),
                latents.permute(0, 2, 1),
            )
            edit_mask = torch.ones_like(padding_mask)
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            masked_rand_schedule = torch.where(
                edit_mask, torch.rand_like(edit_mask, dtype=torch.float), 1e5
            )
        else:
            mask_free = False
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            latents = torch.where(
                padding_mask.unsqueeze(-1),
                torch.zeros(latents.shape[0], l, self.ae_dim).to(device),
                latents.permute(0, 2, 1),
            )
            latents = torch.where(
                edit_mask.unsqueeze(-1),
                self.mask_latent.repeat(latents.shape[0], l, 1),
                latents,
            )
            masked_rand_schedule = torch.where(
                edit_mask, torch.rand_like(edit_mask, dtype=torch.float), 1e5
            )

        for timestep, steps_until_x0 in zip(
            torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))
        ):
            rand_mask_prob = 0.16 if mask_free else cosine_schedule(timestep)
            num_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)
            sorted_indices = masked_rand_schedule.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = ranks < num_masked.unsqueeze(-1)

            latents = torch.where(
                is_mask.unsqueeze(-1),
                self.mask_latent.repeat(latents.shape[0], latents.shape[1], 1),
                latents,
            )
            logits = self.forward_with_CFG(
                latents,
                cond_vector=cond_vector,
                padding_mask=padding_mask,
                cfg=cond_scale,
                mask=is_mask,
                force_mask=force_mask,
                hard_pseudo_reorder=hard_pseudo_reorder,
            )
            latents = torch.where(is_mask.unsqueeze(-1), logits, latents)

            masked_rand_schedule = masked_rand_schedule.masked_fill(~is_mask, 1e5)

        latents = torch.where(edit_mask.unsqueeze(-1), latents, latents)
        latents = torch.where(
            padding_mask.unsqueeze(-1), torch.zeros_like(latents), latents
        )
        return latents.permute(0, 2, 1)


#################################################################################
#                                     MARDM Zoos                                #
#################################################################################
def mardm_ddpm_xl(**kwargs):
    return MARDM(
        latent_dim=1024,
        ff_size=4096,
        num_layers=1,
        num_heads=16,
        dropout=0.2,
        clip_dim=512,
        diffmlps_model="DDPM-XL",
        diffmlps_batch_mul=4,
        cond_drop_prob=0.1,
        **kwargs
    )


def mardm_sit_xl(**kwargs):
    return MARDM(
        latent_dim=1024,
        ff_size=4096,
        num_layers=1,
        num_heads=16,
        dropout=0.2,
        clip_dim=512,
        diffmlps_model="SiT-XL",
        diffmlps_batch_mul=4,
        cond_drop_prob=0.1,
        **kwargs
    )


MARDM_models = {
    "MARDM-DDPM-XL": mardm_ddpm_xl,
    "MARDM-SiT-XL": mardm_sit_xl,
}


#################################################################################
#                                 Inner Architectures                           #
#################################################################################
def modulate_here(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = x.permute((1, 0, 2))
        x = self.poseEmbedding(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.2):
        super().__init__()
        assert embed_dim % 8 == 0
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x, mask):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            mask = mask[:, None, None, :]
            att = att.masked_fill(mask != 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y


class MARTransBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_size=1024, drop_out=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, drop_out_rate=drop_out)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = mlp_size
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(2 * hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, padding_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate_here(self.norm1(x), shift_msa, scale_msa), mask=padding_mask
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate_here(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x
