import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()

        assert d_model % 2 == 0

        pe = torch.zeros(max_seq_len, d_model)
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq, indexing='ij')
        pe_2i = torch.sin(pos / 10000 ** (two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000 ** (two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)

        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)

    def forward(self, t):
        return self.embedding(t)


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, d, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, L = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, L)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(feats_sum)
        attn = self.softmax(attn.view(B, self.height, C, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class UnetBlock(nn.Module):

    def __init__(self, shape, in_c, out_c, residual=False):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv1d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, 1, 1)
        self.activation = nn.GELU()
        self.residual = residual
        if residual:
            if in_c == out_c:
                self.residual_conv = SKFusion(dim=in_c)
            else:
                self.residual_conv = nn.Conv1d(in_c, out_c, 1)

    def forward(self, x):
        out = self.ln(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.residual:
            if self.in_c == self.out_c:
                out = self.residual_conv([out, x])
            else:
                x = self.residual_conv(x)
                out += x

        out = self.activation(out)
        return out


class qkvEmbedd(nn.Module):
    def __init__(self, d_model):
        super(qkvEmbedd, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv1_q = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                      kernel_size=1, bias=False)
        self.tokenConv2_q = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                      kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.tokenConv1_k = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                      kernel_size=1, bias=False)
        self.tokenConv2_k = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                      kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.tokenConv1_v = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                      kernel_size=1, bias=False)
        self.tokenConv2_v = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                      kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.type(torch.FloatTensor).to(device=x.device)

        q = self.tokenConv1_q(x.permute(0, 2, 1))
        q = self.tokenConv2_q(q).transpose(1, 2)
        k = self.tokenConv1_k(x.permute(0, 2, 1))
        k = self.tokenConv2_k(k).transpose(1, 2)
        v = self.tokenConv1_v(x.permute(0, 2, 1))
        v = self.tokenConv2_v(v).transpose(1, 2)
        return q, k, v


class TAttention(nn.Module):
    def __init__(self, d_model, attn_drop=0., proj_drop=0.):
        super(TAttention, self).__init__()

        self.scale = d_model ** -0.5
        self.qkvgen = qkvEmbedd(d_model)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                              kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkvgen(x)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x.permute(0, 2, 1))
        x = x.transpose(2, 1)
        x = self.proj_drop(x)
        return x


class Attention1d(nn.Module):

    def __init__(self, d_model, in_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, num_channels=in_channels)
        self.attnSSA = TAttention(d_model)

    def forward(self, x):
        x = self.norm1(x)
        attTSA = self.attnTSA(x)
        x = x + attTSA
        return x


class UNet(nn.Module):
    def __init__(self,
                 n_steps,
                 channels=[16, 32, 64, 128],
                 pe_dim=128,
                 residual=True) -> None:
        super().__init__()
        C, L = 3, 1500

        layers = len(channels)
        Ls = [L]
        cL = L
        for _ in range(layers - 1):
            cL //= 2
            Ls.append(cL)

        self.pe = PositionalEncoding(n_steps, pe_dim)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pe_linears_en = nn.ModuleList()
        self.pe_linears_de = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        prev_channel = C
        for channel, cL in zip(channels[0:-1], Ls[0:-1]):
            self.pe_linears_en.append(
                nn.Sequential(nn.Linear(pe_dim, prev_channel), nn.ReLU(),
                              nn.Linear(prev_channel, prev_channel)))
            self.encoders.append(
                nn.Sequential(
                    UnetBlock(cL,
                              prev_channel,
                              channel,
                              residual=residual),
                    UnetBlock(cL,
                              channel,
                              channel,
                              residual=residual),

                ))
            self.downs.append(nn.Conv1d(channel, channel, 2, 2))
            prev_channel = channel
        self.en_attn = Attention1d(d_model=1024, in_channels=prev_channel)
        self.pe_mid = nn.Linear(pe_dim, prev_channel)
        channel = channels[-1]
        self.mid = nn.Sequential(
            UnetBlock(Ls[-1],
                      prev_channel,
                      channel,
                      residual=residual),

            UnetBlock(Ls[-1],
                      channel,
                      channel,
                      residual=residual)
        )
        prev_channel = channel
        self.de_attn = Attention1d(d_model=1024, in_channels=prev_channel)
        for channel, cL in zip(channels[-2::-1], Ls[-2::-1]):
            self.pe_linears_de.append(nn.Linear(pe_dim, prev_channel))
            self.ups.append(nn.ConvTranspose1d(prev_channel, channel, 2, 2))
            self.decoders.append(
                nn.Sequential(
                    UnetBlock((channel * 2, cL),
                              channel * 2,
                              channel,
                              residual=residual),
                    UnetBlock((channel, cL),
                              channel,
                              channel,
                              residual=residual)
                ))
            prev_channel = channel
        self.conv_out = nn.Conv1d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t):
        n = t.shape[0]
        t = self.pe(t)
        encoder_outs = []
        for pe_linear, encoder, down in zip(self.pe_linears_en, self.encoders,
                                            self.downs):
            pe = pe_linear(t).reshape(n, -1, 1)
            x = encoder(x + pe)
            encoder_outs.append(x)
            x = down(x)
        x = self.en_attn(x)
        pe = self.pe_mid(t).reshape(n, -1, 1)
        x = self.mid(x + pe)
        x = self.de_attn(x)
        for pe_linear, decoder, up, encoder_out in zip(self.pe_linears_de,
                                                       self.decoders, self.ups,
                                                       encoder_outs[::-1]):
            pe = pe_linear(t).reshape(n, -1, 1)
            x = up(x)

            if x.shape[-1] != encoder_out.shape[-1]:
                x = F.pad(x, (0, 1))
            x = torch.cat((encoder_out, x), dim=1)
            x = decoder(x + pe)
        x = self.conv_out(x)
        return x


def build_network(n_steps):
    network_cls = UNet
    network = network_cls(n_steps)
    return network
