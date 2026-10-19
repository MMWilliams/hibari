"""TinyKokoro: a single-voice student built for very low CPU.

Kokoro's cost is its vocoder: transposed-conv upsampling plus residual stacks at 800 and 4800 Hz
and a sine source at 24 kHz, 70-90% of CPU time. Here nothing runs faster than 80 frames/s:
ConvNeXt blocks at frame rate predict STFT magnitude and phase (Vocos-style) and one inverse FFT
makes the waveform. The fixed voice removes style conditioning entirely.

    tokens -> TextEncoder -> durations (log-domain regression)
           -> repeat to 40 Hz -> prosody convs -> F0, energy at 80 Hz (teacher-supervised)
           -> 80 Hz decoder (features + F0 + energy) -> |S|, angle(S) -> iSTFT (n_fft 1200, hop 300)

The network stops at the spectrum; `spec_to_wav` is outside it so an on-device runtime can use
a real FFT instead of an exported DFT matrix, which would cost more than the whole network.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

N_FFT, HOP, SR = 1200, 300, 24000


class ConvNeXt(nn.Module):
    def __init__(self, c, ff, k=7):
        super().__init__()
        self.dw = nn.Conv1d(c, c, k, padding=k // 2, groups=c)
        self.norm = nn.LayerNorm(c)
        self.pw1, self.pw2 = nn.Linear(c, ff), nn.Linear(ff, c)
        self.gamma = nn.Parameter(torch.full((c,), 1e-2))

    def forward(self, x, mask=None):                       # x [B,C,T], mask [B,1,T]
        y = self.dw(x if mask is None else x * mask).transpose(1, 2)
        y = self.pw2(F.gelu(self.pw1(self.norm(y))))
        x = x + (self.gamma * y).transpose(1, 2)
        return x if mask is None else x * mask


class TextEncoder(nn.Module):
    """Conv front for local context, then ALBERT-style shared attention for sentence context:
    depth costs compute but not parameters."""

    def __init__(self, n_tok, c, layers, heads, ff):
        super().__init__()
        self.emb = nn.Embedding(n_tok, c)
        self.convs = nn.ModuleList([ConvNeXt(c, ff, 5) for _ in range(2)])
        self.attn = nn.TransformerEncoderLayer(c, heads, ff, dropout=0.1, activation="gelu",
                                               batch_first=True, norm_first=True)
        self.layers = layers
        self.norm = nn.LayerNorm(c)

    def forward(self, ids, pad):                           # pad [B,L] True where padding
        m = (~pad)[:, None].float()
        x = self.emb(ids).transpose(1, 2) * m
        for c in self.convs:
            x = c(x, m)
        x = x.transpose(1, 2)
        for _ in range(self.layers):
            x = self.attn(x, src_key_padding_mask=pad)
        return self.norm(x).transpose(1, 2) * m           # [B,C,L]


class TinyKokoro(nn.Module):
    def __init__(self, n_tok=178, c_text=128, text_layers=4, heads=2, c_pro=96, pro_blocks=3,
                 c_dec=128, dec_blocks=6, dec_ff=384, n_fft=N_FFT):
        super().__init__()
        self.text = TextEncoder(n_tok, c_text, text_layers, heads, 2 * c_text)
        self.dur = nn.Sequential(nn.Conv1d(c_text, c_text, 3, padding=1), nn.GELU(),
                                 nn.Conv1d(c_text, 1, 1))
        self.pro_in = nn.Conv1d(c_text, c_pro, 1)
        self.pro = nn.ModuleList([ConvNeXt(c_pro, 2 * c_pro) for _ in range(pro_blocks)])
        self.pro_out = nn.Conv1d(c_pro, 4, 1)              # (F0, N) x 2 sub-frames -> 80 Hz
        self.dec_in = nn.Conv1d(c_text + 2, c_dec, 7, padding=3)
        self.dec = nn.ModuleList([ConvNeXt(c_dec, dec_ff) for _ in range(dec_blocks)])
        self.dec_norm = nn.LayerNorm(c_dec)
        self.bins = n_fft // 2 + 1
        self.head = nn.Linear(c_dec, 2 * self.bins)

    def encode(self, ids, pad):
        h = self.text(ids, pad)
        logd = self.dur(h).squeeze(1)                      # log(1 + frames)
        return h, logd

    def prosody(self, hf):                                 # hf [B,C,F40] -> [B,F80] x 2
        x = self.pro_in(hf)
        for b in self.pro:
            x = b(x)
        y = self.pro_out(x)                                # [B,4,F]
        B, _, Fr = y.shape
        y = y.view(B, 2, 2, Fr).permute(0, 1, 3, 2).reshape(B, 2, 2 * Fr)
        return y[:, 0] * 100.0, y[:, 1] * 5.0             # teacher scale: F0 Hz, N ~[-10, 8]

    def decode(self, hf, f0, n):                           # hf [B,C,F40], f0/n [B,F80]
        x = hf.repeat_interleave(2, dim=-1)
        f0f = torch.log1p(F.relu(f0))[:, None] / 5.0
        x = self.dec_in(torch.cat([x, f0f, n[:, None] / 5.0], 1))
        for b in self.dec:
            x = b(x)
        y = self.head(self.dec_norm(x.transpose(1, 2))).transpose(1, 2)
        logmag, phase = y[:, : self.bins], y[:, self.bins:]
        return logmag, phase                               # [B,bins,F80]

    @staticmethod
    def expand(h, dur):                                    # h [B,C,L], dur [B,L] ints -> [B,C,F]
        return torch.stack([torch.repeat_interleave(h[b], dur[b], dim=-1) for b in range(h.shape[0])])

    @torch.no_grad()
    def infer(self, ids, speed=1.0):                       # ids [1,L] -> logmag, phase
        pad = torch.zeros_like(ids, dtype=torch.bool)
        h, logd = self.encode(ids, pad)
        dur = torch.round(torch.expm1(logd) / speed).clamp(min=1).long()
        hf = self.expand(h, dur)
        f0, n = self.prosody(hf)
        return self.decode(hf, f0, n)


def spec_to_wav(logmag, phase, n_fft=N_FFT, hop=HOP):
    mag = torch.exp(logmag.clamp(max=7.0))
    spec = torch.polar(mag, phase)
    win = torch.hann_window(n_fft, device=logmag.device)
    return torch.istft(spec, n_fft, hop, n_fft, win, center=True, length=logmag.shape[-1] * hop)


if __name__ == "__main__":
    import time
    torch.set_num_threads(1)
    m = TinyKokoro().eval()
    n = sum(p.numel() for p in m.parameters())
    print(f"params {n/1e6:.3f}M")
    for k, mod in (("text", m.text), ("prosody", m.pro), ("decoder", m.dec), ("head", m.head)):
        print(f"  {k:8s} {sum(p.numel() for p in mod.parameters())/1e3:7.1f}K")
    ids = torch.randint(1, 178, (1, 100))
    with torch.no_grad():
        h, logd = m.encode(ids, torch.zeros_like(ids, dtype=torch.bool))
        hf = m.expand(h, torch.full((100,), 4)[None])      # 400 frames = 10 s
        for _ in range(3):
            spec_to_wav(*m.decode(hf, *m.prosody(hf)))
        t = time.perf_counter()
        for _ in range(10):
            h, _ = m.encode(ids, torch.zeros_like(ids, dtype=torch.bool))
            hf = m.expand(h, torch.full((100,), 4)[None])
            w = spec_to_wav(*m.decode(hf, *m.prosody(hf)))
        dt = (time.perf_counter() - t) / 10
    print(f"1 thread: {dt*1000:.1f} ms for {w.shape[-1]/SR:.1f} s audio, RTF {dt/(w.shape[-1]/SR):.4f}")
