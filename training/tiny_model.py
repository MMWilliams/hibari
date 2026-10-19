"""TinyKokoro: a single-voice student built for very low CPU.

Kokoro's cost is its vocoder: transposed-conv upsampling plus residual stacks at 800 and 4800 Hz
and a sine source at 24 kHz, 70-90% of CPU time. Here nothing runs faster than 80 frames/s:
ConvNeXt blocks at frame rate predict STFT magnitude and phase (Vocos-style) and one inverse FFT
makes the waveform. The fixed voice removes style conditioning entirely.

    tokens -> TextEncoder -> durations (log-domain regression)
           -> repeat to 40 Hz -> prosody convs -> F0, energy at 80 Hz (teacher-supervised)
           -> 80 Hz decoder (features + F0 + energy) -> |S|, angle(S) -> iSTFT (n_fft 1200, hop 300)

With src=True the decoder no longer guesses absolute phase, which must advance by 2*pi*f*hop/sr
every frame and came out incoherent (harmonics above ~500 Hz broke up). Instead it filters a
band-limited pulse train built from F0 (two sines per sample), adds filtered noise, and keeps a
small free-phase branch for transients:
    S = A_h e^{i pi tanh(theta)} STFT(pulses) + A_n Z + A_r e^{i phi}

The network stops at the spectrum; `spec_to_wav` is outside it so an on-device runtime can use
a real FFT instead of an exported DFT matrix, which would cost more than the whole network.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

N_FFT, HOP, SR = 1200, 300, 24000


class ConvNeXt(nn.Module):
    def __init__(self, c, ff, k=7, dil=1):
        super().__init__()
        self.dw = nn.Conv1d(c, c, k, padding=dil * (k // 2), dilation=dil, groups=c)
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


def subframes(y):                                         # [B,4,F40] -> F0, N [B,F80]
    B, _, Fr = y.shape
    y = y.view(B, 2, 2, Fr).permute(0, 1, 3, 2).reshape(B, 2, 2 * Fr)
    return y[:, 0] * 100.0, y[:, 1] * 5.0                 # teacher scale: F0 Hz, N ~[-10, 8]


class ProsodyNet(nn.Module):
    """Text side trained on its own, on whole sentences (Paradee's split): durations and the 80 Hz
    F0/energy curves. Wider, deeper and with dilated context, since TinyKokoro's crop-trained prosody
    head came out ~1 semitone off and flatter than the teacher, and pitch was the remaining gap."""

    def __init__(self, n_tok=178, c=192, layers=6, heads=2, c_pro=192, dils=(1, 2, 4, 8, 1, 2)):
        super().__init__()
        self.text = TextEncoder(n_tok, c, layers, heads, 2 * c)
        self.dur = nn.Sequential(nn.Conv1d(c, c, 3, padding=1), nn.GELU(), nn.Conv1d(c, 1, 1))
        self.pro_in = nn.Conv1d(c, c_pro, 1)
        self.pro = nn.ModuleList([ConvNeXt(c_pro, 2 * c_pro, 7, d) for d in dils])
        self.pro_out = nn.Conv1d(c_pro, 4, 1)

    def encode(self, ids, pad):
        h = self.text(ids, pad)
        return h, self.dur(h).squeeze(1)

    def prosody(self, hf, mask=None):                      # hf [B,C,F40], mask [B,1,F40]
        x = self.pro_in(hf)
        for b in self.pro:
            x = b(x, mask)
        return subframes(self.pro_out(x))


class TinyKokoro(nn.Module):
    def __init__(self, n_tok=178, c_text=128, text_layers=4, heads=2, c_pro=96, pro_blocks=3,
                 c_dec=128, dec_blocks=6, dec_ff=384, n_fft=N_FFT, src=False):
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
        self.bins, self.src = n_fft // 2 + 1, src
        self.head = nn.Linear(c_dec, (5 if src else 2) * self.bins)
        if src:                                            # free branch starts quiet so the source carries voicing
            with torch.no_grad():
                self.head.bias[3 * self.bins: 4 * self.bins] = -4.0

    def encode(self, ids, pad):
        h = self.text(ids, pad)
        logd = self.dur(h).squeeze(1)                      # log(1 + frames)
        return h, logd

    def prosody(self, hf):                                 # hf [B,C,F40] -> [B,F80] x 2
        x = self.pro_in(hf)
        for b in self.pro:
            x = b(x)
        return subframes(self.pro_out(x))

    def decode(self, hf, f0, n):                           # hf [B,C,F40], f0/n [B,F80] -> parts, each [B,bins,F80]
        x = hf.repeat_interleave(2, dim=-1)
        f0f = torch.log1p(F.relu(f0))[:, None] / 5.0
        x = self.dec_in(torch.cat([x, f0f, n[:, None] / 5.0], 1))
        for b in self.dec:
            x = b(x)
        y = self.head(self.dec_norm(x.transpose(1, 2))).transpose(1, 2)
        return y.split(self.bins, 1)                       # (logmag, phase) or (lh, theta, ln, lr, phi)

    @staticmethod
    def expand(h, dur):                                    # h [B,C,L], dur [B,L] ints -> [B,C,F]
        return torch.stack([torch.repeat_interleave(h[b], dur[b], dim=-1) for b in range(h.shape[0])])

    @torch.no_grad()
    def infer(self, ids, speed=1.0):                       # ids [1,L] -> decode parts, f0
        pad = torch.zeros_like(ids, dtype=torch.bool)
        h, logd = self.encode(ids, pad)
        dur = torch.round(torch.expm1(logd) / speed).clamp(min=1).long()
        hf = self.expand(h, dur)
        f0, n = self.prosody(hf)
        return self.decode(hf, f0, n), f0


def spec_to_wav(logmag, phase, n_fft=N_FFT, hop=HOP):
    mag = torch.exp(logmag.clamp(max=7.0))
    spec = torch.polar(mag, phase)
    win = torch.hann_window(n_fft, device=logmag.device)
    return torch.istft(spec, n_fft, hop, n_fft, win, center=True, length=logmag.shape[-1] * hop)


@torch.no_grad()
def pulses(f0, hop=HOP, sr=SR, vthresh=40.0, fmax=11000.0):
    """Zero-phase pulse train with every harmonic below fmax at unit amplitude, from the 80 Hz F0
    curve: sum_{h<=H} cos(h t) = sin((H+1/2) t) / (2 sin(t/2)) - 1/2, so cost doesn't grow with H."""
    f = F.interpolate(f0[:, None], scale_factor=hop, mode="linear", align_corners=False)[:, 0]
    voiced = (f > vthresh).float()
    f = f.clamp(min=vthresh)
    t = 2 * math.pi * torch.frac(torch.cumsum(f.double() / sr, -1)).float()   # wrap: formula is 2pi-periodic
    H = torch.floor(fmax / f)
    s = torch.sin(t / 2)
    e = torch.where(s.abs() > 1e-3, torch.sin((H + 0.5) * t) / (2 * s), H + 0.5) - 0.5
    return e * voiced


def to_wav(parts, f0, n_fft=N_FFT, hop=HOP):
    if len(parts) == 2:
        return spec_to_wav(*parts, n_fft, hop)
    lh, th, ln, lr, phi = parts
    Fr = lh.shape[-1]
    win = torch.hann_window(n_fft, device=lh.device)
    # /(n_fft/4): a unit cosine peaks at sum(win)/2 in the STFT, so A_h is the harmonic's magnitude
    E = torch.stft(pulses(f0, hop), n_fft, hop, n_fft, win, center=True, return_complex=True)[..., :Fr] / (n_fft / 4)
    Z = torch.randn_like(lh) + 1j * torch.randn_like(lh)
    ex = lambda x: torch.exp(x.clamp(max=7.0))
    spec = ex(lh) * torch.polar(torch.ones_like(th), math.pi * torch.tanh(th)) * E + ex(ln) * Z + torch.polar(ex(lr), phi)
    return torch.istft(spec, n_fft, hop, n_fft, win, center=True, length=Fr * hop)


@torch.no_grad()
def infer_pnet(m, pn, ids, speed=1.0):
    """The shipping combination: ProsodyNet durations and F0/energy, TinyKokoro text features and decoder."""
    pad = torch.zeros_like(ids, dtype=torch.bool)
    hp, logd = pn.encode(ids, pad)
    dur = torch.round(torch.expm1(logd) / speed).clamp(min=1).long()
    f0, n = pn.prosody(m.expand(hp, dur))
    return m.decode(m.expand(m.text(ids, pad), dur), f0, n), f0


if __name__ == "__main__":
    import json, sys, time
    torch.set_num_threads(1)
    m = TinyKokoro(**json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}).eval()
    n = sum(p.numel() for p in m.parameters())
    print(f"params {n/1e6:.3f}M")
    for k, mod in (("text", m.text), ("prosody", m.pro), ("decoder", m.dec), ("head", m.head)):
        print(f"  {k:8s} {sum(p.numel() for p in mod.parameters())/1e3:7.1f}K")
    ids = torch.randint(1, 178, (1, 100))
    with torch.no_grad():
        h, logd = m.encode(ids, torch.zeros_like(ids, dtype=torch.bool))
        hf = m.expand(h, torch.full((100,), 4)[None])      # 400 frames = 10 s
        for _ in range(3):
            to_wav(m.decode(hf, *m.prosody(hf)), m.prosody(hf)[0])
        t = time.perf_counter()
        for _ in range(10):
            h, _ = m.encode(ids, torch.zeros_like(ids, dtype=torch.bool))
            hf = m.expand(h, torch.full((100,), 4)[None])
            f0, n = m.prosody(hf)
            w = to_wav(m.decode(hf, f0, n), f0)
        dt = (time.perf_counter() - t) / 10
    print(f"1 thread: {dt*1000:.1f} ms for {w.shape[-1]/SR:.1f} s audio, RTF {dt/(w.shape[-1]/SR):.4f}")
