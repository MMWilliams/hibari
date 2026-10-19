# kokoro-tiny session report (2026-10-18)

## What changed
- GAN-phase throughput, batch 16, GPU 0 (RTX, Windows WDDM): 1.8 -> ~7.3 it/s in the real trainer (11.5 it/s in the isolated benchmark, scripts/profile_gan.py).
  Profile: the step was launch-bound (~3300 CUDA launches/step at ~17 us, "Command Buffer Full" stalls), GPU time ~127 ms; MRD (multi-resolution spectrogram discriminator) was ~70% of GPU time.
  Fixes in training/train_tiny.py + styletts2_gan/: bf16 autocast for discriminator and generator GAN losses; MRD hops fft/4 (Vocos) instead of UnivNet 120/240/50; real+fake batched through each discriminator (one forward instead of two); discriminators see a random 0.8 s sub-window of the 1.6 s crop (mel/STFT/silence losses still use the full crop); TPRLS off by default (--tprls restores, costs ~14%); no per-step GPU sync in the logging; it/s is now per 100-step interval.
  channels_last and cudnn.benchmark gave nothing. Training restarted with --resume from the step-2000 checkpoint (pre-GAN weights) on 17,197 rows.
- ONNX export (export/export_onnx.py): ids [1,L] -> logmag, phase [1,601,F], dynamic L and F, durations and token->frame expansion in-graph (CumSum/Range/Less/ReduceSum/Gather). Attention is written out by hand (nn.MultiheadAttention traces with a fixed L); torch constant folding is off (it mis-folded the dynamic duration path) and Linear weight transposes are folded by the script instead.
- export/infer_numpy.py: onnxruntime + numpy inference, numpy irfft overlap-add iSTFT (matches torch.istft to max|err| 1.5e-7 on real model output, export/test_istft.py). export/vocab.json = Kokoro phoneme vocab.
- int8 dynamic quantization, MatMul only: ConvInteger ran 2x slower than fp32 Conv in onnxruntime 1.29 x86.

## Measured (desktop, 1 thread, 64 s of eval audio, trainer + dump competing for CPU)
| model | size | RTF total | net | numpy iSTFT |
| fp32 export/tiny.onnx | 5.43 MB | 0.0037 | 0.0018 | 0.0019 |
| int8 export/tiny.int8.onnx | 2.53 MB | 0.0033 | 0.0015 | 0.0018 |
| int8 incl. Conv (not kept) | 1.99 MB | 0.0087 | 0.0062 | |
int8 vs fp32 logmag: mean abs diff 0.012 (log units); no listening check yet.

## Training health
