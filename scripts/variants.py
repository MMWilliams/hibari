"""Write candidate student configs derived from the 7M config, then time each with random weights."""
import json, copy, sys, os, subprocess
base = json.load(open("upstream/kokoro-7m/config.json", encoding="utf-8"))
def v(name, **kw):
    c = copy.deepcopy(base)
    for k, val in kw.items():
        d = c
        *path, last = k.split("__")
        for p in path: d = d[p]
        d[last] = val
    json.dump(c, open(f"configs/{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
# one upsample stage: 80 Hz -> 800 Hz, iSTFT hop 30 / n_fft 120 does the last 30x
one = dict(istftnet__upsample_rates=[10], istftnet__upsample_kernel_sizes=[20],
           istftnet__gen_istft_n_fft=120, istftnet__gen_istft_hop_size=30,
           istftnet__resblock_kernel_sizes=[3, 7], istftnet__resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]])
v("s1_onestage", **one)
v("s2_onestage_slim", **one, hidden_dim=128, n_layer=2, istftnet__hidden_channels=192,
  istftnet__out_channels=96, plbert__hidden_size=128, plbert__num_attention_heads=2,
  plbert__intermediate_size=512, plbert__num_hidden_layers=6)
v("s3_onestage_slim_c96", **one, hidden_dim=128, n_layer=2, istftnet__hidden_channels=192,
  istftnet__out_channels=96, istftnet__upsample_initial_channel=96, plbert__hidden_size=128,
  plbert__num_attention_heads=2, plbert__intermediate_size=512, plbert__num_hidden_layers=6)
slim = dict(hidden_dim=128, n_layer=2, istftnet__hidden_channels=192, istftnet__out_channels=96,
            istftnet__upsample_initial_channel=96, plbert__hidden_size=128, plbert__num_attention_heads=2,
            plbert__intermediate_size=512, plbert__num_hidden_layers=6)
v("s4_h0", **one, **slim, istftnet__harmonic_num=0, istftnet__noise_res_kernel=7)
v("s5_h0_k35", **{**one, **slim, "istftnet__harmonic_num": 0, "istftnet__noise_res_kernel": 5,
  "istftnet__resblock_kernel_sizes": [3, 5]})
v("s6_h0_single", **{**one, **slim, "istftnet__harmonic_num": 0, "istftnet__noise_res_kernel": 5,
  "istftnet__resblock_kernel_sizes": [5], "istftnet__resblock_dilation_sizes": [[1, 3, 5]]})
v("tiny_v1", **one, **{**slim, "istftnet__hidden_channels": 128}, istftnet__harmonic_num=0, istftnet__noise_res_kernel=7)
