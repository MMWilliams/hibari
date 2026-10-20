"""hibari command line.

    hibari "Hello world" -o hello.wav
    echo "Read from stdin." | hibari -o out.wav
    hibari "Hello" --play                # winsound / afplay / aplay or paplay
    hibari "Hello" --phonemes            # print phonemes only
    hibari "Hello" --voice emma          # another voice (downloaded once); --list-voices shows them
    hibari --bench
"""
import argparse, os, platform, shutil, subprocess, sys, tempfile, time


def play(path):
    system = platform.system()
    if system == "Windows":
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME)
        return
    for cmd in (["afplay"], ["paplay"], ["aplay", "-q"]) if system != "Darwin" else (["afplay"],):
        if shutil.which(cmd[0]):
            subprocess.run(cmd + [path], check=False)
            return
    sys.exit("no audio player found (afplay / paplay / aplay)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hibari", description="hibari text to speech (English, 24 kHz)")
    ap.add_argument("text", nargs="?", help="text to read; omit or '-' to read stdin")
    ap.add_argument("-o", "--out", help="output WAV path (default: hibari.wav, or none with --play/--phonemes)")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = normal; 1.2 faster, 0.8 slower")
    ap.add_argument("--seed", type=int, help="fix the noise component for reproducible audio")
    ap.add_argument("--play", action="store_true", help="play the result")
    ap.add_argument("--phonemes", action="store_true", help="print phonemes and exit")
    ap.add_argument("--bench", action="store_true", help="time the model on a short paragraph")
    ap.add_argument("--threads", type=int, default=1, help="onnxruntime threads")
    ap.add_argument("--voice", default="heart", help="voice name (default heart); see --list-voices")
    ap.add_argument("--list-voices", action="store_true", help="list the voices and where they are")
    ap.add_argument("--version", action="store_true")
    a = ap.parse_args(argv)
    if a.version:
        from . import __version__
        print(__version__)
        return
    if a.list_voices:
        from .voices import describe
        print(describe())
        return
    from .tts import Hibari
    t0 = time.perf_counter()
    tts = Hibari(voice=a.voice, threads=a.threads)
    if a.bench:
        r = tts.bench(a.text if a.text and a.text != "-" else None, a.speed)
        print(f"load {time.perf_counter() - t0 - r['g2p_ms'] / 1e3 - r['synth_ms'] / 1e3:.2f} s | "
              f"{r['sentences']} sentences, {r['audio_s']:.1f} s audio | G2P {r['g2p_ms']:.1f} ms | "
              f"synthesis {r['synth_ms']:.0f} ms | {r['x_realtime']:.0f}x realtime ({a.threads} thread)")
        return
    text = a.text if a.text and a.text != "-" else sys.stdin.read()
    if not text.strip():
        sys.exit("no text")
    if a.phonemes:
        if (sys.stdout.encoding or "").lower().replace("-", "") != "utf8":      # Windows consoles default to cp1252
            sys.stdout.reconfigure(encoding="utf-8")
        for ps in tts.phonemize(text):
            print(ps)
        return
    out = a.out
    if out is None and not a.play:
        out = "hibari.wav"
    if out is None:
        fd, out = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        tmp = True
    else:
        tmp = False
    tts.save(text, out, a.speed, a.seed)
    if not tmp:
        print(out)
    if a.play:
        play(out)
    if tmp:
        os.unlink(out)


if __name__ == "__main__":
    main()
