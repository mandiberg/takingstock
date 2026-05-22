"""
RunPod dependency installer for install_make_coqui.py.

Before running this script, install PyTorch manually (do this once per session):

    pip install --upgrade --force-reinstall torch torchvision torchaudio \\
        --index-url https://download.pytorch.org/whl/cu124

Then run this script:

    python setup_runpod_coqui.py

Then launch the TTS job:

    python install_make_coqui.py --batch-size 32
"""

import subprocess
import sys


def pip(*args: str) -> None:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--upgrade",
        "--ignore-installed",   # skip distutils-managed system packages (e.g. blinker)
        *args,
    ])


def main() -> None:
    print("=== Installing Coqui TTS dependencies ===")

    # Core Coqui TTS package. Pulls in coqpit, librosa, inflect,
    # anyascii, phonemizer, trainer, etc.
    pip("TTS")

    print("\n=== Installing audio / numeric support packages ===")
    pip(
        "scipy",
        "numpy",
        "soundfile",   # used internally by Coqui for WAV I/O
        "pick",        # batch_collector.py dependency
    )

    # espeak-ng is required by the phonemizer backend that VCTK-VITS uses.
    # Must be installed at OS level, not via pip.
    print("\n=== Installing espeak-ng (required for VITS phonemizer) ===")
    try:
        subprocess.check_call(["apt-get", "install", "-y", "espeak-ng"])
        print("espeak-ng installed.")
    except subprocess.CalledProcessError:
        print(
            "WARNING: apt-get install espeak-ng failed.\n"
            "If you see phonemizer errors at runtime, install manually:\n"
            "  apt-get install -y espeak-ng"
        )

    print("\n=== Verifying GPU visibility ===")
    import torch  # noqa: PLC0415
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU detected: {name} ({vram:.1f} GB VRAM)")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("WARNING: No CUDA GPU detected. Coqui will run on CPU (slower).")

    print("\n=== Pre-downloading VCTK-VITS model weights ===")
    print("Downloads ~150 MB on first run, cached to ~/.local/share/tts/")
    try:
        from TTS.api import TTS
        tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True, gpu=False)
        speakers = tts.speakers if tts.speakers else []
        print(f"Model ready. {len(speakers)} speakers available.")
        del tts
    except Exception as e:
        print(f"Pre-download failed (non-fatal): {e}")
        print("Model will be downloaded on first run of install_make_coqui.py instead.")

    print("\nSetup complete. Run:  python install_make_coqui.py")


if __name__ == "__main__":
    main()
