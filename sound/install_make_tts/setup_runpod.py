"""
RunPod dependency installer for install_make_tts.py.

Before running this script, install PyTorch manually (do this once per session):

    pip install --upgrade --force-reinstall torch torchvision torchaudio \\
        --index-url https://download.pytorch.org/whl/cu121

Then run this script to install the remaining dependencies:

    python setup_runpod.py

Then launch the TTS job:

    python install_make_tts.py
"""

import subprocess
import sys


def pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *args])


def main() -> None:
    # PyTorch is installed manually before this script runs (see docstring above).
    # Reinstalling it here would waste ~2 minutes for no benefit.

    print("=== Installing Bark / Transformers dependencies ===")
    pip(
        "transformers",
        "accelerate",   # required by BarkModel for device_map support
        "encodec",      # Bark's audio codec
        "scipy",
        "numpy",
        "pick",
    )

    print("\n=== Attempting Flash Attention 2 install (optional) ===")
    print("This may take a few minutes to compile. Failure is non-fatal.")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "flash-attn", "--no-build-isolation",
        ])
        print("Flash Attention 2 installed — inference will be faster.")
    except subprocess.CalledProcessError:
        print("flash-attn build failed — standard attention will be used (still fast).")

    print("\n=== Verifying GPU visibility ===")
    import torch  # noqa: PLC0415 — intentionally deferred until after install
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU detected: {name} ({vram:.1f} GB VRAM)")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("WARNING: No CUDA GPU detected. Bark will run on CPU (very slow).")

    print("\nSetup complete. Run:  python install_make_tts.py")


if __name__ == "__main__":
    main()
