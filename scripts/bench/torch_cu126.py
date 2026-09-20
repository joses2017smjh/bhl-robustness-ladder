"""Install or verify an isolated Torch 2.7 CUDA 12.6 overlay for DGX2 V100.

Only the Torch/torchvision/CUDA/Triton packages differ from the existing
LeRobot environment. Downloads occur only when the explicit install mode runs.
The base environment is never modified. Package source is PyTorch's index.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("install", "smoke"))
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    target = args.target.resolve()
    if args.mode == "install":
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing overlay: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".torch-cu126-install-", dir=target.parent))
        base = ["uv", "pip", "install", "--python", sys.executable,
                "--target", str(staging), "--no-deps", "--index-url",
                "https://download.pytorch.org/whl/cu126"]
        subprocess.run(base + ["torch==2.7.0+cu126", "torchvision==0.22.0+cu126"], check=True)
        from packaging.requirements import Requirement
        # Install exact accelerator-library pins from the selected wheel's
        # metadata. Reusing CUDA12.8 libraries by accident defeats isolation.
        torch_dist = next(d for d in metadata.distributions(path=[str(staging)])
                          if d.metadata["Name"].lower() == "torch")
        accelerator = []
        for text in torch_dist.requires or []:
            requirement = Requirement(text)
            if (requirement.name.startswith("nvidia-") or requirement.name == "triton") and (
                    requirement.marker is None or requirement.marker.evaluate()):
                accelerator.append(requirement.name + str(requirement.specifier))
        if not accelerator:
            raise RuntimeError("Selected wheel did not declare CUDA dependencies")
        subprocess.run(base + accelerator, check=True)
        child_env = os.environ.copy()
        child_env["PYTHONPATH"] = str(staging) + os.pathsep + child_env.get("PYTHONPATH", "")
        subprocess.run([sys.executable, __file__, "smoke", "--target", str(staging)],
                       check=True, env=child_env)
        (staging / "bhl-overlay.json").write_text(json.dumps({
            "torch": "2.7.0+cu126", "torchvision": "0.22.0+cu126",
            "accelerator_dependencies": accelerator, "base_python": sys.executable,
            "python_version": sys.version, "cuda_execution_tested": False}, indent=2) + "\n")
        staging.rename(target)
        print(f"TORCH_CU126_INSTALLED {target}", flush=True)
        return

    import torch
    import torchvision
    if not Path(torch.__file__).resolve().is_relative_to(target):
        raise RuntimeError(f"Torch came from {torch.__file__}, not {target}; prepend target to PYTHONPATH")
    if torch.__version__ != "2.7.0+cu126" or torchvision.__version__ != "0.22.0+cu126":
        raise RuntimeError("Wrong Torch/torchvision binary pair")
    arches = torch._C._cuda_getArchFlags().split()
    if "sm_70" not in arches:
        raise RuntimeError(f"V100 architecture missing from binary: {arches}")
    print(json.dumps({"torch": torch.__version__, "torchvision": torchvision.__version__,
        "torch_path": torch.__file__, "compiled_architectures": arches,
        "lerobot": metadata.version("lerobot")}), flush=True)
    if not args.cuda:
        print("TORCH_CU126_CPU_PASS", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA allocation required")
    device = torch.device("cuda")
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.ReLU(),
                                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                torch.nn.Linear(8, 4)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.e-3)
    inputs = torch.rand(4, 3, 32, 32, device=device)
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    q = torch.randn(2, 2, 16, 32, device=device)
    attention = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    assert torch.isfinite(loss) and torch.isfinite(attention).all()
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    import lerobot.policies.smolvla.modeling_smolvla  # noqa: F401
    print(json.dumps({"device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(), "loss": float(loss),
        "checks": ["conv_backward", "AdamW", "SDPA", "save_load", "SmolVLA_import"]}), flush=True)
    print("TORCH_CU126_CUDA_PASS", flush=True)


if __name__ == "__main__":
    main()
