"""Report whether the LUMI PyTorch container has the repo runtime dependencies."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Requirement:
    module: str
    package: str
    required: bool = True


REQUIREMENTS = [
    Requirement("torch", "torch"),
    Requirement("torch.distributed.run", "torch"),
    Requirement("yaml", "PyYAML"),
    Requirement("tqdm", "tqdm"),
    Requirement("datasets", "datasets"),
    Requirement("sentencepiece", "sentencepiece"),
    Requirement("zstandard", "zstandard"),
    Requirement("numpy", "numpy"),
]


def package_version(package: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    results = []
    missing_required = []
    for requirement in REQUIREMENTS:
        found = importlib.util.find_spec(requirement.module) is not None
        version = package_version(requirement.package)
        results.append(
            {
                "module": requirement.module,
                "package": requirement.package,
                "found": found,
                "version": version,
                "required": requirement.required,
            }
        )
        if requirement.required and not found:
            missing_required.append(requirement.module)

    torch_info = {}
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch_info = {
            "torch_version": torch.__version__,
            "hip_version": getattr(torch.version, "hip", None),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }

    print(json.dumps({"requirements": results, "torch": torch_info}, indent=2))
    if missing_required:
        print("Missing required modules: " + ", ".join(missing_required), file=sys.stderr)
        return 1
    print("Container has all required Python modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
