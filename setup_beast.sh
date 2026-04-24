#!/usr/bin/env bash
# Local Ubuntu setup for AISCO_code on Beast (RTX 5090 / sm_120).

set -euo pipefail

ENV_NAME="${ENV_NAME:-aisco_env}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
PYTHON_VERSION_REQUIRED="${PYTHON_VERSION_REQUIRED:-3.11.6}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0+cu128}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0+cu128}"
CUDA_HOME_DEFAULT="${CUDA_HOME_DEFAULT:-/usr/local/cuda-12.8}"
TORCH_CUDA_ARCH_LIST_DEFAULT="${TORCH_CUDA_ARCH_LIST_DEFAULT:-12.0}"
PROJECT_EXTRAS="${PROJECT_EXTRAS:-}"
INSTALL_KAOLIN="${INSTALL_KAOLIN:-}"
BUILD_RENDERER="${BUILD_RENDERER:-}"
INSTALL_DISO="${INSTALL_DISO:-}"
BUILD_PYDEVELOPABILITY="${BUILD_PYDEVELOPABILITY:-}"
PYDEVELOPABILITY_STRICT="${PYDEVELOPABILITY_STRICT:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/$ENV_NAME"
VENV_PYTHON="$VENV_DIR/bin/python3"
PYPROJECT_FILE="$PROJECT_ROOT/pyproject.toml"
RENDERER_DIR="$PROJECT_ROOT/subprojects/fixed_renderer"

cd "$PROJECT_ROOT"

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || { echo "Error: missing required command '$1'."; exit 1; }
}

confirm_env_flag() {
    local name="$1"
    local prompt="$2"
    local default="$3"
    local current="${!name}"

    if [[ "$current" == "1" || "$current" == "0" ]]; then
        return 0
    fi

    if [[ ! -t 0 ]]; then
        printf -v "$name" '%s' "$default"
        return 0
    fi

    local answer=""
    read -r -p "$prompt [y/n]: " answer
    if [[ "${answer:-$default}" =~ ^[Yy]$ ]]; then
        printf -v "$name" '%s' "1"
    else
        printf -v "$name" '%s' "0"
    fi
}

normalize_csv() {
    local raw="$1"
    raw="${raw//,/ }"
    echo "$raw" | xargs
}

select_project_extras() {
    if [[ -n "${PROJECT_EXTRAS// }" ]]; then
        PROJECT_EXTRAS="$(normalize_csv "$PROJECT_EXTRAS")"
        return 0
    fi

    if [[ ! -t 0 ]]; then
        PROJECT_EXTRAS=""
        return 0
    fi

    local available
    available="$("$VENV_PYTHON" - "$PYPROJECT_FILE" <<'PY'
import sys
import tomllib
from pathlib import Path

data = tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
extras = sorted(data["project"].get("optional-dependencies", {}))
print(", ".join(extras))
PY
)"

    echo "Optional extras defined in pyproject.toml: ${available:-<none>}"
    read -r -p "Extras to install (comma-separated, blank for none): " PROJECT_EXTRAS
    PROJECT_EXTRAS="$(normalize_csv "$PROJECT_EXTRAS")"
}

install_pyproject_group() {
    local group="$1"
    "$VENV_PYTHON" - "$PYPROJECT_FILE" "$group" <<'PY'
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from packaging.requirements import Requirement

pyproject_path = Path(sys.argv[1])
group = sys.argv[2]

data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
project = data["project"]
optional = project.get("optional-dependencies", {})
skip_pydevelopability = os.environ.get("SKIP_PYDEVELOPABILITY_EXTRA", "1").strip() == "1"

def normalize_name(name: str) -> str:
    return name.lower().replace("-", "_")

project_name = normalize_name(project["name"])

if group == "base":
    requirements = list(project.get("dependencies", []))
else:
    if group not in optional:
        available = ", ".join(sorted(optional))
        raise SystemExit(f"Unknown optional dependency group '{group}'. Available: {available}")
    requirements = list(optional[group])

if not requirements:
    raise SystemExit(0)

overrides = {
    "torch": os.environ.get("TORCH_VERSION", "").strip(),
    "torchvision": os.environ.get("TORCHVISION_VERSION", "").strip(),
}

def expand_requirement(item: str) -> list[str]:
    req = Requirement(item)
    if normalize_name(req.name) == project_name and req.extras:
        expanded: list[str] = []
        for extra_name in sorted(req.extras):
            if skip_pydevelopability and extra_name == "pydevelopability":
                continue
            if extra_name not in optional:
                available = ", ".join(sorted(optional))
                raise SystemExit(f"Unknown nested optional dependency group '{extra_name}'. Available: {available}")
            for nested_item in optional[extra_name]:
                expanded.extend(expand_requirement(nested_item))
        return expanded

    override = overrides.get(req.name)
    if override:
        return [f"{req.name}=={override}"]
    return [item]

resolved = []
seen = set()
for item in requirements:
    for resolved_item in expand_requirement(item):
        if resolved_item not in seen:
            resolved.append(resolved_item)
            seen.add(resolved_item)

index_url = os.environ.get("TORCH_INDEX_URL", "").strip()
torch_requirements = []
other_requirements = []
for item in resolved:
    if Requirement(item).name in {"torch", "torchvision"}:
        torch_requirements.append(item)
    else:
        other_requirements.append(item)

print("Installing", group, "requirements from pyproject.toml:")
for req in resolved:
    print("  ", req)

if other_requirements:
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", *other_requirements], check=True)

if torch_requirements:
    if not index_url:
        raise SystemExit("TORCH_INDEX_URL must be set when installing torch or torchvision overrides.")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "--index-url", index_url, *torch_requirements],
        check=True,
    )
PY
}

echo "Checking Python..."
require_cmd awk
require_cmd sed
require_cmd nvidia-smi

if [[ ! -f "$PYPROJECT_FILE" ]]; then
    echo "Error: pyproject.toml not found at $PYPROJECT_FILE."
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python binary '$PYTHON_BIN' not found."
    exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" --version 2>&1 | awk '{print $2}')"
if [[ "$PYTHON_VERSION" != "$PYTHON_VERSION_REQUIRED" ]]; then
    echo "Error: Python $PYTHON_VERSION_REQUIRED is required, found $PYTHON_VERSION."
    exit 1
fi

if ! "$PYTHON_BIN" -c "import _ctypes" >/dev/null 2>&1; then
    echo "Error: Python is missing _ctypes."
    echo "Rebuild Python $PYTHON_VERSION_REQUIRED with libffi present, then rerun this script."
    exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | sed 's/^ *//;s/ *$//')"
if [[ "$GPU_NAME" != *"RTX 5090"* && "$GPU_NAME" != *"GeForce RTX 5090"* ]]; then
    echo "Error: this script only supports RTX 5090. Detected '$GPU_NAME'."
    exit 1
fi
echo "Detected GPU: $GPU_NAME"

if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
elif [[ -x "$CUDA_HOME_DEFAULT/bin/nvcc" ]]; then
    export CUDA_HOME="$CUDA_HOME_DEFAULT"
else
    echo "Error: nvcc not found. Install CUDA toolkit first or set CUDA_HOME_DEFAULT."
    exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "CUDA_HOME=$CUDA_HOME"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating venv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
export SETUPTOOLS_ENABLE_FEATURES="legacy-editable"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$TORCH_CUDA_ARCH_LIST_DEFAULT}"

echo "Installing base tooling..."
python -m pip install --upgrade pip setuptools wheel build packaging

echo "Installing editable project package..."
python -m pip install --no-deps -e .

echo "Installing base dependencies from pyproject.toml..."
install_pyproject_group base

select_project_extras
if [[ -n "${PROJECT_EXTRAS// }" ]]; then
    for extra in $PROJECT_EXTRAS; do
        if [[ "$extra" == "pydevelopability" ]]; then
            echo "Skipping pydevelopability extra here; it is handled by the optional build step below."
            continue
        fi
        echo "Installing optional dependency group '$extra' from pyproject.toml..."
        install_pyproject_group "$extra"
    done
else
    echo "No optional pyproject extras selected."
fi

pre-commit install || true

confirm_env_flag INSTALL_KAOLIN "Install Kaolin?" "n"
confirm_env_flag BUILD_RENDERER "Build fixed_renderer for RTX 5090?" "y"
confirm_env_flag INSTALL_DISO "Install DISO?" "n"
confirm_env_flag BUILD_PYDEVELOPABILITY "Build pydevelopability C++ extension?" "n"

if [[ "$INSTALL_KAOLIN" == "1" ]]; then
    echo "Installing Kaolin..."
    KAOLIN_INDEX="$("$VENV_PYTHON" - <<'PY'
import torch

torch_version = torch.__version__.split("+")[0]
cuda_version = (torch.version.cuda or "").replace(".", "")
if not cuda_version:
    raise SystemExit("Installed torch does not report a CUDA version; cannot resolve Kaolin wheels.")
print(f"https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-{torch_version}_cu{cuda_version}.html")
PY
)"
    echo "Using Kaolin index: $KAOLIN_INDEX"
    python -m pip uninstall -y kaolin >/dev/null 2>&1 || true
    python -m pip install -f "$KAOLIN_INDEX" kaolin
    python - <<'PY'
import kaolin
print("Kaolin import OK:", getattr(kaolin, "__version__", "unknown"))
PY
fi

if [[ "$BUILD_RENDERER" == "1" ]]; then
    if [[ ! -d "$RENDERER_DIR" || ! -f "$RENDERER_DIR/setup.py" ]]; then
        echo "Error: fixed_renderer not found at $RENDERER_DIR."
        exit 1
    fi

    echo "Building fixed_renderer with TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST..."
    (
        cd "$RENDERER_DIR"
        python setup.py build_ext --inplace
    )

    echo "Verifying fixed_renderer imports..."
    PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
import importlib
from pathlib import Path

root = Path.cwd()
for path in (
    root / "subprojects/fixed_renderer/cuda/load_textures",
    root / "subprojects/fixed_renderer/cuda/rasterize",
    root / "subprojects/fixed_renderer/cuda/create_texture_image",
):
    matches = list(path.parent.glob(path.name + "*.so"))
    if not matches:
        raise SystemExit(f"Missing built extension artifact near {path}")

for name in (
    "subprojects.fixed_renderer.cuda.load_textures",
    "subprojects.fixed_renderer.cuda.rasterize",
    "subprojects.fixed_renderer.cuda.create_texture_image",
    "subprojects.fixed_renderer.renderer",
):
    importlib.import_module(name)
    print("import ok:", name)
PY
fi

if [[ "$INSTALL_DISO" == "1" ]]; then
    echo "Installing DISO..."
    python -m pip install diso --no-build-isolation
    python - <<'PY'
import diso
print("DISO import OK:", getattr(diso, "__file__", "unknown location"))
print("DiffMC available:", hasattr(diso, "DiffMC"))
print("DiffDMC available:", hasattr(diso, "DiffDMC"))
PY
fi

if [[ "$BUILD_PYDEVELOPABILITY" == "1" ]]; then
    echo "Building pydevelopability..."
    python -m pip install --upgrade ".[pydevelopability]"
    PYDEV_ROOT="$PROJECT_ROOT/subprojects/pydevelopability"
    BUILD_DIR="$PYDEV_ROOT/build_manual"
    CC_ROOT="${CC_ROOT:-$PROJECT_ROOT/subprojects/concentrate_curvature}"
    DEV_REPO="${DEV_REPO:-$CC_ROOT/DevelopabilityOfTriangleMeshes}"
    LIBIGL_INCLUDE="${LIBIGL_INCLUDE:-$CC_ROOT/libigl-modern/include}"
    PYBIND11_DIR="$(python -m pybind11 --cmakedir 2>/dev/null || true)"

    if [[ ! -d "$PYBIND11_DIR" || ! -f "$DEV_REPO/include/developableflow/fixed_topology_core.cpp" || ! -f "$LIBIGL_INCLUDE/igl/is_border_vertex.h" ]]; then
        if [[ "$PYDEVELOPABILITY_STRICT" == "1" ]]; then
            echo "Error: missing pydevelopability build prerequisites."
            exit 1
        fi
        echo "Skipping pydevelopability: missing prerequisites."
    else
        rm -rf "$BUILD_DIR"
        mkdir -p "$BUILD_DIR"
        (
            cd "$BUILD_DIR"
            cmake .. \
                -DCMAKE_BUILD_TYPE=Release \
                -Dpybind11_DIR="$PYBIND11_DIR" \
                -DCC_ROOT="$CC_ROOT" \
                -DDEV_REPO="$DEV_REPO" \
                -DLIBIGL_INCLUDE="$LIBIGL_INCLUDE"
            make -j"$(nproc)"
            cp "$(find . -name '_devcore*.so' | head -n1)" "$PYDEV_ROOT/pydevelopability/"
        )
        python -m pip install --no-deps -e .
    fi
fi

echo "Verifying PyTorch CUDA support..."
python - <<'PY'
import sys
import torch

print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)

if not torch.cuda.is_available():
    print("Error: torch.cuda.is_available() is False")
    sys.exit(1)

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
arch_list = torch.cuda.get_arch_list()

print("device", name)
print("capability", cap)
print("arch_list", arch_list)

if "5090" not in name:
    print("Error: expected RTX 5090")
    sys.exit(1)
if cap != (12, 0):
    print(f"Error: expected capability (12, 0), got {cap}")
    sys.exit(1)
if "sm_120" not in arch_list:
    print("Error: installed torch build does not include sm_120 kernels")
    sys.exit(1)

x = torch.tensor([1.0], device="cuda")
print("CUDA smoke test OK:", (x + 1).item())
PY

cat <<EOF

Setup complete.
Activate with: source $ENV_NAME/bin/activate

Installed project extras: ${PROJECT_EXTRAS:-<none>}

Optional extras from pyproject.toml can be installed later with:
  $VENV_PYTHON -m pip install '.[wp1_cfd_ai]'
  $VENV_PYTHON -m pip install '.[wp2_assembler]'
  $VENV_PYTHON -m pip install '.[wp3_solar_estimator]'
  $VENV_PYTHON -m pip install '.[wp4_aeroshell_optimizer]'
  $VENV_PYTHON -m pip install '.[wp7_gui]'
  $VENV_PYTHON -m pip install '.[testing]'
  $VENV_PYTHON -m pip install '.[pydevelopability]'
  $VENV_PYTHON -m pip install '.[all]'
EOF
