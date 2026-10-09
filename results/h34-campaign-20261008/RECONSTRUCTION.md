# Exact H3/H4 input reconstruction

The compact bundles preserve the exact prepublication source and teacher inputs used to freeze each campaign. Each contains 12 source overlays, all four teacher files, and the original protocol, full manifest, and intake receipt. The other 799 source files come from public base commit `ba73008a08c3d67c8870575ac9095ae2cec21354` and its recursively pinned submodules.

| Campaign | Bundle | SHA256 | Verification |
| --- | --- | --- | --- |
| H3 v1 | [reconstruction-inputs.tar.gz](h3/v1/reconstruction-inputs.tar.gz) (1,496,965 bytes) | `5c2033b6d267525a28ebb7a349a35e5f8d01357908e611d4a5e67d748f516460` | [815/815 files](h3/v1/reconstruction-verification.json) |
| H4 v2 | [reconstruction-inputs.tar.gz](h4/v2/reconstruction-inputs.tar.gz) (1,699,095 bytes) | `f3c82c0ec9acdbb083e90618b8ad6bf48b663fe95ee6705deb8fd66d2e067e53` | [815/815 files](h4/v2/reconstruction-verification.json) |

Use a separate directory for each version. Download the selected bundle into the current directory, verify its SHA256 above, then run:

```bash
mkdir campaign
git clone --no-checkout https://github.com/joses2017smjh/bhl-robustness-ladder.git campaign/source
git -C campaign/source checkout ba73008a08c3d67c8870575ac9095ae2cec21354
git -C campaign/source -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive
tar -xzf reconstruction-inputs.tar.gz -C campaign
python3 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('campaign')
manifest = json.loads((root / 'manifest.json').read_text())
assert len(manifest) == 815
for relative, expected in manifest.items():
    data = (root / relative).read_bytes()
    assert len(data) == expected['bytes'], relative
    assert hashlib.sha256(data).hexdigest() == expected['sha256'], relative
intake = json.loads((root / 'intake.json').read_text())
protocol_sha = hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest()
assert protocol_sha == intake['protocol_sha256']
print('PASS: all 815 frozen file sizes and SHA256 values; original protocol verified')
PY
```

The submodule pins are `984741a3623c93b0583ccfdc479f1f8b1c4d900e` (Berkeley-Humanoid-Lite), `fc90fedd008b1e56a22e3c5221548d6b24f49707` (assets), and `652777cc7c49884e7cd7ddfada758dc1979bf627` (lowlevel). Git uses these pins from the base commit when updating recursively.

Local validation materialized unchanged files directly from those immutable Git blobs, overlaid bytes read back from each compact tar, and checked all 815 entries on disk. It also verified every original archive file and the original archive SHA256. This reconstructs file contents, not the original tar/gzip bytes; the original archive digests remain in each intake receipt. External v51 Python dependencies, Isaac runtime/SIF, and machine-specific paths are not supplied by these bundles. Reconstruction verifies inputs and does not establish completed training or scientific results.
