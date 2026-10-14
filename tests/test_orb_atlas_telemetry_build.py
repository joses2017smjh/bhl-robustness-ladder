"""Static checks for the opt-in read-only Atlas telemetry wrapper relink."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRISTINE_EXCERPT = (
    "    int GetTrackingState();\n"
    "    std::vector<MapPoint*> GetTrackedMapPoints();\n"
    "    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();\n"
)


def build_module():
    path = ROOT / "scripts/native/orb_diagnostics_build.py"
    spec = importlib.util.spec_from_file_location("test_orb_atlas_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_patch_is_unchanged_tracker_accessor_only():
    mod = build_module()
    patched = mod.patch_system_header(PRISTINE_EXCERPT)
    assert patched.count("BhlReadOnlyTracker") == 1
    assert "BhlTelemetryAtlas" not in patched
    assert patched.replace(mod.TRACKER_ACCESSOR, "") == PRISTINE_EXCERPT


def test_atlas_patch_adds_only_inline_accessors_before_pinned_anchor():
    mod = build_module()
    patched = mod.patch_system_header(PRISTINE_EXCERPT, atlas_telemetry=True)
    assert patched.replace(mod.TRACKER_ACCESSOR, "").replace(mod.ATLAS_ACCESSOR, "") == PRISTINE_EXCERPT
    assert patched.index("BhlTelemetryAtlas") < patched.index("GetTrackedMapPoints")
    # Inline member functions only: no new data members, virtuals or statics.
    for accessor in (mod.TRACKER_ACCESSOR, mod.ATLAS_ACCESSOR):
        assert "virtual" not in accessor and "static" not in accessor and "{ return mp" in accessor


@pytest.mark.parametrize("text", [PRISTINE_EXCERPT.replace("GetTrackedMapPoints", "Other"),
                                  PRISTINE_EXCERPT * 2,
                                  build_module().TRACKER_ACCESSOR + PRISTINE_EXCERPT])
def test_patch_refuses_changed_duplicate_or_already_patched_header(text):
    with pytest.raises(ValueError, match="anchor"):
        build_module().patch_system_header(text, atlas_telemetry=True)


def test_wrapper_emits_atlas_snapshot_only_under_opt_in_flag():
    source = (ROOT / "scripts/native/orb_runner.cc").read_text()
    start = source.index("#ifdef BHL_ORB_ATLAS_TELEMETRY\n      // Wrapper-only Atlas snapshot")
    block = source[start:source.index("#endif", start)]
    for call in ("slam.MapChanged()", "BhlTelemetryAtlas()", "GetCurrentMap()", "CountMaps()",
                 "GetLastBigChangeIdx()", "GetInitKFid()", "KeyFramesInMap()", "GetMaxKFid()"):
        assert call in block
    # No mutating Atlas/Map/System entry points are called by the wrapper code
    # (comments explaining upstream behaviour are excluded from this check).
    code = "\n".join(line.split("//", 1)[0] for line in source.splitlines())
    for forbidden in ("ChangeMap(", "CreateNewMap(", "clearMap(", "SetMapBad(", "InformNewBigChange(",
                      "ResetActiveMap(", "Reset(", "SetImuInitialized(", "ApplyScaledRotation(",
                      "ActivateLocalizationMode(", "ChangeDataset("):
        assert forbidden not in code
    assert '"atlas":{\\"schema\\"' not in source  # emitted with C++ escaping below
    assert '\\"atlas\\":{\\"schema\\":\\"bhl-orb-atlas-telemetry-v1\\"' in source
    assert "atlas-telemetry-v1" in source and "BHL_ORB_VERSION_SUFFIX" in source
