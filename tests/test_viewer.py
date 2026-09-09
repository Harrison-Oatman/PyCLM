"""
The napari viewer with positions as an axis: two outputs become one layer
per channel stacked over positions, the positions list / minimap / keys
move the position slider, "follow the run" tracks status.json, and contrast
stays where the user put it.

napari needs a real OpenGL context, which the offscreen Qt platform the other
GUI tests use does not give on Windows, so the scenario runs in a subprocess
with a hidden window on the native platform and reports back as JSON. It is
skipped where no display is available.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from helpers import make_experiment, make_plan, make_schedule

pytest.importorskip("napari")

from test_storage import AFFINE, SLM, acquire, core_with

from pyclm.core.storage import OMEZarrWriter

STEPS = 4


@pytest.fixture
def two_positions(tmp_path):
    """Two experiments of the same cadence, two imaging frames each, in one directory."""
    a = make_experiment("a.00", every_t=2, stim_every_t=1)
    b = make_experiment("b.00", every_t=2, stim_every_t=1)
    plan = make_plan(make_schedule([a, b], steps=STEPS))
    writer = OMEZarrWriter()
    writer.open(plan, core_with(), tmp_path, AFFINE, SLM)
    for t in range(STEPS):
        for name, base in (("a.00", 10), ("b.00", 1000)):
            pat = np.full(SLM, 1, np.uint8)
            writer.write_frame(
                acquire(plan, name, t, "DMD", np.full((16, 16), 9, np.uint16), pat, "p")
            )
            if t % 2 == 0:
                writer.write_frame(
                    acquire(
                        plan, name, t, "545", np.full((16, 16), base + t, np.uint16)
                    )
                )
    writer.close()
    (tmp_path / "status.json").write_text(
        '{"t": 3, "timepoints": 4, "done": true, "current_experiment": "b.00", '
        '"experiments": {}}'
    )
    return tmp_path


# what the subprocess does with the viewer: every observation into one dict
SCENARIO = r"""
import json, sys
from pathlib import Path
from pyclm.gui.gui_controller import ViewerApp

d = Path(sys.argv[1])
specs = [(str(d / "a.00.zarr"), "imaging/545"), (str(d / "b.00.zarr"), "imaging/545")]
app = ViewerApp(specs, d / "status.json", d, show=False)
out = {}
stack = app.stacks.channels["imaging/545"]
out["positions"] = app.positions
out["channels"] = list(app.stacks.channels)
out["patterns"] = list(app.stacks.patterns)
out["shape"] = list(stack.data.shape)
out["pattern_shape"] = list(app.stacks.patterns["imaging/pattern"].data.shape[:2])
out["a_t0"] = int(stack.data[0, 0, 0, 0])
out["b_t1"] = int(stack.data[1, 1, 0, 0])
out["axis_labels"] = list(app.viewer.dims.axis_labels)
out["active_is_channel"] = app.viewer.layers.selection.active is stack.layer

# choosing a position: list, minimap, keys, slider all agree
steps = [app.current_position]
app.controls.list.setCurrentRow(1); steps.append(app.current_position)
app.overview.minimap.positionClicked.emit("a.00"); steps.append(app.current_position)
app.next_position(); steps.append(app.current_position)
app.next_position(); steps.append(app.current_position)   # wraps
app.previous_position(); steps.append(app.current_position)
app.viewer.dims.set_current_step(0, 0)
steps.append(app.controls.list.currentRow())
out["steps"] = steps

# follow the run: status.json says b.00
app.controls.follow.setChecked(True)
app.refresh()
out["followed"] = app.current_position
app.controls.follow.setChecked(False)

# contrast stays where the user put it
c = {}
c["auto_before"] = stack.auto_contrast
stack.layer.contrast_limits = (0, 5000)
c["auto_after_slider"] = stack.auto_contrast
app.refresh()
c["checkbox_after_refresh"] = app.controls.auto_contrast.isChecked()
c["limits_kept"] = list(stack.layer.contrast_limits)
app.controls.normalize.click()
c["limits_after_normalize"] = list(stack.layer.contrast_limits)
stack.layer.contrast_limits = (0, 5000)
app.controls.auto_contrast.setChecked(True)
c["auto_after_checkbox"] = stack.auto_contrast
c["limits_after_checkbox"] = list(stack.layer.contrast_limits)
out["contrast"] = c

app.close()
app.viewer.close()
print("RESULT " + json.dumps(out))
"""


def run_scenario(directory: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parent), env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        [sys.executable, "-c", SCENARIO, str(directory)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if proc.returncode != 0:
        if "GLError" in proc.stderr or "OpenGL" in proc.stderr:
            pytest.skip("no OpenGL context for napari here")
        raise AssertionError(f"viewer scenario failed:\n{proc.stdout}\n{proc.stderr}")
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


def test_viewer_positions_as_an_axis(two_positions):
    out = run_scenario(two_positions)

    # one layer per channel, stacked (position, t, y, x), plus the pattern overlay
    assert out["positions"] == ["a.00", "b.00"]
    assert out["channels"] == ["imaging/545"]
    assert out["patterns"] == ["imaging/pattern"]
    assert out["shape"] == [2, 2, 16, 16]
    assert out["pattern_shape"] == [2, 2]
    assert (out["a_t0"], out["b_t1"]) == (10, 1002)
    assert out["axis_labels"] == ["position", "t", "y", "x"]
    assert out["active_is_channel"]

    # list -> 1, minimap a.00 -> 0, next -> 1, next wraps -> 0, previous -> 1,
    # then the slider set to 0 moves the list to row 0
    assert out["steps"] == [0, 1, 0, 1, 0, 1, 0]
    assert out["followed"] == 1

    c = out["contrast"]
    assert c["auto_before"]
    assert not c["auto_after_slider"]
    assert not c["checkbox_after_refresh"]
    assert c["limits_kept"] == [0, 5000]
    assert c["limits_after_normalize"] != [0, 5000]
    assert c["auto_after_checkbox"]
    assert c["limits_after_checkbox"] != [0, 5000]
