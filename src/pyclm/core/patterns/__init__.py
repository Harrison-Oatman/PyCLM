from .bar_patterns import (
    BarPatternBase,
    BouncingBarPattern,
    SawToothMethod,
    SpokesPattern,
)
from .cell_intensity_patterns import (
    BinaryNucleusClampModel,
    CenteredImageModel,
    GlobalCycleModel,
)
from .composite import SplitPattern
from .embryo_patterns import InnerPatternMethod, OuterPatternMethod
from .fbc_cell_movement import (
    BounceModel,
    MoveDownModel,
    MoveInModel,
    MoveOutModel,
    PeriodDirectedPattern,
    RotateCcwModel,
)
from .pattern import (
    ROI,
    AcquiredImageRequest,
    CameraProperties,
    DataDock,
    ExperimentState,
    PatternContext,
    PatternMethod,
    renamed_method,
)
from .pulse_patterns import (
    BlinkingBarPattern,
    GlobalPulsePattern,
    PulseSchedulePattern,
)
from .static_patterns import CirclePattern, FullOnPattern
from .wave_patterns import WavePatternBase

known_models = {
    "circle": CirclePattern,
    "split": SplitPattern,
    "bar": BarPatternBase,
    "bar_bounce": BouncingBarPattern,
    "blinking_bar": BlinkingBarPattern,
    "global_pulse": GlobalPulsePattern,
    "pulse_schedule": PulseSchedulePattern,
    "full_on": FullOnPattern,
    "move_rotate": RotateCcwModel,
    "sawtooth": SawToothMethod,
    "move_out": MoveOutModel,
    "move_in": MoveInModel,
    "move_sequence": PeriodDirectedPattern,
    "nucleus_clamp": BinaryNucleusClampModel,
    "target_image": CenteredImageModel,
    "spokes": SpokesPattern,
    "embryo_inner": InnerPatternMethod,
    "embryo_outer": OuterPatternMethod,
    "rings": WavePatternBase,
    # renamed in 2.0: the old names still run, with a warning from pyclm check
    "rotate_ccw": renamed_method(RotateCcwModel, "rotate_ccw"),
    "direction": renamed_method(PeriodDirectedPattern, "direction", channel="545"),
    "binary_nucleus_clamp": renamed_method(
        BinaryNucleusClampModel, "binary_nucleus_clamp", channel="545"
    ),
    "centered_image": renamed_method(
        CenteredImageModel, "centered_image", channel="545"
    ),
    "rotate_bar": renamed_method(SpokesPattern, "rotate_bar"),
    # kept only as fixed cases of another method
    "move_down": MoveDownModel,
    "fb_bounce": BounceModel,
    "global_cycle": GlobalCycleModel,
}
