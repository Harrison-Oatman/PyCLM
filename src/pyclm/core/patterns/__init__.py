from .adaptive import (
    AutoExposurePattern,
    DriftCorrectPattern,
    RecenterPattern,
    SettingsSchedulePattern,
)
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
    PopulationClampPattern,
    RelativeClampPattern,
)
from .cell_selection import AllCellsPattern, RandomCellsPattern, SelectCellsPattern
from .composite import SplitPattern, WrappingPattern
from .diagnostic import GameOfLifePattern
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
    ChirpPattern,
    GlobalPulsePattern,
    PulsedPattern,
    PulseSchedulePattern,
)
from .static_patterns import (
    CirclePattern,
    DoseGridPattern,
    FullOnPattern,
    ImageMaskPattern,
    RectanglePattern,
)
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
    # added in 2.0
    "rectangle": RectanglePattern,
    "dose_grid": DoseGridPattern,
    "image_mask": ImageMaskPattern,
    "pulsed": PulsedPattern,
    "chirp": ChirpPattern,
    "all_cells": AllCellsPattern,
    "random_cells": RandomCellsPattern,
    "select_cells": SelectCellsPattern,
    "population_clamp": PopulationClampPattern,
    "relative_clamp": RelativeClampPattern,
    "recenter": RecenterPattern,
    "drift_correct": DriftCorrectPattern,
    "auto_exposure": AutoExposurePattern,
    "settings_schedule": SettingsSchedulePattern,
    "game_of_life": GameOfLifePattern,
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
