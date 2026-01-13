from .bar_patterns import BarPatternBase, BouncingBarPattern, SawToothMethod
from .feedback_control_patterns import (
    BounceModel,
    MoveDownModel,
    MoveInModel,
    MoveOutModel,
    RotateCcwModel,
)
from .ktr_patterns import BinaryNucleusClampModel, CenteredImageModel, GlobalCycleModel
from .pattern import (
    ROI,
    AcquiredImageRequest,
    CameraProperties,
    DataDock,
    PatternContext,
    PatternMethod,
    PatternMethodReturnsSLM,
    PatternReview,
)
from .static_patterns import CirclePattern, FullOnPattern

known_models = {
    "circle": CirclePattern,
    "bar": BarPatternBase,
    "pattern_review": PatternReview,
    "bar_bounce": BouncingBarPattern,
    "full_on": FullOnPattern,
    "rotate_ccw": RotateCcwModel,
    "sawtooth": SawToothMethod,
    "move_out": MoveOutModel,
    "move_in": MoveInModel,
    "move_down": MoveDownModel,
    "fb_bounce": BounceModel,
    "binary_nucleus_clamp": BinaryNucleusClampModel,
    "global_cycle": GlobalCycleModel,
    "centered_image": CenteredImageModel,
}
