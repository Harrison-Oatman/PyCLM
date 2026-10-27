from typing import ClassVar

import numpy as np

from .pattern import PatternMethod
from .zoo import ZooMeta


class CirclePattern(PatternMethod):
    """
    Projects a filled circle of radius ``radius_um`` in the center of the camera
    """

    name = "circle"
    renamed_kwargs: ClassVar[dict[str, str]] = {"rad": "radius_um"}
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={"radius_um": 60},
        title="Circle",
        description="Filled circle centered on the camera field of view.",
        group="static",
    )

    def __init__(self, radius_um=1.0, **kwargs):
        """
        :param radius_um: radius of the circle, in um
        """
        super().__init__(**kwargs)

        if radius_um <= 0:
            raise ValueError(
                f"{self.name}: radius_um must be positive, got {radius_um}"
            )
        self.radius_um = radius_um

    def generate(self, context):
        h, w = self.pattern_shape

        center_x = self.pixel_size_um * w / 2.0
        center_y = self.pixel_size_um * h / 2.0

        xx, yy = self.get_um_meshgrid()

        return (
            ((xx - center_x) ** 2 + (yy - center_y) ** 2) < (self.radius_um**2)
        ).astype(np.float16)


class FullOnPattern(PatternMethod):
    """
    Turns on the entire ROI
    """

    name = "full_on"
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={},
        title="Full On",
        description="Illuminates the entire camera field of view.",
        group="static",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, context):
        h, w = self.pattern_shape
        pattern = np.ones((int(h), int(w)), dtype=np.float16)

        return pattern


class RectanglePattern(PatternMethod):
    """
    A filled rectangle ``width_um`` x ``height_um``, its center ``x_um`` /
    ``y_um`` from the field center (+x right, +y down the image), turned
    ``angle_deg`` counterclockwise. A half field is a rectangle as wide as
    half the field, offset by a quarter of it.
    """

    name = "rectangle"
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={
            "width_um": 300,
            "height_um": 120,
            "x_um": 0,
            "y_um": -80,
            "angle_deg": 20,
        },
        title="Rectangle",
        description="A filled rectangle given in um from the field center, optionally turned.",
        group="static",
    )

    def __init__(
        self,
        width_um=None,
        height_um=None,
        x_um=0.0,
        y_um=0.0,
        angle_deg=0.0,
        **kwargs,
    ):
        """
        :param width_um: size along the rectangle's own x axis, in um
        :param height_um: size along its own y axis, in um
        :param x_um: offset of its center from the field center, in um (+ is right)
        :param y_um: offset of its center from the field center, in um (+ is down)
        :param angle_deg: rotation, counterclockwise, in degrees
        """
        super().__init__(**kwargs)
        for key, value in (("width_um", width_um), ("height_um", height_um)):
            if value is None or value <= 0:
                raise ValueError(f"{self.name}: {key} must be positive, got {value}")
        self.width_um = float(width_um)
        self.height_um = float(height_um)
        self.x_um = float(x_um)
        self.y_um = float(y_um)
        self.angle_deg = float(angle_deg)

    def generate(self, context):
        xx, yy = self.get_um_meshgrid()
        cx, cy = self.center_um()
        dx = xx - (cx + self.x_um)
        dy = yy - (cy + self.y_um)
        # row 0 is up, so a counterclockwise turn has its x axis at (cos, -sin)
        a = np.radians(self.angle_deg)
        u = dx * np.cos(a) - dy * np.sin(a)
        v = dx * np.sin(a) + dy * np.cos(a)
        inside = (np.abs(u) <= self.width_um / 2) & (np.abs(v) <= self.height_um / 2)
        return inside.astype(np.float16)


class DoseGridPattern(PatternMethod):
    """
    Several doses in one field: the field is cut into ``rows`` x ``cols``
    blocks, ``gap_um`` apart (half a gap at the field edge), and block ``i``
    (row-major) gets ``doses[i]``, the list cycled. ``spot_um`` lights a
    centered disc of that diameter in each block instead of the whole block.
    A dose between 0 and 1 is delivered by ``dose_mode`` (``"gray"``,
    ``"dither"``, or ``"duty"`` over ``period_min``).
    """

    name = "dose_grid"
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={
            "rows": 2,
            "cols": 2,
            "doses": [0.0, 0.33, 0.66, 1.0],
            "gap_um": 50,
        },
        title="Dose Grid",
        description="A grid of blocks (or spots), each at its own dose: quadrants, or a dose-response array.",
        group="static",
    )

    def __init__(
        self,
        rows=2,
        cols=2,
        doses=(1.0,),
        gap_um=50.0,
        spot_um=None,
        dose_mode="gray",
        period_min=10.0,
        **kwargs,
    ):
        """
        :param rows: blocks down the field
        :param cols: blocks across the field
        :param doses: dose of each block, 0-1, row-major, cycled
        :param gap_um: dark gap between neighboring blocks, in um
        :param spot_um: diameter of a disc lit in each block (None: the whole block)
        :param dose_mode: "gray", "dither" or "duty"
        :param period_min: the period of a "duty" dose, in minutes
        """
        super().__init__(**kwargs)
        from .dose import check_dose_mode

        if int(rows) < 1 or int(cols) < 1:
            raise ValueError(f"{self.name}: rows and cols must be at least 1")
        doses = [float(d) for d in np.atleast_1d(doses)]
        if not doses or not all(0.0 <= d <= 1.0 for d in doses):
            raise ValueError(f"{self.name}: doses must be a list of values in 0-1")
        if gap_um < 0:
            raise ValueError(f"{self.name}: gap_um must not be negative")
        if spot_um is not None and spot_um <= 0:
            raise ValueError(f"{self.name}: spot_um must be positive")
        check_dose_mode(self.name, dose_mode, period_min)
        self.rows = int(rows)
        self.cols = int(cols)
        self.doses = doses
        self.gap_um = float(gap_um)
        self.spot_um = None if spot_um is None else float(spot_um)
        self.dose_mode = dose_mode
        self.period_min = float(period_min)

    def dose_map(self) -> np.ndarray:
        """Every pixel's dose, before ``dose_mode``."""
        xx, yy = self.get_um_meshgrid()
        h, w = self.pattern_shape
        block_w = w * self.pixel_size_um / self.cols
        block_h = h * self.pixel_size_um / self.rows
        # pixel centers, so a block edge splits pixels evenly
        xc = xx + self.pixel_size_um / 2
        yc = yy + self.pixel_size_um / 2
        col = np.minimum((xc // block_w).astype(int), self.cols - 1)
        row = np.minimum((yc // block_h).astype(int), self.rows - 1)
        lx = xc - (col + 0.5) * block_w  # offset from the block center
        ly = yc - (row + 0.5) * block_h
        if self.spot_um is None:
            inside = (np.abs(lx) <= block_w / 2 - self.gap_um / 2) & (
                np.abs(ly) <= block_h / 2 - self.gap_um / 2
            )
        else:
            inside = lx**2 + ly**2 <= (self.spot_um / 2) ** 2
        doses = np.array(
            [self.doses[i % len(self.doses)] for i in range(self.rows * self.cols)],
            dtype=np.float32,
        )
        return np.where(inside, doses[row * self.cols + col], 0.0).astype(np.float32)

    def generate(self, context):
        from .dose import deliver

        return deliver(
            self.dose_map(), self.dose_mode, context.time / 60, self.period_min
        )


class ImageMaskPattern(PatternMethod):
    """
    Projects an image (a TIF) as the pattern: scaled to 0-1 between its
    darkest and brightest pixel, and either centered at the camera's pixel
    size (``fit = "center"``, cropped or padded with dark) or resized to fill
    the field (``fit = "stretch"``). With ``threshold`` (0-1 of that scale)
    the mask is binary: lit where the image is above it; without, graded.
    """

    name = "image_mask"
    zoo_meta = ZooMeta(
        source="mdck",
        kwargs={"path": "target_shield.tif", "fit": "stretch", "threshold": 0.5},
        title="Image Mask",
        description="Any image (a TIF) projected as the pattern, centered or stretched to the field, graded or thresholded.",
        group="static",
    )

    FITS = ("center", "stretch")

    def __init__(self, path=None, threshold=None, fit="center", **kwargs):
        """
        :param path: the image, a TIF (the first plane of a stack)
        :param threshold: light only where the scaled image is above this
            (0-1); None: project the graded image
        :param fit: "center" (at camera pixel size) or "stretch" (to the field)
        """
        super().__init__(**kwargs)
        import tifffile

        if path is None:
            raise ValueError(f"{self.name}: path (the TIF to project) is required")
        if fit not in self.FITS:
            raise ValueError(
                f"{self.name}: fit must be 'center' or 'stretch', got {fit!r}"
            )
        if threshold is not None and not 0.0 <= threshold < 1.0:
            raise ValueError(f"{self.name}: threshold must be in 0-1, got {threshold}")
        img = np.asarray(tifffile.imread(path), dtype=np.float32)
        while img.ndim > 2:
            img = img[0]
        lo, hi = float(img.min()), float(img.max())
        self.image = (img - lo) / (hi - lo) if hi > lo else (img > 0).astype(np.float32)
        self.path = path
        self.threshold = threshold
        self.fit = fit

    def generate(self, context):
        h, w = (int(v) for v in self.pattern_shape)
        img = self.image
        if self.fit == "stretch":
            from skimage.transform import resize

            out = resize(img, (h, w), order=1, preserve_range=True, anti_aliasing=True)
        else:
            out = np.zeros((h, w), np.float32)
            ih, iw = img.shape
            # the overlap of the image and the field, both centered
            oy, ox = (h - ih) // 2, (w - iw) // 2
            ys, xs = max(0, -oy), max(0, -ox)
            yd, xd = max(0, oy), max(0, ox)
            n_y, n_x = min(ih - ys, h - yd), min(iw - xs, w - xd)
            out[yd : yd + n_y, xd : xd + n_x] = img[ys : ys + n_y, xs : xs + n_x]
        if self.threshold is not None:
            out = out > self.threshold
        return np.clip(out, 0.0, 1.0).astype(np.float32)
