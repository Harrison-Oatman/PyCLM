import pandas as pd
import numpy as np
from pathlib import Path
from skimage.measure import regionprops_table
import tifffile
from skimage.transform import downscale_local_mean
from tqdm import tqdm


def process_file():
    pass

def find_in_folder(folder: Path, stem: str) -> None | Path:

    matches = [fp for fp in folder.iterdir() if str(fp.stem).startswith(stem)]

    if len(matches) == 0:
        return None

    return matches[0]


def main():
    base_dir = Path(r"D:\Harrison\RTx3 imaging\2025-08-20 bars 3")
    channel = "638"

    for fp in Path(base_dir).glob("*.hdf5"):

        stem = fp.stem

        mask_path = find_in_folder(base_dir / "masks", stem)
        img_path = find_in_folder(base_dir / channel , stem)
        track_path = find_in_folder(base_dir / "tracks", stem)

        if not mask_path or not img_path or not track_path:
            continue

        df = pd.read_csv(track_path)
        seg = tifffile.imread(mask_path)
        raw = tifffile.imread(img_path)

        df = df.set_index(["frame", "label"])

        df[f"chan_{channel}"] = np.nan

        for frame in tqdm(range(len(seg))):

            raw_frame = downscale_local_mean(raw[frame], (2, 2)).astype(np.uint16)
            seg_frame = seg[frame]

            # print(raw_frame.shape)
            # print(seg_frame.shape)

            props = regionprops_table(seg_frame, raw_frame, properties=["label", "intensity_mean"])

            df_idx = [(frame, label) for label in props["label"]]

            df.loc[df_idx, "intensity_mean"] = props["intensity_mean"]

            # print(props)

        df.to_csv(track_path)


if __name__ == "__main__":
    main()