import tifffile
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import cm
from matplotlib.colors import Normalize


def main():
    base_path = Path(r"D:\Harrison\cells\analysis\to_process")

    for fp in base_path.glob("*.tif"):

        stem = fp.stem
        stem = "bar05.00"

        raw = tifffile.imread(base_path / f"{stem}.tif")
        masks = tifffile.imread(base_path / "masks" / f"{stem}_masks.tif")
        spots_df = pd.read_csv(base_path / "tracks" / f"{stem}_tracks.csv")

        spots_df["id"] = spots_df.index
        spots_df["parent"] = spots_df.groupby("track_id").shift(1)["id"]
        spots_df["parent"] = spots_df["parent"].fillna(-1).astype(int)
        spots_df["um_x"] = spots_df["px_x"] * 0.66
        spots_df["um_y"] = spots_df["px_y"] * 0.66
        spots_df["t"] = spots_df["frame"] * 5

        spots_df["dif_frame"] = spots_df["frame"] - spots_df.groupby("track_id").shift(1)["frame"]

        for col in ["px_x", "px_y", "um_x", "um_y"]:
            spots_df[f"parent_{col}"] = spots_df.groupby("track_id").shift(1)[col]
            spots_df[f"dif_{col}"] = (spots_df[col] - spots_df[f"parent_{col}"]) / spots_df["dif_frame"]
            spots_df[f"{col}_per_min"] = spots_df[f"dif_{col}"] / 5
            spots_df[f"{col}_per_hour"] = spots_df[f"{col}_per_min"] * 60

        spots_df["um_speed"] = np.sqrt(spots_df["um_x_per_min"] ** 2 + spots_df["um_y_per_min"] ** 2)
        spots_df["avg_speed"] = spots_df["track_id"].map(spots_df.groupby("track_id")["um_speed"].mean())
        spots_df["track_length"] = spots_df["track_id"].map(spots_df.groupby("track_id")["um_speed"].count())

        for c in ["x", "y"]:
            spots_df[f"um_{c}_per_hour"] = spots_df[f"um_{c}_per_hour"].fillna(0.0)
            spots_df[f"um_{c}_per_hour_avg"] = (spots_df.index.map(spots_df.groupby("track_id")[f"um_{c}_per_hour"]
                                                                   .rolling(5, center=True)
                                                                   .mean()
                                                                   .reset_index(0, drop=True))
                                                .fillna(0.0))
            print(spots_df[f"um_{c}_per_hour_avg"].describe())

            out_frames = []

        for i, mask_frame in enumerate(masks):
            ss = spots_df[spots_df["frame"] == i]
            # ss["dif_um_y"] = ss["dif_um_y"].fillna(0.0)
            mapping = {lab: dy for lab, dy in zip(ss["label"], ss[f"um_{c}_per_hour_avg"])}

            mapp_arr = np.zeros(mask_frame.max() + 1)
            for k, v in mapping.items():
                mapp_arr[k] = v

            mapped_array = mapp_arr[mask_frame]
            out_frames.append(mapped_array)

        seg_out = np.array(out_frames).astype(np.float32)

        # Normalize the array and apply a colormap
        norm = Normalize(vmin=-15, vmax=15)
        colormap = np.floor((cm.managua(norm(seg_out)) * 255)).astype(np.uint8)

        colormap[masks == 0] = 0

        colormap = np.moveaxis(colormap, -1, 1)
        # img = np.stack([raw_out]*3, 0)

        tifffile.imwrite(base_path / "test" / f"{stem}_vis_{c}.tif", colormap[:, :3], imagej=True,
                         metadata={"axes": "tcyx", "composite": True})


if __name__ == "__main__":
    main()
