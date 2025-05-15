import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from cellpose.io import imread
from skimage.measure import regionprops_table
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
import tifffile
import logging
from laptrack import LapTrack

logging.basicConfig(level="INFO")


def run_cellpose(infile):

    model = models.CellposeModel(gpu=True)

    stack = imread(infile)
    stack = [frame for frame in stack]

    masks, flows, styles = model.eval(stack, batch_size=8,)

    masks = np.stack(masks, axis=0)

    return masks


def process_masks(masks):

    collect_spots = []

    for t, frame in enumerate(masks):
        props = regionprops_table(frame, properties=['label', 'area', 'centroid'])

        df = pd.DataFrame(props)
        df["frame"] = t
        df = df.rename(columns={"centroid-0": "px_y", "centroid-1": "px_x"})

        print(df)

        collect_spots.append(df)

    spots_df = pd.concat([s for s in collect_spots], ignore_index=True)

    return spots_df


def track_spots(spots_df):
    max_distance = 20

    lt = LapTrack(
        metric="sqeuclidean",
        # The similarity metric for particles. See `scipy.spatial.distance.cdist` for allowed values.
        splitting_metric="sqeuclidean",
        merging_metric="sqeuclidean",
        gap_closing_metric="sqeuclidean",
        # the square of the cutoff distance for the "sqeuclidean" metric
        cutoff=max_distance ** 2,
        splitting_cutoff=False,  # or False for non-splitting case
        merging_cutoff=False,  # or False for non-merging case
        gap_closing_cutoff=max_distance ** 2,
        gap_closing_max_frame_count=2,
    )

    track_df, split_df, merge_df = lt.predict_dataframe(
        spots_df,
        coordinate_cols=[
            "px_x",
            "px_y",
        ],  # the column names for the coordinates
        frame_col="frame",  # the column name for the frame (default "frame")
        only_coordinate_cols=False,
    )

    return track_df


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("file", type=str, help="Path to the input file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    file = args.file
    masks_outfile = file[:-4] + '_masks.tif'
    spots_outfile = file[:-4] + '_spots.csv'
    tracks_outfile = file[:-4] + '_tracks.csv'

    # cellpose_masks = run_cellpose(file)
    # tifffile.imwrite(masks_outfile, cellpose_masks.astype(np.uint16), imagej=True, metadata={"axes": "tyx"})

    cellpose_masks = tifffile.imread(masks_outfile)

    spots = process_masks(cellpose_masks)
    spots.to_csv(spots_outfile, index=False)

    tracks = track_spots(spots)
    tracks.to_csv(tracks_outfile, index=False)
