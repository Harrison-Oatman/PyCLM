import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from cellpose.io import imread
from skimage.measure import regionprops_table
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
import tifffile

def run_cellpose(infile):

    model = models.CellposeModel(gpu=True)

    stack = imread(infile)
    stack = [frame for frame in stack]

    masks, flows, styles = model.eval(stack)

    return masks

def process_masks(masks):

    collect_spots = []

    for frame in masks:
        props = regionprops_table(frame, properties=['label', 'area', 'centroid'])
        collect_spots.append(props)

    spots_df = pd.concat([pd.DataFrame(s) for s in collect_spots], ignore_index=True)

    return spots_df


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("file", type=str, help="Path to the input file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    file = args.file
    masks_outfile = Path(file).with_suffix('_masks.tif')
    spots_outfile = Path(file).with_suffix('_spots.csv')

    cellpose_masks = run_cellpose(file)
    tifffile.imwrite(masks_outfile, cellpose_masks.astype(np.uint16), imagej=True, metadata={"axes": "tyx"})

    spots = process_masks(cellpose_masks)
    spots.to_csv(spots_outfile, index=False)
