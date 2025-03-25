import tifffile
import nd2
import numpy as np
from pathlib import Path
from skimage.transform import downscale_local_mean
from tqdm import tqdm
import cv2


def main(nd2path, downscale_factor=1):
    with nd2.ND2File(nd2path) as images:
        position_names = [pt.name for pt in images.experiment[1].parameters.points]
        nd2path = Path(nd2path)
        outpath = nd2path.parent / "series"
        outpath.mkdir(exist_ok=True)

        darr = images.to_dask()

        for i, position in tqdm(enumerate(position_names)):

            outfile = outpath / f"{nd2path.stem}_{position}.tif"

            data = darr[:, i].compute()
            print(data.shape)

            if downscale_factor > 1:
                data = downscale_local_mean(data, (1, 1, downscale_factor, downscale_factor))

            data = np.round(data).astype(np.uint16)

            print(f"Saving {outfile}")
            tifffile.imwrite(outfile, data, imagej=True, metadata={"axes": "TCYX"})




if __name__ == "__main__":
    main(r"D:\FeedbackControl\data\calibration\calibration001_crop.nd2", 4)
