import pandas as pd

from tqdm import tqdm
from xml.etree import ElementTree as ET
from networkx import Graph, DiGraph
import numpy as np


def process_trackmate_tree(tree: ET) -> (pd.DataFrame, DiGraph):
    """
    Process trackmate tree
    :param tree: ElementTree object from trackmate xml file
    :return:
    """

    graph = DiGraph()
    root = tree.getroot()

    # iterate through spot elements and collect attributes
    spots = root.find("Model").find("AllSpots")
    spots_collect = []

    for spot_frame in tqdm(spots.iterchildren(), desc="parsing spots; frame"):
        for spot in spot_frame.iterchildren():
            # spot id is always an int
            spot_id = int(spot.get("ID"))
            graph.add_node(spot_id)

            # get all attributes and convert to floats
            spot_attributes = spot.attrib
            spot_attributes = {key: float(value) for key, value in spot_attributes.items() if key != "name"}

            spot_attributes["graph_key"] = spot_id
            spot_attributes["FRAME"] = int(spot_attributes["FRAME"])

            # mostly used in 2d
            if spot.text:
                spot_attributes["roi"] = [float(pt) for pt in spot.text.split(" ")]

            spots_collect.append(spot_attributes)

    # use graph key universally as an index
    spots_df = pd.DataFrame(spots_collect, index=[c["graph_key"] for c in spots_collect])
    assert np.all(spots_df.index == spots_df["ID"])

    # iterate through track elements to construct graph and assign trackid
    tracks = root.find("Model").find("AllTracks")

    for i, track in enumerate(tqdm(tracks.iterchildren(), desc="parsing edges; track"), start=1):
        track_id = i

        this_track_spots = set()

        for edge in track.iterchildren():
            edge_attributes = edge.attrib

            source_spot_id = int(edge_attributes["SPOT_SOURCE_ID"])
            target_spot_id = int(edge_attributes["SPOT_TARGET_ID"])

            this_track_spots.add(source_spot_id)
            this_track_spots.add(target_spot_id)

        this_track_spots = list(this_track_spots)

        track_spots = spots_df.loc[this_track_spots].sort_values(by=["FRAME"]).index
        for source, target in zip(track_spots[:-1], track_spots[1:]):
            source_spot_frame = int(spots_df.loc[source]["FRAME"])
            target_spot_frame = int(spots_df.loc[target]["FRAME"])

            # add edge to graph
            graph.add_edge(source, target, track_id=track_id, time=target_spot_frame - source_spot_frame)

        spots_df.loc[this_track_spots, "linear_track_id"] = track_id

    print(f"track id 0 corresponds to {np.sum(spots_df['linear_track_id'].isna())} edgeless spots")

    return spots_df, graph



if __name__ == "__main__"
