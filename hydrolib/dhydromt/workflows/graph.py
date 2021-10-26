# -*- coding: utf-8 -*-

import os
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import logging
import networkx as nx
import contextily as ctx
import random
from networkx.drawing.nx_agraph import graphviz_layout


logger = logging.getLogger(__name__)


__all__ = []


def create_graph_from_branches(
    branches: gpd.GeoDataFrame, id_col: str = "branchId"
) -> nx.Graph():
    """This function creates the graph by adding edges.
    Result graph has coordinates as nodes"""

    G = nx.DiGraph()

    for index, row in branches.iterrows():
        from_node = row.geometry.coords[0]
        to_node = row.geometry.coords[-1]

        G.add_edge(from_node, to_node, **{id_col: row[id_col]})

    return G


def update_graph_edges_attributes(
    G: nx.Graph, id_col: str = "branchId", attr_df: dict = {}
) -> nx.Graph():
    """This function updates the graph by adding new edges attributes specified in attr_df"""

    # graph df
    _graph_df = nx.to_pandas_edgelist(G).set_index(id_col)

    # check if edges id in attribute df
    if attr_df.index.name == id_col:
        graph_df = _graph_df.join(attr_df)
    elif id_col in attr_df.columns:
        graph_df = _graph_df.join(attr_df.set_index(id_col))
    else:
        raise ValueError(
            "attributes could not be updated to graph: could not perform join"
        )

    G_updated = nx.from_pandas_edgelist(
        graph_df.reset_index(),
        source="source",
        target="target",
        edge_attr=True,
        create_using=type(G),
    )

    return G_updated


def query_graph_edges_attributes(G, id_col: str = "branchId", edge_query: str = None):
    """This function queries the graph by selecting only the edges specified in edge_query"""

    if edge_query is None:
        G_query = G

    else:
        _graph_df = nx.to_pandas_edgelist(G).set_index(id_col)
        graph_df = _graph_df.query(edge_query)

        if len(graph_df) != 0:
            G_query = nx.from_pandas_edgelist(
                graph_df.reset_index(),
                source="source",
                target="target",
                edge_attr=True,
                create_using=type(G),
            )
        else:
            raise ValueError("edges_query results in nothing left")

    return G_query


def make_graphplot_for_targetnodes(
    G: nx.DiGraph,
    target_nodes: list,
    target_nodes_labeldict: dict = None,
    layout="xy",
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # layout graphviz
    if layout == "graphviz":
        # get position
        pos = graphviz_layout(G, prog="dot", args="")

        # draw network
        nx.draw(G, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False)

        if target_nodes_labeldict is not None:

            # draw labels
            nx.draw_networkx_labels(
                G, pos, target_nodes_labeldict, font_size=16, font_color="k"
            )

    # layout xy
    elif layout == "xy":
        # get position
        pos = {xy: xy for xy in G.nodes()}
        # make plot for each target node
        RG = G.reverse()

        for target in target_nodes:
            c = random_color()

            # make target upstream a graph
            target_G = G.subgraph(
                list(dict(nx.bfs_predecessors(RG, target)).keys()) + [target]
            )

            # draw graph
            nx.draw_networkx(
                target_G,
                pos,
                node_size=10,
                node_color=[c],
                width=2,
                edge_color=[c],
                with_labels=False,
                ax=ax,
            )

            # draw outlets
            nx.draw_networkx_nodes(
                target_G,
                pos,
                nodelist=[target],
                node_size=100,
                node_color="k",
                edgecolors=c,
                ax=ax,
            )

            # draw labels
            if target_nodes_labeldict is not None:
                nx.draw_networkx_labels(
                    target_G, pos, target_nodes_labeldict, font_size=16, font_color="k"
                )
    return fig, ax


def plot_graph(G: nx.DiGraph):
    """This function makes plots for two different layout"""

    # convert to undirected graph
    UG = G.to_undirected()

    # find connected components in undirected graph
    outlets = []
    for i, SG in enumerate(nx.connected_components(UG)):

        # make components a subgraph
        SG = G.subgraph(SG)

        # find outlets of the subgraph
        outlets.append([n for n in SG.nodes() if G.out_degree(n) == 0])

    outlets = sum(outlets, [])
    outlet_ids = {}

    fig1, ax1 = make_graphplot_for_targetnodes(
        G, outlets, outlet_ids, layout="graphviz"
    )
    fig2, ax2 = make_graphplot_for_targetnodes(G, outlets, outlet_ids, layout="xy")

    return (fig1, ax1), (fig2, ax2)


def random_color():
    return tuple(
        [
            random.randint(0, 255) / 255,
            random.randint(0, 255) / 255,
            random.randint(0, 255) / 255,
        ]
    )


def validate_1dnetwork_connectivity(
    branches: gpd.GeoDataFrame,
    plotit=False,
    ax=None,
    exportpath=os.getcwd(),
    logger=logging,
):
    """Function to validate the connectivity of provided branch"""

    # affirm datatype
    branches = gpd.GeoDataFrame(branches)

    # create digraph
    G = create_graph_from_branches(branches)
    pos = {xy: xy for xy in G.nodes()}

    # convert to undirected graph
    UG = G.to_undirected()

    # find connected components in undirected graph
    outlets = []
    for i, SG in enumerate(nx.connected_components(UG)):

        # make components a subgraph
        SG = G.subgraph(SG)

        # find outlets of the subgraph
        outlets.append([n for n in SG.nodes() if G.out_degree(n) == 0])

    outlets = sum(outlets, [])
    outlet_ids = {
        p: [li for li, l in branches.geometry.iteritems() if l.intersects(Point(p))]
        for p in outlets
    }

    # report
    if i == 0:
        logger.info(
            "Validation results: the 1D network are fully connected.  Supress plotit function."
        )
    else:
        logger.info(
            f"Validation results: the 1D network are disconnected have {i+1} connected components"
        )

    if plotit:
        ax = make_graphplot_for_targetnodes(G, outlets, outlet_ids, layout="graphviz")
        ax.set_title(
            "Connectivity of the 1d network, with outlets"
            + "(connectivity outlets, not neccessarily network outlets due to bi-directional flow, please check these)",
            wrap=True,
        )
        plt.savefig(exportpath.joinpath("validate_1dnetwork_connectivity"))

    return None


def validate_1dnetwork_flowpath(
    branches: gpd.GeoDataFrame,
    branchType_col="branchType",
    plotit=False,
    ax=None,
    exportpath=os.getcwd(),
    logger=logging,
):
    """function to validate flowpath (flowpath to outlet) for provided branch"""

    # affirm datatype
    branches = gpd.GeoDataFrame(branches)

    # create digraph
    G = gpd_to_digraph(branches)
    pos = {xy: xy for xy in G.nodes()}

    # create separate graphs for pipes and branches
    pipes = branches.query(f"{branchType_col} == 'Pipe'")
    channels = branches.query(f"{branchType_col} == 'Channel'")

    # validate 1d network based on pipes -> channel logic
    if len(pipes) > 0:
        # create graph
        PG = gpd_to_digraph(pipes)
        # pipes outlets
        pipes_outlets = [n for n in PG.nodes() if G.out_degree(n) == 0]
        pipes_outlet_ids = {
            p: [li for li, l in pipes.geometry.iteritems() if l.intersects(Point(p))]
            for p in pipes_outlets
        }
        logger.info(
            f"Validation result: the 1d network has {len(pipes_outlets)} pipe outlets."
        )

    if len(channels) > 0:
        # create graph
        CG = gpd_to_digraph(channels)
        # pipes outlets
        channels_outlets = [n for n in CG.nodes() if G.out_degree(n) == 0]
        channels_outlet_ids = {
            p: [li for li, l in channels.geometry.iteritems() if l.intersects(Point(p))]
            for p in channels_outlets
        }
        logger.info(
            f"Validation result: the 1d network has {len(channels_outlets)} channel outlets."
        )

    if (len(channels) > 0) and (len(pipes) > 0):
        isolated_outlets = [
            p
            for p in pipes_outlets
            if not any(Point(p).intersects(l) for _, l in channels.geometry.iteritems())
        ]
        isolated_outlet_ids = {}
        for p in isolated_outlets:
            isolated_outlet_id = [
                li for li, l in pipes.geometry.iteritems() if l.intersects(Point(p))
            ]
            isolated_outlet_ids[p] = isolated_outlet_id
            logger.warning(
                f"Validation result: downstream of {isolated_outlet_id} are not located on channels. Please double check. "
            )

    # plot
    if plotit:
        ax = make_graphplot_for_targetnodes(
            G,
            target_nodes={**isolated_outlet_ids, **channels_outlet_ids}.keys(),
            target_nodes_labeldict={**isolated_outlet_ids, **channels_outlet_ids},
        )
        ctx.add_basemap(
            ax=ax, url=ctx.providers.OpenStreetMap.Mapnik, crs=branches.crs.to_epsg()
        )
        ax.set_title(
            "Flow path of the 1d network, with outlets"
            + "(flowpath outlets, not neccessarily network outlets due to bi-directional flow , please check these)",
            wrap=True,
        )
        plt.savefig(exportpath.joinpath("validate_1dnetwork_flowpath"))

    return None
