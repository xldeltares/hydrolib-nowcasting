"""Implement plugin model class"""
from __future__ import annotations

import glob
from os.path import join, basename, isfile
from pathlib import Path
import logging
import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
from rasterio.warp import transform_bounds
import pyproj
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import xarray as xr
from collections import Counter

import hydromt
from hydromt.models.model_api import Model
from hydromt import gis_utils, io
from hydromt import raster

from .workflows import *
from . import DATADIR

__all__ = ["NowcastingModel"]
logger = logging.getLogger(__name__)


class NowcastingModel(Model):
    """General and basic API for models in HydroMT"""

    # FIXME
    _NAME = "nowcasting"
    _CONF = ""
    _DATADIR = DATADIR
    _GEOMS = {}  # FIXME Mapping from hydromt names to model specific names
    _MAPS = {}  # FIXME Mapping from hydromt names to model specific names
    _FOLDERS = ["graph", "staticgeoms"]

    def __init__(
        self,
        root=None,
        mode="w",
        config_fn=None,  # hydromt config contain glob section, anything needed can be added here as args
        data_libs=None,
        # yml # TODO: how to choose global mapping files (.csv) and project specific mapping files (.csv)
        logger=logger,
        deltares_data=False,  # data from pdrive
    ):

        if not isinstance(root, (str, Path)):
            raise ValueError("The 'root' parameter should be a of str or Path.")

        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            deltares_data=deltares_data,
            logger=logger,
        )

        # model specific
        self._meta = {}
        self._graphmodel = None
        self._subgraphmodels = {}

    def setup_basemaps(
        self,
        region,
        report: str = None,
        **kwargs,
    ):
        """Define the model region.
        and setup the base graph - with geometry and id

        Adds model layer:
        * **region** geom: model region

        Parameters
        ----------
        region: dict
            Dictionary describing region of interest, e.g. {'bbox': [xmin, ymin, xmax, ymax]}.
            See :py:meth:`~hydromt.workflows.parse_region()` for all options.
        **kwargs
            Keyword arguments passed to _setup_graph(**kwargs)
        """

        kind, region = hydromt.workflows.parse_region(region, logger=self.logger)
        if kind == "bbox":
            geom = gpd.GeoDataFrame(geometry=[box(*region["bbox"])], crs=4326)
        elif kind == "geom":
            geom = region["geom"]
        else:
            raise ValueError(
                f"Unknown region kind {kind} for DFlowFM, expected one of ['bbox', 'geom']."
            )

        # Set the model region geometry (to be accessed through the shortcut self.region).
        self.set_staticgeoms(geom, "region")
        # FIXME: how to deprecate WARNING:root:No staticmaps defined

        if report:
            plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.title(report, wrap=True, loc="left")

    def setup_graph(
        self,
        graph_class: str = None,
        report: str = None,
        **kwargs,
    ):

        """Setup the graph - with geometry and id

        Parameters
        ----------
        graph_class: str
            Class in networkx for creating the graph.
        edges_fn: str
            Name of the edges shp file specified in data yml.
        edges_id_col: str
            Columns name used in the shp file as the id of edges.
        nodes_fn: str Not Implemented
            Name of the nodes shp file specified in data yml.
        nodes_id_col: str Not Implemented
            Columns name used in the shp file as the id of nodes.

        Notes
        ----------
        This function can only be called once
        """

        # Set the graph model class
        if self._graphmodel:
            logger.debug(
                "graph model exists. "
            )
        else:
            logger.debug("Creating graph model. ")
            assert graph_class in [
            "Graph",
            "DiGraph",
            "MultiGraph",
            "MultiDiGraph",
        ], "graph_class not recognised. Please assign grah_class using one of the following [Graph, DiGraph, MultiGraph, MultiDiGraph]"
            self._graphmodel = eval(f"nx.{graph_class}()")

        if report:
            G = self._graphmodel
            graph.plot_xy(G, plot_outfall=True)
            plt.title(report, wrap=True, loc="left")

    def setup_edges(
        self,
        edges_fn: str,
        id_col: str | None = None,
        attribute_cols: list| None = None,
        snap_offset: float = 10e-6,
        **kwargs,
    ):
        """This component setup edges locations and attributes to the graph model

        See Also
        --------
        _setup_edges
        """

        if edges_fn is None:
            raise ValueError(f"Expected edges_fn.")
        else:
            edges = self._get_geodataframe(edges_fn)

        if id_col is None:
            raise ValueError('Expected id_col.')

        if attribute_cols is None:
            attribute_cols = []

        if edges is not None:

            if snap_offset is not None:
                # preprocessing geometry using snap_offset
                edges = helper.reduce_gdf_precision(edges, rounding_precision=1e-8)
                edges = snap_branch_ends(edges, offset=snap_offset)
                logger.debug(
                    f"Performing snapping at edges ends.")

            if id_col is not None:
                self._setup_edges(
                    edges = edges,
                    id_col=id_col,
                    attribute_cols=attribute_cols,
                    use_location=True
                )

            self.logger.info(f"Adding staticgeoms {edges_fn}.")
            self.set_staticgeoms(edges, edges_fn)

        else:
            raise ValueError("Failed due to zero geometry.")

    def setup_edge_attributes(
            self,
            edges_fn: str,
            id_col: str | None = None,
            attribute_cols: list| None = None,
            snap_offset: float | None = None,
            **kwargs
    ):
        """This component update edges attributes to the graph model

        See Also
        --------
        _setup_edges

        **kwargs
            Additional keywords have no effect but might be accepted for
            compatibility with numpy.
        """

        if edges_fn is None:
            raise ValueError(f"Expected edges_fn.")
        else:
            edges = self._get_geodataframe(edges_fn)

        if not any(v is not None for v in {id_col, snap_offset}):
            raise ValueError('Expected either id_col or snap_offset.')

        if attribute_cols is None:
            attribute_cols = []

        if edges is not None:

            if id_col is None:
                # preprocessing geometry using snap_offset
                edges = graph.find_edge_ids_by_snapping(self._graphmodel, edges, snap_offset)
                id_col = "_id"
                logger.debug(
                    f"Performing snapping to edges.")

            if id_col is not None:
                # join edges attributes by matching id_col
                self._setup_edges(
                    edges=edges,
                    id_col=id_col,
                    attribute_cols=attribute_cols,
                    use_location=False,
                )

            self.logger.info(f"Adding staticgeoms {edges_fn}.")
            self.set_staticgeoms(edges, edges_fn)

        else:
            raise ValueError("Failed due to zero geometry.")


    def setup_nodes(
        self,
        nodes_fn: str,
        id_col: str,
        attribute_cols: list = None,
        snap_offset: float = 10e-6,
        **kwargs,
    ):
        """This component setup nodes locations and attributes to the graph model

        See Also
        --------
        _setup_nodes
        """

        if nodes_fn is None:
            raise ValueError(f"Expected nodes_fn.")
        else:
            nodes = self._get_geodataframe(nodes_fn)

        if id_col is None:
            raise ValueError('Expected id_col.')

        if attribute_cols is None:
            attribute_cols = []

        if snap_offset is None:
            # for nodes, snap_offset is always applied
            snap_offset: float = 10e-6

        if nodes is not None:

            self._setup_nodes(
                nodes=nodes,
                id_col=id_col,
                attribute_cols=attribute_cols,
                use_location=True,
                snap_offset=snap_offset,
            )

            self.logger.info(f"Adding staticgeoms {nodes_fn}.")
            self.set_staticgeoms(nodes, nodes_fn)

        else:
            raise ValueError("Failed due to zero geometry.")



    def setup_node_attributes(
        self, nodes_fn: str, id_col: str | None = None, attribute_cols: list | None = None, snap_offset:float | None = None, **kwargs
    ):
        """This component update nodes attributes to the graph model

        See Also
        --------
        _setup_nodes
        """


        if nodes_fn is None:
            raise ValueError(f"Expected nodes_fn.")
        else:
            nodes = self._get_geodataframe(nodes_fn)

        if not any(v is not None for v in {id_col, snap_offset}):
            raise ValueError('Expected either id_col or snap_offset.')

        if attribute_cols is None:
            attribute_cols = []

        if id_col is None:
            # for nodes, snap_offset is always applied
            snap_offset: float = 10e-6

        if nodes is not None:

            if id_col is None:
                nodes = graph.find_node_ids_by_snapping(self._graphmodel, nodes, snap_offset)
                id_col = "_id"
                logger.debug(
                    f"Performing snapping to nodes.")

            if id_col is not None:
                self._setup_nodes(
                    nodes=nodes,
                    id_col=id_col,
                    attribute_cols=attribute_cols,
                    use_location=False,
                    snap_offset=snap_offset
                )

            self.logger.info(f"Adding staticgeoms {nodes_fn}.")
            self.set_staticgeoms(nodes, nodes_fn)

        else:
            raise ValueError("Failed due to zero geometry.")


    def _setup_edges(
        self,
        edges: gpd.GeoDataFrame,
        id_col: str,
        attribute_cols: list = [],
        use_location: bool = False,
        **kwargs,
    ):
        """This component add edges or edges attributes to the graph model

        * **edges** geom: vector

        Parameters
        ----------
        edges_fn : str
            Name of data source for edges, see data/data_sources.yml.
            * Required variables: [id_col]
            * Optional variables: [attribute_cols]
        id_col : str
            Column that is converted into edge attributes using key: id
        attribute_cols : str
            Columns that are converted into edge attributes using the same key
        snap_offset: float
            A float to control how the edges ends will be snapped

        Arguments
        ----------
        use_location : bool
            If True, the edges will be added; if False, only edges attributes are added.
            Latter will be mapped to id.

        """

        assert set([id_col] + attribute_cols).issubset(
            edges.columns
        ), f"id and/or attribute cols {[id_col] + attribute_cols} do not exist in {edges.columns}"

        if use_location is True:
            self.logger.info(f"Adding new edges.")
            attribute_cols = ["geometry"] + attribute_cols
            self._graphmodel = graph.add_edges_with_id(
                self._graphmodel, edges=edges, id_col=id_col
            )

        if len(attribute_cols) > 0:
            self.logger.info(f"Adding new edges attributes")
            self._graphmodel = graph.update_edges_attributes(
                self._graphmodel, edges=edges[[id_col] + attribute_cols], id_col=id_col
            )


    def _setup_nodes(
        self,
        nodes: gpd.GeoDataFrame,
        id_col: str | None = None,
        attribute_cols: list = [],
        use_location: bool = False,
        snap_offset: float = 10e-6,
        **kwargs,
    ):
        """This component add nodes or nodes attributes to the graph model

        * **nodes** geom: vector

        Parameters
        ----------
        nodes_fn : str
            Name of data source for edges, see data/data_sources.yml.
            * Required variables: [id_col]
            * Optional variables: [attribute_cols]
        id_col : str
            Column that is converted into node attributes using key: id
        attribute_cols : str
            Columns that are converted into node attributes using the same key
        snap_offset: float
            A float to control how the nodes will be snapped to graph

        Arguments
        ----------
        use_location : bool
            If True, the nodes will be added; if False, only nodes attributes are added.
            Latter will be mapped to id.

        """

        assert set([id_col] + attribute_cols).issubset(
            nodes.columns
        ), f"id and/or attribute cols {[id_col] + attribute_cols} do not exist in {nodes.columns}"

        if use_location is True:
            self.logger.debug(f"Adding new nodes.")
            attribute_cols = ["geometry"] + attribute_cols
            self._graphmodel = graph.add_nodes_with_id(
                self._graphmodel, nodes=nodes, id_col=id_col, snap_offset = snap_offset,
            )

        if len(attribute_cols) > 0:
            self.logger.debug(f"updating nodes attributes")
            self._graphmodel = graph.update_nodes_attributes(
                self._graphmodel, nodes=nodes[[id_col] + attribute_cols], id_col=id_col
            )


    def _get_geodataframe(self, path_or_key: str) -> gpd.GeoDataFrame:
        """Function to get geodataframe.

        This function is a wrapper around :py:meth:`~hydromt.data_adapter.DataCatalog.get_geodataset`,
        added support for updating columns based on funcs in yml

        Arguments
        ---------
        path_or_key: str
            Data catalog key. If a path to a vector file is provided it will be added
            to the data_catalog with its based on the file basename without extension.

        Returns
        -------
        gdf: geopandas.GeoDataFrame
            GeoDataFrame

        """

        # read data
        df = self.data_catalog.get_geodataframe(
            path_or_key,
            geom=None,
            # FIXME use geom region gives error "geopandas Invalid projection: : (Internal Proj Error: proj_create: unrecognized format / unknown name)"
        )

        # update columns
        funcs = None
        try:
            funcs = self.data_catalog.to_dict(path_or_key)[path_or_key]["kwargs"][
                "funcs"
            ]
        except:
            pass

        if funcs is not None:
            self.logger.debug(f"updating {path_or_key} GeoDataFrame vector data")

            for k, v in funcs.items():
                try:
                    # eval funcs for columns that exist
                    if "geometry" in v:
                        # ensure type of geoseries - might be a bug in pandas / geopandas
                        _ = type(df["geometry"])
                    df[k] = df.eval(v)
                    self.logger.debug(f"update column {k} based on {v}")
                except:
                    try:
                        # assign new columns
                        df[k] = v
                        self.logger.debug(f"update column {k} based on {v}")
                    except Exception as e:
                        self.logger.debug(
                            f"can not update column {k} based on {v}: {e}"
                        )

        # post processing
        df = df.drop_duplicates()
        self.logger.debug(f"drop duplicates in dataframe")

        self.logger.info(f"{len(df)} features read from {path_or_key}")

        return df




    def setup_subgraph(
        self,
        G:nx.Graph = None,
        subgraph_fn: str = None,
        edge_query: str = None,
        edge_target: str = None,
        node_query: str = None,
        node_target: str = None,
        algorithm: str = None,
        target_query: str = None,
        weight: str = None,
        report: str = None,
        **kwargs,
    ):
        """This component prepares the subgraph from a graph

        In progress

        Parameters
        ----------
        subgraph_fn : str
            Name of the subgraph instance.
            If None, the function will update the self._graphmodel
            if String and new, the function will create the instance in self._subgraphmodels
            if String and old, the function will update the instance in self._subgraphmodels
        Arguments
        ----------
        edge_query : str
            Conditions to query the subgraph from graph.
            Applies on existing attributes of edges, so make sure to add the attributes first.
        edge_target : str
            Conditions to request from subgraph the connected components that has the edge_target id
            Applies on existing id of edges.
        node_query : str
            Conditions to query the subgraph from graph.
            Applies on existing attributes of nodes, so make sure to add the attributes first.
        node_target : str
            Conditions to request from subgraph the connected components that has the node_target id
            Applies on existing id of nodes.
        algorithm : str = None
            Algorithms to apply addtional processing on subgraph.
            Supported options: 'patch', 'mst', 'dag'.
            If None, do not apply any method - disconnected subgraph
            If "patch", patch paths from queried nodes in SG to an signed outlet
            if "mst", steiner tree approximation for nodes in SG
            if "dag" dijkstra shortest length resulted Directed Acyclic Graphs (DAG)
            # FIXME: add weight as argument; combine setup_dag function
        weight : str = None

        """

        # check input parameter and arguments
        if subgraph_fn in self._subgraphmodels:
            self.logger.warning(
                f"subgraph instance {subgraph_fn} already exist, apply setup_subgraph on subgraph_fn and overwrite."
            )
            G = self._subgraphmodels[subgraph_fn].copy()
        elif subgraph_fn:
            self.logger.debug(
                f"subgraph instance {subgraph_fn} will be created from graph"
            )
            G = self._graphmodel.copy()
        else:
            if G is None:
                self.logger.debug(f"will apply on graph itself.")
                G = self._graphmodel.copy()
            else:
                self.logger.debug(f"will apply on given graph.")
                pass

        SG = G.copy()

        # check queries
        if all (v is not None for v in [edge_query, node_query]):
            raise ValueError("could not apply on both edges and nodes")

        # query edges/nodes
        if edge_query is None:
            SG = SG
        else:
            self.logger.debug(f"query sedges: {edge_query}")
            SG = graph.query_graph_edges_attributes(
                SG,
                id_col="id",
                edge_query=edge_query,
            )
            self.logger.debug(f"{len(SG.edges())}/{len(G.edges())} edges are selected")

        # select the subgraph component based on edge id
        if edge_target is None:
            pass
        else:
            for c in nx.connected_components(SG.to_undirected()):
                if any(e[2] == edge_target for e in SG.subgraph(c).edges(data = 'id')):
                    SG = SG.subgraph(c).copy()
                    self.logger.debug(f"connected components from subgraph containing edge {edge_target} is selected")

        # query nodes
        if node_query is None:
            SG = SG
        else:
            self.logger.debug(f"query nodes: {node_query}")
            SG = graph.query_graph_nodes_attributes(
                SG,
                id_col="id",
                node_query=node_query,
            )
            self.logger.debug(f"{len(SG.nodes())}/{len(G.nodes())} nodes are selected")

        # select the subgraph component based on node id
        if node_target is None:
            pass
        else:
            for c in nx.connected_components(SG.to_undirected()):
                if any(n[-1] == node_target for n in SG.subgraph(c).nodes(data = 'id')):
                    SG = SG.subgraph(c).copy()
                    self.logger.debug(f"connected components from subgraph containing node {node_target} is selected")


        if algorithm is not None:
            if algorithm not in ("patch", "mst", "dag"):
                raise ValueError(f"algorithm not supported: {algorithm}")

        # check targets
        targets = self._find_target_nodes(G, target_query, target_query)
        self.logger.debug(f"{len(targets)} targets are selected")

        if weight is not None:
            assert (
                weight in list(SG.edges(data=True))[0][2].keys()
            ), f"edge weight {weight} does not exist!"

        # start making subgraph
        G = self._graphmodel.copy()
        UG = G.to_undirected()
        SG = SG.copy()

        # apply additional algorithm
        # TODO: how to query a subset of edges yet retain the connectivity of the network?
        if algorithm is None:
            #  do not apply any method - disconnected subgraph
            pass

        elif algorithm == "patch":
            # patch paths from queried nodes in SG to an signed outlet
            self.logger.debug(f"patch paths from SG nodes to targets")

            SG_nodes = list(SG.nodes)
            for outlet in targets:
                for n in SG_nodes:
                    if nx.has_path(SG, n, outlet):
                        pass
                    elif nx.has_path(G, n, outlet):  # directional path exist
                        nx.add_path(SG, nx.dijkstra_path(G, n, outlet))
                    if nx.has_path(UG, n, outlet):  # unidirectional path exist
                        nx.add_path(SG, nx.dijkstra_path(UG, n, outlet))
                    else:
                        print(f"No path to {outlet}")

        elif algorithm == "mst":
            # minimum spanning tree
            self.logger.debug(
                f"steine rtree approximation for SG nodes using weight {weight}"
            )
            targets = None
            SG_nodes = list(SG.nodes)
            SG = nx.algorithms.approximation.steinertree.steiner_tree(
                UG, SG_nodes, weight=weight
            )

        elif algorithm == "dag":
            # dag
            self.logger.debug(f"creating dag from SG nodes to targets")
            SG = self.setup_dag(SG, targets=targets, weight=weight, **kwargs)

        # assign SG
        if subgraph_fn:
            self._subgraphmodels[subgraph_fn] = SG
            nx.set_edge_attributes(
                self._subgraphmodels[subgraph_fn], values=True, name=subgraph_fn
            )
            # add subgraph_fn as attribute in the G
            nx.set_edge_attributes(
                self._graphmodel,
                {
                    e: {subgraph_fn: True}
                    for e in self._subgraphmodels[subgraph_fn].edges()
                },
            )
        else:
            self._graphmodel = SG

        # report
        if report:

            # plot graphviz
            graph.make_graphplot_for_targetnodes(SG, targets, layout="graphviz")
            plt.title(report, wrap=True, loc="left")

            # plot xy
            plt.figure(figsize=(8, 8))
            plt.title(report, wrap=True, loc="left")
            plt.axis("off")
            pos = {xy: xy for xy in G.nodes()}

            # base
            nx.draw(
                G,
                pos=pos,
                node_size=0,
                with_labels=False,
                arrows=False,
                node_color="gray",
                edge_color="silver",
                width=0.5,
            )
            # SG nodes and edges
            if algorithm == "dag":
                nx.draw_networkx_nodes(
                    G,
                    pos=pos,
                    nodelist=targets,
                    node_size=200,
                    node_shape="*",
                    node_color="r",
                )
                edge_width = [d[2] / 100 for d in SG.edges(data="nnodes")]
                nx.draw_networkx_edges(
                    SG,
                    pos=pos,
                    edgelist=SG.edges(),
                    arrows=False,
                    width=[
                        float(i) / (max(edge_width) + 0.1) * 20 + 0.5
                        for i in edge_width
                    ],
                )
            else:
                edge_width = [0 / 100 for d in SG.edges()]
                nx.draw_networkx_edges(
                    SG,
                    pos=pos,
                    edgelist=SG.edges(),
                    arrows=False,
                    width=[
                        float(i) / (max(edge_width) + 0.1) * 20 + 0.5
                        for i in edge_width
                    ],
                )
        return SG

    def _find_target_nodes(self, G:nx.DiGraph, node_query=None, edges_query=None):
        """helper to find the target nodes"""

        targets = None

        if all(v is None for v in [node_query,edges_query]):
            targets = [n for n in G.nodes if G.out_degree[n] == 0]

        if isinstance(edges_query, str):
            try:
                _target_G = graph.query_graph_edges_attributes(
                    G,
                    edge_query=edges_query,
                )
                targets = [v for u, v in _target_G.edges()]
                self.logger.debug("Find targets in edges")
            except Exception as e:
                self.logger.debug(e)
                pass

        if isinstance(node_query, str):
            try:
                _target_G = graph.query_graph_nodes_attributes(
                    G,
                    node_query=node_query,
                )
                targets = [n for n in _target_G.nodes()]
                self.logger.debug("Find targets in nodes")
            except Exception as e:
                self.logger.debug(e)
                pass

        if targets is None:
            raise ValueError("could not find targets.")

        return targets


    def setup_dag(
        self,
        G: nx.Graph = None,
        targets = None,
        target_query:str = None,
        weight: str = None,
        loads: list = [],
        report: str = None,
        use_super_target: bool = True,
        **kwargs,
    ):
        """This component prepares subgraph as Directed Acyclic Graphs (dag) using shortest path
        step 1: add a supernode to subgraph (representing graph - subgraph)
        step 2: use shortest path to prepare dag edges (source: node in subgraph; target: super node)

        in progress

        Parameters
        ----------
        G : nx.Graph
        targets : None or String or list, optional (default = None)
            DAG super targets.
            If None, a target node will be any node with out_degree == 0.
            If a string, use this to query part of graph as a super target node.
            If a list, use targets as a list of target nodes.
        weight : None or string, optional (default = None)
            Weight used for shortest path.
            If None, every edge has weight/distance/cost 1.
            If a string, use this edge attribute as the edge weight.
            Any edge attribute not present defaults to 1.
        loads : None or list of strings, optional (default - None)
            Load used from edges/nodes attributes.
            If None, every node/edge has a load equal to the total number of nodes upstream (nnodes), and number of edges upstream (nedges).
            If a list, use the attributes in the list as load.
            The attribute can be either from edges or from nodes.
            Any attributes that are not present defaults to 0.
        algorithm : string, optional (default = 'dijkstra')
            The algorithm to use to compute the path.
            Supported options: 'dijkstra', 'bellman-ford'.
            Other inputs produce a ValueError.
            If `weight` is None, unweighted graph methods are used, and this
            suggestion is ignored.

        Arguments
        ----------
        use_super_target : bool
            whether to add a super target at the ends of all targets.
            True if the weight of DAG exist for all edges.
            False if the weight of DAG also need to consider attribute specified for targets.

        """
        # convert Digraph to Graph
        if G is None:
            G = self._graphmodel.copy()
            self.logger.debug("Apply dag on graphmodel.")
        if isinstance(G, nx.DiGraph):
                _G = G.copy()
                G = G.to_undirected()

        # check targets of the dag
        if isinstance(target_query, str):
            targets = self._find_target_nodes(_G, target_query, target_query)
        self.logger.debug(f"{len(targets)} targets are selected")

        # check if graph is fully connected
        if len([_ for _ in nx.connected_components(G)]) > 1:
            # try adding super nodes
            G.add_edges_from([(n, -1) for n in targets])
            self.logger.debug(f"connecting targets to supernode")
            if len([_ for _ in nx.connected_components(G)]) > 1:
                raise TypeError("Cannot apply dag on disconnected graph.")

        # check algorithm of the dag
        algorithm = "dijkstra"
        if algorithm not in ("dijkstra", "bellman-ford"):
            raise ValueError(f"algorithm not supported: {algorithm}")
        self.logger.debug(f"Performing {algorithm}")

        # started making dag
        DAG = nx.DiGraph()

        # 1. add path
        # FIXME: if don't do this step: networkx.exception.NetworkXNoPath: No path to **.
        if use_super_target:
            # add super target
            _t = [-1]
            if weight == "streetlev":
                # FIXME temporary
                # mirror weights at super edges
                for w in ["geom_length", "streetlev"]:
                    G.add_weighted_edges_from(
                        [
                            (
                                t,
                                _t[0],
                                max(
                                    [
                                        k[1][w] if w in k[1] else 0
                                        for k in G[t].items()
                                    ]
                                ),
                            )
                            for t in list(set(G.nodes) & set(targets))
                        ],
                        weight=w,
                    )
            else:
                G.add_edges_from(
                    [(t, _t[0]) for t in list(set(G.nodes) & set(targets))]
                )
        else:
            _t = targets
        for t in _t:
            for n in G.nodes:
                if n not in DAG:
                    if weight == "streetlev":
                        # FIXME temporary
                        smax = max([k[1]["streetlev"] for k in G[t].items()])
                        path = nx.shortest_path(
                            G,
                            n,
                            t,
                            weight=lambda u, v, e: 10 - smax / e["geom_length"],
                            method=algorithm,
                        )
                    else:
                        path = nx.shortest_path(
                            G, n, t, weight=weight, method=algorithm
                        )
                    nx.add_path(DAG, path)
                    DAG.remove_node(t)

        # 2. add back weights
        for u, v, new_d in DAG.edges(data=True):
            # get the old data from X
            old_d = G[u].get(v)
            non_shared = set(old_d) - set(new_d)
            if non_shared:
                # add old weights to new weights if not in new data
                new_d.update(dict((key, old_d[key]) for key in non_shared))

        # 3. add auxiliary calculations
        _ = DAG.reverse()
        nodes_attributes = [k for n in G.nodes for k in G.nodes[n].keys()]
        edges_attribute = [k for e in G.edges for k in G.edges[e].keys()]
        nodes_loads = [l for l in loads if l in nodes_attributes]
        edegs_loads = [l for l in loads if l in edges_attribute]

        for s, t in _.edges():
            # fixed loads
            upstream_nodes = list(nx.dfs_postorder_nodes(_, t)) # exclusive
            upstream_edges = list(G.subgraph(upstream_nodes).edges())
            DAG[t][s].update({"upstream_nodes": upstream_nodes, "upstream_edges": upstream_edges})
            DAG.nodes[s].update({"upstream_nodes": upstream_nodes, "upstream_edges": upstream_edges})
            nnodes = len(upstream_nodes)
            nedges = len(upstream_edges)
            DAG[t][s].update({"nnodes": nnodes, "nedges": nedges})
            DAG.nodes[s].update({"nnodes": nnodes, "nedges": nedges})
            # customised nodes
            sumload_from_nodes = 0
            sumload_from_edges = 0
            for l in loads:
                if l in nodes_loads:
                    sumload_from_nodes = np.nansum([G.nodes[n][l] for n in upstream_nodes])
                elif l in edegs_loads:
                    sumload_from_edges = np.nansum([G[e[0]][e[1]][l] for e in upstream_edges])
                else:
                    raise KeyError(f"Load {l} does exist in nodes/edges attributes.")
                sumload = sumload_from_nodes + sumload_from_edges
                DAG[t][s].update({l: sumload})
                DAG.nodes[s].update({l: sumload})

        # validate DAG
        if nx.is_directed_acyclic_graph(DAG):
            self.logger.debug("dag is directed acyclic graph")
        else:
            self.logger.error("dag is NOT directed acyclic graph")

        # visualise DAG
        if report:

            # plot graphviz
            graph.make_graphplot_for_targetnodes(DAG, targets, layout="graphviz")
            plt.title(report, wrap=True, loc="left")

            # plot xy
            plt.figure(figsize=(8, 8))
            plt.title(report, wrap=True, loc="left")
            plt.axis("off")
            pos = {xy: xy for xy in G.nodes()}

            # base
            nx.draw(
                DAG,
                pos=pos,
                node_size=0,
                with_labels=False,
                arrows=False,
                node_color="gray",
                edge_color="silver",
                width=0.5,
            )
            nx.draw_networkx_nodes(
                DAG,
                pos=pos,
                nodelist=targets,
                node_size=200,
                node_shape="*",
                node_color="r",
            )
            edge_width = [d[2] / 100 for d in DAG.edges(data="nnodes")]
            nx.draw_networkx_edges(
                DAG,
                pos=pos,
                edgelist=DAG.edges(),
                arrows=False,
                width=[
                    float(i) / (max(edge_width) + 0.1) * 20 + 0.5
                    for i in edge_width
                ],
            )

        self._graphmodel = DAG
        return DAG


    def setup_partition(
        self,
        subgraph_fn: str = None,
        algorithm: str = "simple",
        report: str = None,
        contracted:bool = False,
        **kwargs,
    ):
        """This component prepares the partition based on the connectivity of graph

        Parameters
        ----------
        subgraph_fn : str
            Name of the subgraph instance.
            If None, self._graphmodel will be used for partition; the function will update the self._graphmodel
            if String and new, self._graphmodel will be used for partition; the function will create the instance in self._subgraphmodels
            if String and old, self._subgraphmodels[subgraph_fn] will be used for partition; the function will update the instance in self._subgraphmodels[subgraph_fn]
        algorithm : str
            Algorithm to derive partitions from subgraph. Options: ['simple', 'louvain' ]
            testing methods:
            "simple" : based on connected components, every connected components will be considered as a partition (undirected)
            "flowpath" : based on direction of the edges, the graph is devided into a few partitions, each of which represents a target node with all of its sources.
            "louvain": based on louvain algorithm (work in progress)  (undirected). "It measures the relative density of edges inside communities with respect to edges outside communities. Optimizing this value theoretically results in the best possible grouping of the nodes of a given network."(from wikipedia)
        contracted : bool
            Specify whether to build contracted graph from the parititons. So a new contracted graph is created, with a node representing a part of the graph; edges represernting the connectivity between the parts
            If True, each partition will be contracted to a node
            If False, no contraction is performed
        """

        # get graph model
        G = self._graphmodel.copy()
        G_targets = [n for n in G.nodes if G.out_degree[n] == 0]

        # get subgraph if applicable
        if subgraph_fn in self._subgraphmodels:
            self.logger.warning(
                f"subgraph instance {subgraph_fn} will be used for partition."
            )
            SG = self._subgraphmodels[subgraph_fn].copy()
        elif subgraph_fn is not None:
            self.logger.warning(
                f"graph will be used for partition."
            )
            SG = self._graphmodel.copy()
        else:
            self.logger.debug(f"graph will be used for partition.")
            SG = self._graphmodel.copy()
        SG_targets = [n for n in SG.nodes if SG.out_degree[n] == 0]


        # start partition
        partition = {n: -1 for n in G.nodes}
        partition_edges = {e: -1 for e in G.edges}


        if algorithm == "simple":  # based on connected subgraphs
            UG = SG.to_undirected() # convert SG to UG for further partition
            for i, ig in enumerate(nx.connected_components(UG)):
                ig = UG.subgraph(ig)
                partition.update({n: i for n in ig.nodes})
            partition_edges.update({(s, t): partition[s] for s, t in partition_edges if partition[s] == partition[t]})
            logger.info(f"algorithm {algorithm} is applied. Note that different partitions are disconnected.")
        elif algorithm == 'flowpath':
            assert isinstance(SG, nx.DiGraph), f"algorithm {algorithm} can only be applied on directional graph"
            SG = graph.sort_direction(SG)
            endnodes = [n[0] for n in SG.nodes(data = 'endnodes') if n[-1] is not None]
            partition.update({nn:i for i,n in enumerate(endnodes) for nn in graph.find_predecessors(SG, n)})
            partition_edges.update({(s, t): partition[s] for s, t in partition_edges if partition[s] == partition[t]})
            logger.info(f"algorithm {algorithm} is applied. Note that flowpath might be duplicated.")
        elif algorithm == "louvain":  # based on louvain algorithm
            UG = SG.to_undirected()  # convert SG to UG for further partition
            partition.update(graph.louvain_partition(UG))
            partition_edges.update({(s, t): partition[s] for s, t in partition_edges if partition[s] == partition[t]})
        else:
            raise ValueError(
                f"{algorithm} is not recognised. allowed algorithms: simple, louvain"
            )

        n_partitions = max(partition.values())
        self.logger.debug(
            f"{n_partitions} partitions is derived from subgraph using {algorithm} algorithm"
        )

        # update partition to graph
        nx.set_node_attributes(SG, partition, 'part')
        nx.set_edge_attributes(SG, partition_edges, 'part')
        if subgraph_fn in self._subgraphmodels:
            self.logger.warning(
                f"subgraph instance {subgraph_fn} will be updated with partition information (part)."
            )
            self._subgraphmodels[subgraph_fn] = SG
        elif subgraph_fn is not None:
            self.logger.warning(
                f"subgraph instance {subgraph_fn} will be created with partition information (part)."
            )
            self._subgraphmodels[subgraph_fn] = SG
        else:
            self.logger.warning(
                f"graph will be updated with partition information (part)."
            )
            self._graphmodel = SG

        # contracted graph
        # induced graph from the partitions - faster but a bit confusing results
        # ind = community.induced_graph(partition, G)
        # ind.remove_edges_from(nx.selfloop_edges(ind))
        # induced by contracting - slower but neater
        if contracted == True:
            ind = self._graphmodel.copy()
            nx.set_node_attributes(ind, {n: {"ind_size": 1} for n in ind.nodes})
            for part in np.unique(list(partition.values())):
                part_nodes = [n for n in partition if partition[n] == part]
                if part == -1:
                    # do not contract
                    pass
                else:
                    for to_node in [n for n in part_nodes if n in SG_targets]:
                        ind, targets = graph.contract_graph_nodes(
                            ind, part_nodes, to_node
                        )
                        ind.nodes[to_node]["ind_size"] = len(part_nodes)

        # visualise
        if report:

            # Cartesian coordinates centroid
            pos_G = {xy: xy for xy in G.nodes()}
            pos_SG = {xy: xy for xy in SG.nodes()}

            # partitions
            plt.figure(figsize=(8, 8))
            plt.title(report, wrap=True, loc="left")
            plt.axis("off")
            nx.draw(
                G,
                pos=pos_G,
                node_size=0,
                with_labels=False,
                arrows=False,
                node_color="k",
                edge_color="k",
                width=0.2,
            )
            nx.draw_networkx_nodes(
                G,
                pos=pos_G,
                node_size=30,
                cmap=plt.cm.RdYlBu,
                node_color=list(partition.values()),
            )

            # induced partition graph
            if contracted == True:
                pos_ind = {xy: xy for xy in ind.nodes()}

                graph.make_graphplot_for_targetnodes(ind, None, None, layout="graphviz")
                plt.title(report, wrap=True, loc="left")

                plt.figure(figsize=(8, 8))
                plt.title(report, wrap=True, loc="left")
                plt.axis("off")
                # base
                nx.draw(
                    G,
                    pos=pos_G,
                    node_size=0,
                    with_labels=False,
                    arrows=False,
                    node_color="gray",
                    edge_color="silver",
                    width=0.2,
                )
                nx.draw_networkx_nodes(
                    ind,
                    pos=pos_ind,
                    node_size=list(dict(ind.nodes(data="ind_size")).values()),
                    cmap=plt.cm.RdYlBu,
                    node_color=range(len(ind)),
                )
                nx.draw_networkx_edges(ind, pos_ind, alpha=0.3)

    def setup_pruning(self,
                      G: nx.Graph = None,
                      subgraph_fn:str = None,
                      edge_prune_query:str = None,
                      node_prune_query:str = None,
                      report:str = None,
                      **kwargs):
        """function to prune the 1D flow network"""

        # create the initial graph
        G = self._io_subgraph(subgraph_fn, G, 'r')

        # pruned graph
        PG = self.setup_subgraph(G, edge_query=edge_prune_query,node_query=node_prune_query
                                 # report ='plot sg for pruning'
                                 )

        # remained graph
        RG = graph.find_difference(G, PG)

        # graph connections pruned graph
        tree_roots = [n for n in RG if n in PG]
        # PG.add_edges_from([(n, -1) for n in tree_roots])
        # PG.remove_edges_from([(n, -1) for n in tree_roots])

        # apply DAG to pruned graph -  get summed information for root nodes
        PG_dag = self.setup_dag(PG, weight = 'geom_length', targets = tree_roots,
                                report = "plot dag for pruning"
                                )

        # add PG_dag node loads back to tree roots in SG
        loads = ["nnodes"]
        for load in loads:
            tree_roots_load_org = Counter({n[0]:n[-1] for n in PG.nodes(data = load) if n[-1] is not None })
            tree_roots_load_dag = Counter({n[0]:n[-1] for n in PG_dag.nodes(data = load) if n[-1] is not None })
            tree_roots_load_total = tree_roots_load_org + tree_roots_load_dag
            nx.set_node_attributes(RG, tree_roots_load_total, load)
        # TODO add geometry handling here --> # do we need a mapping csv

        # write into graph
        self._io_subgraph(subgraph_fn, G, 'w')

        # draw to see
        if report:
            # plot xy
            plt.figure(figsize=(8, 8))
            plt.title(report, wrap=True, loc="left")
            plt.axis("off")
            pos = {xy: xy for xy in G.nodes()}

            # base
            nx.draw(
                G,
                pos=pos,
                node_size=0,
                with_labels=False,
                arrows=False,
                node_color="gray",
                edge_color="silver",
                width=0.5,
            )
            nx.draw(
                RG,
                pos=pos,
                node_size=0,
                with_labels=False,
                arrows=False,
                node_color="gray",
                edge_color="gray",
                width=1,
            )


            size = list(dict(RG.nodes(data = "nnodes")).values())
            size = [0 if v is None else v for v in size]
            nx.draw_networkx_nodes(RG, pos, node_size = size, node_color = 'r')

        return None

    def _io_subgraph(self, subgraph_fn, G:nx.Graph = None, mode = 'r'):
        """function to assit the graph_fn handeling for subgraph"""

        # read
        if mode == 'r':
            if G is None:
                if subgraph_fn is None:
                    self.logger.debug(
                        f"using main graph."
                    )
                    G = self._graphmodel.copy()
                else:
                    if subgraph_fn in self._subgraphmodels:
                        self.logger.debug(
                            f"using subgraph instance {subgraph_fn}. "
                        )
                        G = self._subgraphmodels[subgraph_fn]
                    else:
                        self.logger.debug(
                            f"using main model{subgraph_fn}. "
                        )
                        G = self._graphmodel.copy()
            else:
                self.logger.debug(
                        f"no graph is read from subgraph_fn."
                    )

        # write
        if mode == 'w':
            if G is None:
                raise ValueError("no graph to write to subgraph_fn")
            else:
                if subgraph_fn in self._subgraphmodels:
                    self.logger.warning(
                        f"using given model and overwriting subgraph instance {subgraph_fn}. "
                    )
                    self._subgraphmodels[subgraph_fn] = G
                else:
                    self.logger.warning(
                        f"using given model and creating subgraph instance {subgraph_fn}. "
                    )
                    self._subgraphmodels[subgraph_fn] = G
        return G


    ## I/O
    def read(self):
        """Method to read the complete model schematization and configuration from file."""
        self.logger.info(f"Reading model data from {self.root}")
        self.read_config()
        self.read_staticmaps()
        self.read_staticgeoms()
        self.read_graphmodel()

    def write(self):  # complete model
        """Method to write the complete model schematization and configuration to file."""
        self.logger.info(f"Writing model data to {self.root}")
        # if in r, r+ mode, only write updated components
        if not self._write:
            self.logger.warning("Cannot write in read-only mode")
            return
        if self.config:  # try to read default if not yet set
            self.write_config()  # FIXME: config now isread from default, modified and saved temporaryly in the models folder --> being read by dfm and modify?
        if self._graphmodel:
            self.write_graphmodel()
        if self._staticmaps:
            self.write_staticmaps()
        if self._staticgeoms:
            self.write_staticgeoms()


    def read_staticmaps(self):
        """Read staticmaps at <root/?/> and parse to xarray Dataset"""
        # to read gdal raster files use: hydromt.open_mfraster()
        # to read netcdf use: xarray.open_dataset()
        if not self._write:
            # start fresh in read-only mode
            self._staticmaps = xr.Dataset()
        self.set_staticmaps(hydromt.open_mfraster(join(self.root, "*.tif")))

    def write_staticmaps(self):
        """Write staticmaps at <root/?/> in model ready format"""
        # to write to gdal raster files use: self.staticmaps.raster.to_mapstack()
        # to write to netcdf use: self.staticmaps.to_netcdf()
        if not self._write:
            raise IOError("Model opened in read-only mode")
        self.staticmaps.raster.to_mapstack(join(self.root, "dflowfm"))

    def read_staticgeoms(self):
        """Read staticgeoms at <root/?/> and parse to dict of geopandas"""
        if not self._write:
            # start fresh in read-only mode
            self._staticgeoms = dict()
        for fn in glob.glob(join(self.root, "staticgeoms", "*.shp")):
            name = basename(fn).replace(".shp", "")
            geom = hydromt.open_vector(fn, driver="shp", crs=self.crs)
            self.set_staticgeoms(geom, name)

    def write_staticgeoms(self):  # write_all()
        """Write staticmaps at <root/?/> in model ready format"""
        # to write use self.staticgeoms[var].to_file()
        if not self._write:
            raise IOError("Model opened in read-only mode")
        for name, gdf in self.staticgeoms.items():
            fn_out = join(self.root, "staticgeoms", f"{name}.shp")
            gdf.to_file(fn_out)
            # FIXME solve the issue when output columns are too long

    def read_graphmodel(self):
        """Read graphmodel at <root/?/> and parse to networkx DiGraph"""
        if not self._write:
            # start fresh in read-only mode
            self._graphmodel = None
        fn = join(self.root, "graphmodel.gpickle")
        self._graphmodel = nx.read_gpickle(fn)

    def write_graphmodel(self):
        """write graphmodel at <root/?/> in model ready format"""

        # report
        figs = [plt.figure(n) for n in plt.get_fignums()]
        helper.multipage(join(self.root, "graph", "report.pdf"), figs=figs)

        # write graph
        outdir = join(self.root, "graph")
        self._write_graph(self._graphmodel, outdir, outname="graph")

        # write subgraph
        if self._subgraphmodels:
            for subgraph_fn, subgraphmodel in self._subgraphmodels.items():
                if subgraphmodel:
                    self._write_graph(subgraphmodel, outdir, outname=subgraph_fn)

    def _write_graph(self, G: nx.Graph, outdir: str, outname: str = "graph"):

        # write pickle
        nx.write_gpickle(G.copy(), join(outdir, f"{outname}.gpickle"))

        # write edges
        shp = gpd.GeoDataFrame(nx.to_pandas_edgelist(G).set_index("id"))
        shp.drop(columns=["source", "target"]).to_file(
            join(outdir, f"{outname}_edges.shp")
        ) # drop them because they are tuple

        # write nodes
        df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index').reset_index()
        if len(df) == 0: # no attribute
            df = pd.DataFrame(dict(G.nodes(data=True)).keys())
            shp = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df[0], df[1]))
            shp.drop(columns=[0, 1]).to_file(
                join(outdir, f"{outname}_nodes.shp")
            )  # drop them because they are tuple
        else:
            shp = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df.level_0, df.level_1))
            shp.drop(columns=["level_0", "level_1"]).to_file(
                join(outdir, f"{outname}_nodes.shp")
            ) # drop them because they are tuple

    def read_forcing(self):
        """Read forcing at <root/?/> and parse to dict of xr.DataArray"""
        return self._forcing
        # raise NotImplementedError()

    def write_forcing(self):
        """write forcing at <root/?/> in model ready format"""
        pass
        # raise NotImplementedError()

    def read_states(self):
        """Read states at <root/?/> and parse to dict of xr.DataArray"""
        return self._states
        # raise NotImplementedError()

    def write_states(self):
        """write states at <root/?/> in model ready format"""
        pass
        # raise NotImplementedError()

    def read_results(self):
        """Read results at <root/?/> and parse to dict of xr.DataArray"""
        return self._results
        # raise NotImplementedError()

    def write_results(self):
        """write results at <root/?/> in model ready format"""
        pass
        # raise NotImplementedError()

    @property
    def crs(self):
        return pyproj.CRS.from_epsg(self.get_config("global.epsg", fallback=4326))

    @property
    def graphmodel(self):
        if self._graphmodel == None:
            self._graphmodel = nx.DiGraph()
        return self._graphmodel
