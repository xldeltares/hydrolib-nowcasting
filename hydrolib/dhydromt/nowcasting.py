"""Implement plugin model class"""

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
            graph_class: str,
            edges_fn: str = None,
            edges_id_col: str = None,
            nodes_fn: str = None,
            nodes_id_col: str = None,
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
            raise ValueError("graph model can only be created once. Remove duplicated [setup_graph] section. ")

        assert graph_class in [
            "Graph",
            "DiGraph",
            "MultiGraph",
            "MultiDiGraph",
        ], "graph class not recognised"
        self._graphmodel = eval(f"nx.{graph_class}()")

        if edges_fn is not None and edges_id_col is not None:
            self._setup_edges(edges_fn=edges_fn, id_col=edges_id_col, use_location=True)

        if nodes_fn is not None and nodes_id_col is not None:
            raise NotImplementedError(
                "assigning nodes to graph model is  not yet implemented"
            )

        if report:
            G = self._graphmodel
            graph.plot_xy(G)
            plt.title(report, wrap=True, loc="left")

    def _setup_edges(
            self,
            edges_fn: str,
            id_col: str,
            attribute_cols: list = None,
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

        Arguments
        ----------
        use_location : bool
            If True, the edges will be added; if False, only edges attributes are added.
            Latter will be mapped to id.

        """

        # parameter handling
        if attribute_cols is None:
            attribute_cols = ["geometry"]

        if edges_fn is None:
            logger.error("edges_fn must be specified.")

        self.logger.info(f"Adding edges to graph.")
        edges = self._get_geodataframe(edges_fn)

        assert set([id_col] + attribute_cols).issubset(
            edges.columns
        ), f"id and/or attribute cols do not exist in {edges.columns}"

        if use_location is True:
            self.logger.debug(f"Adding edges with id.")
            self._graphmodel = graph.add_edges_with_id(
                self._graphmodel, edges=edges, id_col=id_col
            )

        if len(attribute_cols) > 0:
            self.logger.debug(f"updating edges attributes")
            self._graphmodel = graph.update_edges_attributes(
                self._graphmodel, edges=edges[[id_col] + attribute_cols], id_col=id_col
            )

        self.logger.debug(f"Adding edges vector to staticgeoms as {edges_fn}.")
        self.set_staticgeoms(edges, edges_fn)

    def _setup_nodes(self, **kwargs):
        """This component add nodes or nodes attributes to the graph model

        * **nodes** geom: vector

        raise NotImplementedError
        """
        raise NotImplementedError

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
            self.logger.debug(f"updating {path_or_key}")

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
                        self.logger.debug(f"can not update column {k} based on {v}: {e}")

        # post processing
        df = df.drop_duplicates()
        self.logger.debug(f"drop duplicates in dataframe")

        self.logger.info(f"{len(df)} features read from {path_or_key}")

        return df

    def update_edges(
            self, edges_fn: str, id_col: str, attribute_cols: list = None, **kwargs
    ):
        """This component update edges attributes to the graph model

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

        See Also
        --------
        _setup_edges
        """

        self._setup_edges(
            edges_fn=edges_fn,
            id_col=id_col,
            attribute_cols=attribute_cols,
            use_location=False,
        )

    def setup_subgraph(
            self,
            subgraph_fn: str = None,
            edge_query: str = None,
            node_query: str = None,
            algorithm: str = None,
            targets=None,
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
            self.logger.warning(f"subgraph instance {subgraph_fn} already exist, apply setup_subgraph on subgraph_fn and overwrite.")
            G = self._subgraphmodels[subgraph_fn].copy()
        elif subgraph_fn:
            self.logger.debug(f"subgraph instance {subgraph_fn} will be created from graph")
            G = self._graphmodel.copy()
        else:
            self.logger.debug(f"graph will be updated.")
            G = self._graphmodel.copy()

        if edge_query is None:
            SG = G
        else:
            self.logger.debug(f"query sedges: {edge_query}")
            SG = graph.query_graph_edges_attributes(
                G,
                id_col="id",
                edge_query=edge_query,
            )
            self.logger.debug(
                f"{len(SG.edges())}/{len(G.edges())} edges are selected"
            )

        # FIXME: add node query
        if algorithm is not None:
            if algorithm not in ("patch", "mst", "dag"):
                raise ValueError(f"algorithm not supported: {algorithm}")

        if targets is None:
            targets = [n for n in SG.nodes if SG.out_degree[n] == 0]
        elif isinstance(targets, str):
            try:
                _target_G = graph.query_graph_edges_attributes(
                    SG,
                    edge_query=targets,
                )
                targets = [v for u,v in _target_G.edges()]

            except:
                pass
        elif isinstance(targets, list):
            assert set(targets).issubset(set(SG.nodes)), "targets must be the nodes id"
        self.logger.debug(f"target are {targets}")

        if weight is not None:
            assert weight in list(SG.edges(data=True))[0][2].keys(), f"edge weight {weight} does not exist!"

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
            self.logger.debug(
                f"patch paths from SG nodes to targets"
            )

            SG_nodes = list(SG.nodes)
            for outlet in targets:
                for n in SG_nodes:
                    if nx.has_path(SG, n, outlet):
                        pass
                    elif nx.has_path(G, n, outlet):  # directional path exist
                        nx.add_path(SG, nx.dijkstra_path(G, n, outlet))
                    if nx.has_path(
                            UG, n, outlet
                    ):  # unidirectional path exist
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
            SG = nx.algorithms.approximation.steinertree.steiner_tree(UG, SG_nodes, weight=weight)

        elif algorithm == 'dag':
            # dag
            self.logger.debug(
                f"creating dag from SG nodes to targets"
            )
            SG = self._setup_dag(SG, targets=targets, weight=weight, **kwargs)

        # assign SG
        if subgraph_fn:
            self._subgraphmodels[subgraph_fn] = SG
            nx.set_edge_attributes(
                self._subgraphmodels[subgraph_fn], values=True, name=subgraph_fn
            )
            # add subgraph_fn as attribute in the G
            nx.set_edge_attributes(
                self._graphmodel,
                {e: {subgraph_fn: True} for e in self._subgraphmodels[subgraph_fn].edges()},
            )
        else:
            self._graphmodel = SG

        # report
        if report:

            # plot graphviz
            graph.make_graphplot_for_targetnodes(SG, targets, layout='graphviz')
            plt.title(report, wrap=True, loc="left")

            # plot xy
            plt.figure(figsize=(8, 8))
            plt.title(report, wrap=True, loc="left")
            plt.axis("off")
            pos = {xy: xy for xy in G.nodes()}

            # base
            nx.draw(G, pos=pos, node_size=0, with_labels=False, arrows=False, node_color='gray',
                    edge_color='silver',
                    width=0.5)
            # SG nodes and edges
            if algorithm == 'dag':
                nx.draw_networkx_nodes(G, pos=pos, nodelist=targets, node_size=200, node_shape="*", node_color='r')
                edge_width = [d[2] / 100 for d in SG.edges(data="nnodes_upstream")]
                nx.draw_networkx_edges(SG, pos=pos, edgelist=SG.edges(), arrows=False,
                                       width=[float(i) / (max(edge_width) + 0.1) * 20 + 0.5 for i in edge_width])
            else:
                edge_width = [0 / 100 for d in SG.edges()]
                nx.draw_networkx_edges(SG, pos=pos, edgelist=SG.edges(), arrows=False,
                                       width=[float(i) / (max(edge_width) + 0.1) * 20 + 0.5 for i in edge_width])


    def _setup_dag(
            self,
            G: nx.Graph = None,
            targets=None,
            weight: str = None,
            algorithm: str = "dijkstra",
            report: str = None,
            use_super_target:bool = True,
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

        if G is not None:
            if isinstance(G, nx.DiGraph):
                G = G.to_undirected()
            pass
        else:
            return G

        if targets is None:
            targets = [n for n in G.nodes if G.out_degree[n] == 0]
        elif isinstance(targets, str):
            try:
                _target_G = graph.query_graph_edges_attributes(
                    G,
                    edge_query=targets,
                )
                G, targets = graph.contract_graph_nodes(G, _target_G.nodes)
            except:
                pass
        elif isinstance(targets, list):
            assert set(targets).issubset(set(G.nodes)), "targets must be the nodes id"
        self.logger.debug(f"target are {targets}")

        if algorithm not in ("dijkstra", "bellman-ford"):
            raise ValueError(f"algorithm not supported: {algorithm}")

        # started making dag
        DAG = nx.DiGraph()

        # 1. add path
        for _ in nx.connected_components(G):
            SG = G.subgraph(_).copy()
            # FIXME: if don't do this step: networkx.exception.NetworkXNoPath: No path to **.
            if use_super_target:
                # add super target
                _t = [-1]
                if weight == 'streetlev':
                    # FIXME temporary
                    # mirror weights at super edges
                    for w in ["geom_length", 'streetlev']:
                        SG.add_weighted_edges_from([(t,
                                                    _t[0],
                                                    max([k[1][w] if w in k[1] else 0 for k in SG[t].items() ]))
                                                    for t in list(set(SG.nodes) & set(targets))], weight = w)
                else:
                    SG.add_edges_from([(t,_t[0]) for t in list(set(SG.nodes) & set(targets))])
            else:
                _t = targets
            for t in _t:
                for n in SG.nodes:
                    if n not in DAG:
                        if weight == 'streetlev':
                            # FIXME temporary
                            smax = max([k[1]['streetlev'] for k in SG[t].items()])
                            path = nx.shortest_path(SG, n, t, weight=lambda u, v, e: 10-smax/e['geom_length'], method=algorithm)
                        else:
                            path = nx.shortest_path(SG, n, t, weight=weight, method=algorithm)
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
        for s, t in _.edges():
            nnodes = len(list(nx.dfs_postorder_nodes(_, t)))
            DAG[t][s].update({'nnodes_upstream': nnodes})

        # validate DAG
        if nx.is_directed_acyclic_graph(DAG):
            self.logger.debug("dag is directed acyclic graph")
        else:
            self.logger.error("dag is NOT directed acyclic graph")

        # visualise DAG
        if report:

            # plot graphviz
            graph.make_graphplot_for_targetnodes(DAG, targets, layout='graphviz')
            plt.title(report, wrap=True, loc="left")

            # plot xy
            plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.title(report)
            pos = {xy: xy for xy in DAG.nodes()}
            # base
            nx.draw(G, pos=pos, node_size=0, with_labels=False, arrows=False, node_color='gray', edge_color='silver',
                    width=1)
            # nodes dag
            nx.draw_networkx_nodes(G, pos=pos, nodelist=targets, node_size=50, node_shape="*", node_color='r')
            # edges dag
            edge_width = [d[2] / 100 for d in DAG.edges(data="nnodes_upstream")]
            nx.draw_networkx_edges(DAG, pos=pos, edgelist=DAG.edges(), arrows=False,
                                   width=[float(i) / (max(edge_width) + 0.1) * 20 + 0.5 for i in edge_width])
            nx.draw_networkx_nodes(DAG, pos=pos, nodelist=targets, node_size=50, node_shape="*", node_color='r')

        self._daggraphmodel = DAG
        return DAG

    def setup_partition(
            self,
            subgraph_fn: str = "subgraph",
            algorithm: str = "simple",
            report: str = None,
            **kwargs,
    ):
        """This component prepares the partition from subgraph

        in progress

        Parameters
        ----------
        algorithm : str
            Algorithm to derive partitions from subgraph. Options: ['louvain', 'simple']
            testing methods:
            "simple" based on connected subgraphs
            "louvain":  # based on louvain algorithm
        """

        # algorithm louvain
        import community
        import matplotlib.pyplot as plt
        import numpy as np

        G = self._graphmodel.copy()
        G_targets = [n for n in G.nodes if G .out_degree[n] == 0]

        SG = self._subgraphmodels[subgraph_fn].copy()
        SG_targets = [n for n in SG.nodes if SG.out_degree[n] == 0]

        # UG for further partition
        UG = SG.to_undirected()
        partition = {n: -1 for n in G.nodes}
        if algorithm == "simple":  # based on connected subgraphs
            for i, ig in enumerate(nx.connected_components(UG)):
                partition.update({n: i for n in ig})
        elif algorithm == "louvain":  # based on louvain algorithm
            partition.update(community.best_partition(UG))
        else:
            raise ValueError(
                f"{algorithm} is not recognised. allowed algorithms: simple, louvain"
            )
        n_partitions = max(partition.values())
        self.logger.debug(
            f"{n_partitions} partitions is derived from subgraph using {algorithm} algorithm"
        )
        # edges_partitions = {(s, t): partition[s] for s, t in G.edges() if partition[s] == partition[t]}

        # induced graph from the partitions - faster but a bit confusing results
        # ind = community.induced_graph(partition, G)
        # ind.remove_edges_from(nx.selfloop_edges(ind))

        # contracted graph - slower but neater
        # use both G and SG
        ind = self._graphmodel.copy()
        nx.set_node_attributes(ind,{n:{'ind_size':1} for n in ind.nodes})
        for part in np.unique(list(partition.values())):
            part_nodes = [n for n in partition if partition[n] == part]
            if part == -1:
                # do not contract
                pass
            else:
                for to_node in [n for n in part_nodes if n in SG_targets]:
                    ind, targets = graph.contract_graph_nodes(ind, part_nodes, to_node)
                    ind.nodes[to_node]['ind_size'] = len(part_nodes)

        # visualise
        if report:

            # Cartesian coordinates centroid
            pos_G = {xy: xy for xy in G.nodes()}
            pos_SG = {xy: xy for xy in SG.nodes()}
            pos_ind = {xy: xy for xy in ind.nodes()}

            # partitions
            plt.figure(figsize=(8, 8))
            plt.title(report, wrap=True, loc="left")
            plt.axis("off")
            nx.draw_networkx_nodes(
                G,
                pos=pos_G,
                node_size=30,
                cmap=plt.cm.RdYlBu,
                node_color=list(partition.values()),
            )
            nx.draw_networkx_edges(G, pos_G, alpha=0.3)

            # induced partition graph
            graph.make_graphplot_for_targetnodes(
                ind, None, None, layout="graphviz"
            )
            plt.title(report, wrap=True, loc="left")

            plt.figure(figsize=(8, 8))
            plt.title(report, wrap=True, loc="left")
            plt.axis("off")
            # base
            nx.draw(G, pos=pos_G, node_size=0, with_labels=False, arrows=False, node_color='gray', edge_color='silver',
                    width=0.2)
            nx.draw_networkx_nodes(
                ind, pos=pos_ind, node_size=list(dict(ind.nodes(data = "ind_size")).values()),
                cmap=plt.cm.RdYlBu, node_color=range(len(ind))
            )
            nx.draw_networkx_edges(ind, pos_ind, alpha=0.3)



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
        if self._staticmaps:
            self.write_staticmaps()
        if self._staticgeoms:
            self.write_staticgeoms()
        if self._graphmodel:
            self.write_graphmodel()

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

        # write sugraph
        if self._subgraphmodels:
            for subgraph_fn, subgraphmodel in self._subgraphmodels.items():
                if subgraphmodel:
                    self._write_graph(
                        subgraphmodel, outdir, outname=subgraph_fn
                    )


    def _write_graph(self, G: nx.Graph, outdir: str, outname: str = "graph"):

        # write pickle
        # nx.write_gpickle(G.copy(), join(outdir, f"{outname}.gpickle"))

        # write edges
        shp = gpd.GeoDataFrame(nx.to_pandas_edgelist(G).set_index("id"))
        shp.drop(columns=["source", "target"]).to_file(
            join(outdir, f"{outname}_edges.shp")
        )

        # TODO write nodes

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
