"""Implement plugin model class"""

import glob
from os.path import join, basename, isfile
from pathlib import Path
import logging
import matplotlib.pyplot as plt

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
        data_libs=None,  # yml # TODO: how to choose global mapping files (.csv) and project specific mapping files (.csv)
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
        graph_class: str = "Graph",
        **kwargs,
    ):
        """Define the model region.

        Adds model layer:
        * **region** geom: model region

        Parameters
        ----------
        region: dict
            Dictionary describing region of interest, e.g. {'bbox': [xmin, ymin, xmax, ymax]}.
            See :py:meth:`~hydromt.workflows.parse_region()` for all options.
        graph_class: str
            Networkx Class names describing which class should be used for the graph model, e.g. ['Graph', 'DiGraph', 'MultiGraph', 'MultiDiGraph']
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

        # Set the graph model class
        assert graph_class in [
            "Graph",
            "DiGraph",
            "MultiGraph",
            "MultiDiGraph",
        ], "graph class not recognised"
        self._graphmodel = eval(f"nx.{graph_class}()")

    def setup_edges(
        self,
        edges_fn: str,
        id_col: str,
        attribute_cols: list = [],
        use_location: bool = False,
        **kwargs,
    ):
        """This component add edges or edges attributes to the graph model

        Adds model layers (use_location == True):

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
            attribute_cols = []

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
            geom=None,  # FIXME use geom region gives error "geopandas Invalid projection: : (Internal Proj Error: proj_create: unrecognized format / unknown name)"
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
                    if "geometry" in v:
                        # ensure type of geoseries - might be a bug in pandas / geopandas
                        _ = type(df["geometry"])
                    df[k] = df.eval(v)
                    self.logger.debug(f"update column {k} based on {v}")
                except Exception as e:
                    self.logger.debug(f"can not update column {k} based on {v}: {e}")

        return df

    def setup_subgraph(
        self,
        subgraph_fn: str = "subgraph",
        edge_query: str = None,
        node_query: str = None,
        algorithm: int = 0,
        visualise: bool = False,
        **kwargs,
    ):
        """This component prepares the subgraph from a graph

        In progress

        Parameters
        ----------
        subgraph_fn : str
            Name of the subgraph. Used in staticgeom and model writers

        Arguments
        ----------
        edge_query : str
            Conditions to query the subgraph from graph.
            Applies on existing attributes of edges, so make sure to add the attributes first.
        algorithm : str
            Methods to ensure connectivity of the resulting subgraph, e.g. 0,1,2,3,4
            Methods in testing:
            0. do not apply any method - disconnected subgraph
            1. patch paths from queried nodes in SG to an signed outlet
            2. steine rtree approximation for nodes in SG (weight = simple distance)
            3. steine rtree approximation for nodes in SG (weight = queried importance)
            4. steine rtree approximation for nodes in SG (weight = geometry distance)
            # FIXME: add weight as argument; combine setup_dag function

        """

        if edge_query is None:
            self.logger.debug(f"use graph for subgraph")
            SG = self._graphmodel
        else:
            self.logger.debug(f"query subgraph based on edge_query {edge_query}")
            SG = graph.query_graph_edges_attributes(
                self._graphmodel,
                id_col="id",
                edge_query=edge_query,
            )
            self.logger.debug(
                f"{len(SG.edges())}/{len(self._graphmodel.edges())} edges are selected"
            )

        self._subgraphmodels[subgraph_fn] = SG
        nodes = [n for n in self._subgraphmodels[subgraph_fn].nodes]

        #   FIXME: add node query

        # TODO: how to query a subset of edges yet retain the connectivity of the network?
        if algorithm == 0:
            # 0. do not apply any method - disconnected subgraph
            self.logger.debug(f"do not apply any method - disconnected subgraph")
            pass

        elif algorithm == 1:
            # 1. patch paths from queried nodes in SG to an signed outlet
            self.logger.debug(
                f"patch paths from queried nodes in SG to an signed outlet"
            )
            G = self._graphmodel.copy()
            SG = self._subgraphmodels[subgraph_fn].copy()
            # FIXME hard coded outlet
            outlet = (662779.7744999994, 1524628.4437000006)
            i = 0
            for n in nodes:
                if nx.has_path(SG, n, outlet):
                    pass
                elif nx.has_path(G, n, outlet):  # directional path exist
                    nx.add_path(SG, nx.dijkstra_path(G, n, outlet))
                if nx.has_path(
                    G.to_undirected(), n, outlet
                ):  # unidirectional path exist
                    nx.add_path(SG, nx.dijkstra_path(G.to_undirected(), n, outlet))
                else:
                    i += 1
                    print(f"No path to {outlet}")
            self._subgraphmodels[subgraph_fn] = SG

        elif algorithm == 2:
            # 2. steine rtree approximation for nodes in SG (weight = simple distance)
            self.logger.debug(
                f"steine rtree approximation for nodes in SG (weight = simple distance)"
            )
            G = self._graphmodel.copy()
            SG = self._subgraphmodels[subgraph_fn].copy()
            e = {(u, v): {"weight": 1} for u, v in G.edges()}
            G = nx.Graph()
            G.add_edges_from(e)
            SG = nx.algorithms.approximation.steinertree.steiner_tree(G, nodes)
            self._subgraphmodels[subgraph_fn] = SG

        elif algorithm == 3:
            # 3. steine rtree approximation for nodes in SG (weight = queried importance)
            self.logger.debug(
                f"steine rtree approximation for nodes in SG (weight = queried importance)"
            )
            G = self._graphmodel.copy()
            SG = self._subgraphmodels[subgraph_fn].copy()
            e = {(u, v): {"weight": 0} for u, v in G.edges()}
            e.update({(u, v): {"weight": 1000} for u, v in SG.edges()})
            G = nx.Graph()
            G.add_edges_from(e)
            SG = nx.algorithms.approximation.steinertree.steiner_tree(G, nodes)
            self._subgraphmodels[subgraph_fn] = SG

        elif algorithm == 4:
            # 4. steine rtree approximation for nodes in SG (weight = geometry distance)
            self.logger.debug(
                f"steine rtree approximation for nodes in SG (weight = geometry distance)"
            )
            G = self._graphmodel.copy()
            SG = self._subgraphmodels[subgraph_fn].copy()
            e = {(u, v): {"weight": d["geom_length"]} for u, v, d in G.edges(data=True)}
            G = nx.Graph()
            G.add_edges_from(e)
            SG = nx.algorithms.approximation.steinertree.steiner_tree(G, nodes)
            self._subgraphmodels[subgraph_fn] = SG

        # add marker/tracer of the subgraph
        nx.set_edge_attributes(
            self._subgraphmodels[subgraph_fn], values=True, name=subgraph_fn
        )
        nx.set_edge_attributes(self._graphmodel, values=True, name=subgraph_fn)

        # visualise
        if visualise == True:

            G = self._graphmodel.copy()
            SG = self._subgraphmodels[subgraph_fn].copy()
            plt.figure(figsize=(8, 8))
            plt.title(
                f"Subgraph in red: subgraph_fn = {subgraph_fn}; algorithm = {algorithm}"
            )
            plt.axis("off")
            pos_G = {xy: xy for xy in G.nodes()}
            nx.draw_networkx_nodes(G, pos=pos_G, node_size=10, node_color="k")
            nx.draw_networkx_edges(G, pos=pos_G, edge_color="k", arrows=False)
            pos_SG = {xy: xy for xy in SG.nodes()}
            nx.draw_networkx_nodes(SG, pos=pos_SG, node_size=10, node_color="r")
            nx.draw_networkx_edges(SG, pos=pos_SG, edge_color="r", arrows=False)

    def setup_dag(
        self,
        subgraph_fn: str = None,
        target_query: str = None,
        weight: str = None,
        algorithm: str = "dijkstra",
        visualise: bool = False,
        **kwargs,
    ):
        """This component prepares subgraph as Directed Acyclic Graphs (dag) using shortest path
        step 1: add a supernode to subgraph (representing graph - subgraph)
        step 2: use shortest path to prepare dag edges (source: node in subgraph; target: super node)

        in progress

        Parameters
        ----------
        subgraph_fn : string, optional (default = None)
            Name of the subgraph to apply the method on.
            If None, the full graph will be used.
        target_query : None or string, optional (default = None)
            Edges query used for creating super target.
            If None, a target node will be any node with out_degree == 0.
            If a string, use this to query part of graph as a super target node.
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

        """

        # check input parameter and arguments
        if subgraph_fn is None:
            self.logger.debug(f"making dag for graph model")
            G = self._graphmodel
        elif subgraph_fn in self._subgraphmodels.keys():
            self.logger.debug(f"making dag for {subgraph_fn} model")
            G = self._subgraphmodels[subgraph_fn]
        else:
            raise ValueError(
                f"{subgraph_fn} model does not exist. create one using [setup_subgraph]."
            )

        if target_query is None:
            targets = [n for n in G.nodes if G.out_degree[n] == 0]
        else:
            _target_G = graph.query_graph_edges_attributes(
                G,
                edge_query=target_query,
            )
            G, targets = graph.contract_graph_nodes(G, _target_G.nodes)
        self.logger.debug(f"target are {targets}")

        if weight is None:
            algorithm = "unweighted"

        if algorithm not in ("dijkstra", "bellman-ford", "unweighted"):
            raise ValueError(f"algorithm not supported: {algorithm}")

        # started making dag
        if isinstance(G, nx.DiGraph):
            G = G.to_undirected()

        DAG = nx.DiGraph()

        # 1. add path
        for t in targets:
            for _ in nx.connected_components(G):
                SG = G.subgraph(
                    _
                )  # FIXME: if dont do this step: networkx.exception.NetworkXNoPath: No path to **.
                for n in SG.nodes:
                    path = nx.shortest_path(SG, n, t, weight=weight)
                    nx.add_path(DAG, path)

        # 2. add back weights
        for u, v, new_d in DAG.edges(data=True):
            # get the old data from X
            old_d = G[u].get(v)
            non_shared = set(old_d) - set(new_d)
            if non_shared:
                # add old weights to new weights if not in new data
                new_d.update(dict((key, old_d[key]) for key in non_shared))

        # visualise
        if visualise == True:
            # before remove target
            graph.plot_xy(DAG)
            plt.title(f"DAG: subgraph_fn =  {subgraph_fn}")
            # after remove target
            DAG.remove_nodes_from(targets)
            graph.plot_xy(DAG)
            plt.title(f"DAG: subgraph_fn =  {subgraph_fn}: remove targets")

    def setup_partition(
        self,
        subgraph_fn: str = "subgraph",
        algorithm: str = "simple",
        visualise: bool = False,
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

        G = self._graphmodel

        SG = self._subgraphmodels[subgraph_fn]
        UG = SG.to_undirected()

        DIFF = G.copy()
        DIFF.remove_edges_from(e for e in G.edges() if e in SG.edges())
        DIFF.remove_nodes_from(list(nx.isolates(DIFF)))

        # UG
        # further partition
        partition = {n: -1 for n in SG.nodes}
        if algorithm == "simple":  # based on connected subgraphs
            for i, ig in enumerate(nx.connected_components(UG)):
                partition.update({n: i for n in ig})
        elif algorithm == "louvain":  # based on louvain algorithm
            partition.update(community.best_partition(UG))
        else:
            raise ValueError(
                f"{algorithm} is not recognised. allowed algorithms: simple, louvain"
            )

        # finish partition
        n_partitions = max(partition.values())
        self.logger.debug(
            f"{n_partitions} partitions is derived from subgraph using {algorithm} algorithm"
        )
        # edges_partitions = {(s, t): partition[s] for s, t in G.edges() if partition[s] == partition[t]}

        # induced graph from the partitions
        ind = community.induced_graph(partition, SG)
        ind.remove_edges_from(nx.selfloop_edges(ind))
        # pos = nx.spring_layout(ind)
        # pos = nx.drawing.nx_agraph.pygraphviz_layout(ind, prog='dot', args='')

        # visualise
        if visualise == True:

            # Cartesian coordinates centroid
            pos_SG = {xy: xy for xy in SG.nodes()}
            pos_ind = {}
            for k, v in partition.items():
                pos_ind.setdefault(v, []).append(k)
            pos_ind = {k: np.mean(v, axis=0) for k, v in pos_ind.items()}
            pos_DIFF = {xy: xy for xy in DIFF.nodes()}

            # partitions
            plt.figure(figsize=(8, 8))
            plt.title(
                f"Partitions in color: subgraph_fn = {subgraph_fn}; algorithm = {algorithm}"
            )
            plt.axis("off")
            # add DIFF
            nx.draw_networkx_nodes(DIFF, pos=pos_DIFF, node_size=10, node_color="k")
            nx.draw_networkx_edges(DIFF, pos=pos_DIFF, edge_color="k", arrows=False)

            nx.draw_networkx_nodes(
                SG,
                pos=pos_SG,
                node_size=30,
                cmap=plt.cm.RdYlBu,
                node_color=list(partition.values()),
            )
            nx.draw_networkx_edges(SG, pos_SG, alpha=0.3)

            # induced partition graph
            plt.figure(figsize=(8, 8))
            plt.title(
                f"Indiced graph from partitions in color: subgraph_fn = {subgraph_fn}; algorithm = {algorithm}"
            )
            plt.axis("off")

            # add DIFF
            nx.draw_networkx_nodes(DIFF, pos=pos_DIFF, node_size=10, node_color="k")
            nx.draw_networkx_edges(DIFF, pos=pos_DIFF, edge_color="k", arrows=False)

            nx.draw_networkx_nodes(
                ind, pos=pos_ind, node_size=30, cmap=plt.cm.RdYlBu, node_color=list(ind)
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
        fn_out = join(self.root, "graph", f"graphmodel.gpickle")
        nx.write_gpickle(self._graphmodel, fn_out)
        # IO
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        def multipage(filename, figs=None, dpi=200):
            pp = PdfPages(filename)
            if figs is None:
                figs = [plt.figure(n) for n in plt.get_fignums()]
            for fig in figs:
                fig.savefig(pp, format="pdf")
            pp.close()

        multipage(join(self.root, "graph", "graphmodel.pdf"))

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
