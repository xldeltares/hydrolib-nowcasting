"""Implement plugin model class"""

import glob
from os.path import join, basename, isfile
import logging

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

from pathlib import Path

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
        self._subgraphmodel = None

    def setup_basemaps(
        self,
        region,
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

    def setup_branches(
        self,
        branches_fn: str,
        preprocess_branches: bool = False,
        id_col: str = "branchId",
        type_col: str = "branchType",
        use_attributes: list = [],
        **kwargs,
    ):
        """This component prepares the graph model with basic links

        Adds model layers:

        * **branches** geom: vector

        Parameters
        ----------
        branches_fn : str
            Name of data source for branches parameters, see data/data_sources.yml.
            * Required variables: [branchId, branchType]
            * Optional variables: []
        use_attributes : list
            A list of the columns in the data source to be used as graph edges attributes.
        """
        self.logger.info(f"Preparing 1D branches.")

        if branches_fn is None:
            logger.error("branches_fn must be specified.")

        branches = self.data_catalog.get_geodataframe(
            branches_fn,
            geom=None,  # FIXME use geom region gives error "geopandas Invalid projection: : (Internal Proj Error: proj_create: unrecognized format / unknown name)"
        )

        # check if the branch has been preprocessed
        if preprocess_branches == True:
            raise NotImplementedError("branches must have been processed.")

        self.logger.debug(f"Adding branches vector to staticgeoms.")
        self.set_staticgeoms(branches, "branches")

        self.logger.debug(f"Adding branches id and type to graph model.")
        self._graphmodel = graph.create_graph_from_branches(branches, id_col=id_col)
        self._graphmodel = graph.update_graph_edges_attributes(
            self._graphmodel, attr_df=branches[[id_col, type_col]].set_index(id_col)
        )
        self._graphmodel_edgeidcol = id_col

        if len(use_attributes) > 0:
            self.logger.debug(
                f"Adding additional branches attributes {use_attributes} to graph model."
            )
            self._graphmodel = graph.update_graph_edges_attributes(
                self._graphmodel,
                attr_df=branches[[id_col] + use_attributes].set_index(id_col),
            )

    def setup_structures(
        self,
        structures_fn: str,
        preprocess_structures: bool = False,
        id_col: str = "structId",
        type_col: str = "structType",
        use_attributes: list = [],
        visualise: bool = False,
        **kwargs,
    ):
        """This component prepares the graph with special links

        Adds model layers:

        * **structures** geom: vector

        Parameters
        ----------
        structures_fn : str
            Name of data source for structures parameters, see data/data_sources.yml.
            * Required variables: [structureId, structureType]
            * Optional variables: []
        use_attributes : list
            A list of the columns in the data source to be used as graph edges attributes.
        """
        self.logger.info(f"Preparing 1D structures.")

        if structures_fn is None:
            logger.error("structures_fn must be specified.")

        structures = self.data_catalog.get_geodataframe(
            structures_fn,
            geom=None,  # FIXME use geom region gives error "geopandas Invalid projection: : (Internal Proj Error: proj_create: unrecognized format / unknown name)"
        )

        # check if the branch has been preprocessed
        if preprocess_structures == True:
            raise NotImplementedError("structures must have been processed.")

        self.logger.debug(f"Adding structures vector to staticgeoms.")
        if "structures" in self.staticgeoms:
            _structures = self.staticgeoms["structures"]
            structures = gpd.GeoDataFrame(
                pd.concat([_structures, structures]).drop_duplicates()
            )
            self.set_staticgeoms(structures, "structures")

        else:
            self.set_staticgeoms(structures, "structures")

        # check if graph model exist
        if self._graphmodel is None:
            raise ValueError(
                "graph model does not exist. Set up graph model using branches first"
            )

        self.logger.debug(f"Adding structures id and type to graph edges.")
        edge_id_col = self._graphmodel_edgeidcol
        self._graphmodel = graph.update_graph_edges_attributes(
            self._graphmodel,
            attr_df=structures[[edge_id_col, id_col, type_col]].set_index(edge_id_col),
        )

        if len(use_attributes) > 0:
            self.logger.debug(
                f"Adding additional structure attributes {use_attributes} to graph model."
            )
            self._graphmodel = graph.update_graph_edges_attributes(
                self._graphmodel,
                attr_df=structures[[edge_id_col] + use_attributes].set_index(
                    edge_id_col
                ),
            )

        if visualise == True:
            f1, f2 = graph.plot_graph(self._graphmodel)
            f1[1].set_title("graphmodel")

    def setup_subgraph(
        self, edge_query: str = None, node_query: str = None, visualise: bool = False, **kwargs
    ) -> nx.Graph:
        """This component prepares the subgraph from a graph

        Do no add new model layers

        Parameters
        ----------
        edge_query : str
            Conditions to query the subgraph from graph. Applies on existing attributes of edges, so make sure to add the attributes first.
        """

        if edge_query is None:
            self.logger.debug(f"Use graph for subgraph")
            self._subgraphmodel = self._graphmodel

        self.logger.debug(f"query graph based on edge_query {edge_query}")
        SG = graph.query_graph_edges_attributes(
            self._graphmodel, id_col=self._graphmodel_edgeidcol, edge_query=edge_query
        )
        self.logger.debug(
            f"{len(SG.edges())}/{len(self._graphmodel.edges())} edges are selected"
        )

        self._subgraphmodel = SG

        # visualise
        if visualise == True:
            f1,f2 = graph.plot_graph(self._subgraphmodel)
            f1[1].set_title("subgraphmodel")



    def setup_partition(self, algorithm: str = "simple", visualise: bool = False, **kwargs):
        """This component prepares the partition from subgraph

        in progress

        Parameters
        ----------
        algorithm : str
            Algorithm to derive partitions from subgraph. Options: ['louvain', 'simple']
        """

        # algorithm louvain
        import community
        import matplotlib.pyplot as plt
        import numpy as np

        G = self._graphmodel

        SG = self._subgraphmodel
        UG = SG.to_undirected()

        DIFF = G.copy()
        DIFF.remove_nodes_from(n for n in G if n in SG)

        # UG
        # further partition
        partition = {n: -1 for n in UG.nodes}
        if algorithm == 'simple': # based on connected subgraphs
            for i,ig in enumerate(nx.connected_components(UG)):
                partition.update({n:i for n in ig})
        elif algorithm == 'louvain': # based on louvain algorithm
            partition.update(community.best_partition(UG))
        else:
            raise ValueError(f'{algorithm} is not recognised. allowed algorithms: simple, louvain')

        # finish partition
        n_partitions = max(partition.values())
        self.logger.debug(f'{n_partitions} partitions is derived from subgraph using {algorithm} algorithm')
        # edges_partitions = {(s, t): partition[s] for s, t in G.edges() if partition[s] == partition[t]}

        # induced graph from the partitions
        ind = community.induced_graph(partition, UG)
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
            plt.title(f'partitions')
            plt.axis("off")
            nx.draw_networkx_nodes(
                SG,
                pos = pos_SG,
                node_size=30,
                cmap=plt.cm.RdYlBu,
                node_color=list(partition.values()),
            )
            nx.draw_networkx_edges(SG, pos_SG, alpha=0.3)
            # add DIFF
            nx.draw_networkx_nodes(DIFF, pos=pos_DIFF, node_size=10, node_color = 'k')
            nx.draw_networkx_edges(DIFF, pos=pos_DIFF, edge_color='k', arrows = False)

            # induced partition graph
            plt.figure(figsize=(8, 8))
            plt.title(f'Induced graph from partitions')
            plt.axis("off")
            nx.draw_networkx_nodes(
                ind,
                pos = pos_ind,
                node_size=30, cmap=plt.cm.RdYlBu, node_color=list(ind)
            )
            nx.draw_networkx_edges(ind, pos_ind, alpha=0.3)
            # add DIFF
            nx.draw_networkx_nodes(DIFF, pos=pos_DIFF, node_size=10, node_color = 'k')
            nx.draw_networkx_edges(DIFF, pos=pos_DIFF, edge_color='k', arrows = False)


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
                fig.savefig(pp, format='pdf')
            pp.close()
        multipage(join(self.root, "graph", 'graphmodel.pdf'))



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
