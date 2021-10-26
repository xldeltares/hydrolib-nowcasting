# -*- coding: utf-8 -*-

import logging
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


__all__ = []

# IO
def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()