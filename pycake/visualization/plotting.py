#/usr/bin/env python
# -*- coding: utf-8 -*-

"""Collection of matplotlib customizations for visualization."""

import matplotlib


def latex_mode(on=True):
    """Enable LaTeX for labels"""
    matplotlib.rc('text', usetex=on)


class ScaledFormatter(matplotlib.ticker.ScalarFormatter):
    """Formats tick labels scaled by *dx* and shifted by *x0*."""
    def __init__(self, dx=1.0, x0=0.0, **kwargs):
        self.dx, self.x0 = dx, x0

    def rescale(self, x):
        return x / self.dx + self.x0

    def __call__(self, x, pos=None):
        xmin, xmax = self.axis.get_view_interval()
        xmin, xmax = self.rescale(xmin), self.rescale(xmax)
        d = abs(xmax - xmin)
        x = self.rescale(x)
        s = self.pprint_val(x, d)
        return s
