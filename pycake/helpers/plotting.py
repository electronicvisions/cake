#!/usr/bin/env python

import datetime
import numpy as np
from collections import defaultdict

import bokeh.plotting
import bokeh.layouts
import bokeh.embed
import bokeh.resources
import bokeh.models
import bokeh.models.glyphs

import matplotlib.cm
import matplotlib.colors

import Coordinate as C

def get_cmap_dict(max_value, cmap_name, missing_color='black'):
    cmap_mpl = matplotlib.cm.get_cmap(cmap_name)
    cmap = [cmap_mpl(int(ii*(1.*cmap_mpl.N/(max_value+1)))) for ii in range(max_value+1)]
    cmap_hex = [matplotlib.colors.rgb2hex(rgba[:3]) for rgba in cmap_mpl]
    cmap_dict = {ii: color for ii, color in enumerate(cmap_hex)}
    cmap_dict[-1] = missing_color
    return cmap_dict

def get_hicanns_from_dnc(dnc):
    per_dnc = C.HICANNOnDNC.enum_type.size
    offset = (per_dnc // 2)
    dnc = C.DNCOnWafer(C.Enum(dnc))
    h0 = C.HICANNOnDNC(C.Enum(0)).toHICANNOnWafer(dnc).toEnum().value()
    h1 = C.HICANNOnDNC(C.Enum(per_dnc // 2)).toHICANNOnWafer(dnc).toEnum().value()
    return np.array([range(h0, h0+offset) , range(h1, h1+offset)])

def get_bokeh_figure(title, value_by_hicann_enum, max_value = None, cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green", "red"]), add_text=False, default_fill_color='black'):
    """
    value_by_hicann_enum: dict, hicann enum key
    """
    hicanns_per_reticle = 8
    rows = 2
    ret_width = hicanns_per_reticle // rows
    hc_height = ret_width // rows

    dnc2reticle_map = np.array([
    [-1, -1, -1,  0,  1,  2, -1, -1, -1],
    [-1, -1,  3,  4,  5,  6,  7, -1, -1],
    [-1,  8,  9, 10, 11, 12, 13, 14, -1],
    [15, 16, 17, 18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29, 30, 31, 32],
    [-1, 33, 34, 35, 36, 37, 38, 39, -1],
    [-1, -1, 40, 41, 42, 43, 44, -1, -1],
    [-1, -1, -1, 45, 46, 47, -1, -1, -1] ])

    dncs = [dnc for row in dnc2reticle_map for dnc in row if dnc != -1]
    hicann_array = np.array(map(get_hicanns_from_dnc, dncs))
    ret_coords = [(ret_width*j,ret_width*i) for i,row in enumerate(dnc2reticle_map)
                                            for j,dnc in enumerate(row) if dnc != -1]
    hc_dict =  {hicann_array[i,j,k] : (x+pos, -(y+hc_height*row))
                for i, (x,y) in enumerate(ret_coords)
                for j,row in enumerate(range(rows))
                for k,pos in enumerate(range(ret_width))}

    plot_data = defaultdict(list)

    for hicann_enum, values in hc_dict.items():
        plot_data["HICANN"].append(hicann_enum)
        plot_data["x"].append(values[0])
        plot_data["y"].append(values[1])
        plot_data["DNC"].append(C.HICANNOnWafer(C.Enum(hicann_enum)).toDNCOnWafer().toEnum().value())
        plot_data["FPGA"].append(C.HICANNOnWafer(C.Enum(hicann_enum)).toFPGAOnWafer().value())
        if hicann_enum in value_by_hicann_enum:
            plot_data["Value"].append(value_by_hicann_enum[hicann_enum])

            if max_value:
                value = value_by_hicann_enum[hicann_enum]/float(max_value)
            else:
                value = value_by_hicann_enum[hicann_enum]

            plot_data["fill_color"].append(matplotlib.colors.rgb2hex(cmap(value)[:3]))
        else:
            plot_data["fill_color"].append(default_fill_color)
            plot_data["Value"].append(None)

    tooltip_keys = plot_data.keys()
    for key in ['x', 'y', 'fill_color']:
        tooltip_keys.remove(key)
    tooltips = [(key, "@{}".format(key)) for key in sorted(tooltip_keys)]
    hover = bokeh.models.HoverTool(tooltips=tooltips)

    bokeh_figure = bokeh.plotting.figure(title=title, plot_width=800, plot_height=800, tools=[hover])
    bokeh_figure.xgrid.grid_line_color = None
    bokeh_figure.ygrid.grid_line_color = None
    bokeh_figure.axis.visible = False
    bokeh_figure.background_fill_alpha = 0.
    bokeh_figure.border_fill_alpha = 0.

    bokeh_figure.rect('x', 'y', height=hc_height, width=1, fill_color='fill_color',
                      source=bokeh.plotting.ColumnDataSource(data=plot_data), line_color='black')

    if add_text:
        text_source = dict(x=np.add(plot_data['x'], 0.5), y=plot_data['y'],
                           text=plot_data["Value"])
        source = bokeh.plotting.ColumnDataSource(text_source)
        glyph = bokeh.models.glyphs.Text(x="x", y="y", text="text", angle=1.57, text_color="#000000", text_align='center')
        bokeh_figure.add_glyph(source, glyph)

    return bokeh_figure

def store_bokeh(name, bokeh_figures, filename):

    layouted = bokeh.layouts.layout(bokeh_figures)
    html = bokeh.embed.file_html(layouted, bokeh.resources.INLINE, name + " {0}".format(datetime.datetime.now()))

    with open(filename, 'wb') as html_file:
        html_file.write(html)
