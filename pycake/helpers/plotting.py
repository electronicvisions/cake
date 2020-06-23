#!/usr/bin/env python

import datetime
import numpy as np
import pandas as pd
from collections import defaultdict

import bokeh.plotting
import bokeh.embed
import bokeh.resources
import bokeh.models
import bokeh.models.glyphs
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models.widgets import DataTable, TableColumn, HTMLTemplateFormatter, Button
from bokeh.layouts import grid, row
from bokeh.io import output_file, save

import matplotlib.cm
import matplotlib.colors

from pyhalco_common import Enum
import pyhalco_hicann_v2 as C

def generate_bokeh_table(table, column_names):
    template= """
    <div style="background: white
        color: black">
    <%= value %></div>
    """
    formatter = HTMLTemplateFormatter(template=template)
    source = ColumnDataSource(data=table)
    columns = [TableColumn(field=column_name, title=column_name,
                           formatter=formatter) for column_name in column_names]
    blacklist_table = DataTable(
        source=source, columns=columns, sizing_mode="stretch_both")

    savebutton = Button(label="Save", button_type="success", width=200)
    savebutton.callback = CustomJS(
        args=dict(source_data=source),
        code="""
            const columns = Object.keys(source_data.data);
            const nrows = source_data.get_length();
            const lines = [columns.join(',')];
            for (let i = 0; i < nrows; i++) {
                let row = [];
                for (let j = 0; j < columns.length; j++) {
                    const column = columns[j]
                    row.push(source_data.data[column][i].toString())
                }
                lines.push(row.join(','))
            }
            var file = new Blob([lines.join('\\n').concat('\\n')], {type: 'text/plain'});
            var elem = window.document.createElement('a');
            elem.href = window.URL.createObjectURL(file);
            elem.download = 'selected-data.txt';
            document.body.appendChild(elem);
            elem.click();
            document.body.removeChild(elem);
            """,
    )

    colorbutton = Button(label="color", width=200)
    colorbutton.callback = CustomJS(
        args=dict(source=blacklist_table),
        code="""
             var nocolor = '<div style="background: white'+
                           'color: black">'+
                           '<%= value %></div>'
             var color = '<div style="background:<%='+
                         '(function colorfromint(){'+
                         'if(value == -1){'+
                         'return("blue")}'+
                         'else if(value > 0){'+
                         'return("red")}'+
                         'else {return("white")}'+
                         '}()) %>;'+
                         'color: <%='+
                         '(function colorfromint(){'+
                         'if(value == -1){'+
                         'return("blue")}'+
                         '}()) %>;">'+
                         '<%= value %></div>'
             if (source.columns[0].formatter.attributes.template == color){
             source.columns[0].formatter.attributes.template=nocolor
             } else {
             source.columns[0].formatter.attributes.template=color
             }
             source.change.emit();
             """,
    )

    buttons = row(savebutton, colorbutton)
    layout = grid([buttons, blacklist_table])
    return layout

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
    dnc = C.DNCOnWafer(Enum(dnc))
    h0 = C.HICANNOnDNC(Enum(0)).toHICANNOnWafer(dnc).toEnum().value()
    h1 = C.HICANNOnDNC(Enum(per_dnc // 2)).toHICANNOnWafer(dnc).toEnum().value()
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
        plot_data["DNC"].append(C.HICANNOnWafer(Enum(hicann_enum)).toDNCOnWafer().toEnum().value())
        plot_data["FPGA"].append(C.HICANNOnWafer(Enum(hicann_enum)).toFPGAOnWafer().value())
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

def get_bokeh_histogram(title, hist, edges):
    hist_df = pd.DataFrame({"value": hist,
                            "normal": hist,
                            "type": "normal",
                            "left": edges[:-1],
                            "right": edges[1:]})
    hist_df["log"] = np.log(hist_df["value"], where=(hist_df["value"] != 0))
    hist_df["interval"] = ["%d to %d" % (left, right) for left,
                           right in zip(hist_df["left"], hist_df["right"])]
    source = ColumnDataSource(hist_df)

    p = bokeh.plotting.figure(
        title=title, y_axis_label="# of HICANNs", tools='box_zoom, pan, wheel_zoom, undo, redo, reset, save', background_fill_color="#fafafa")
    p.quad(top="value", bottom=0, left="left", right="right", source=source,
           fill_color="SteelBlue", line_color="white", alpha=0.5, hover_fill_alpha=1.0, hover_fill_color="Tan")
    p.y_range.start = 0
    p.xaxis.axis_label = '# blacklisted components'
    p.grid.grid_line_color = "white"
    hover = bokeh.models.HoverTool(tooltips=[("# blacklisted {}".format(
        title), '@interval'), ('# HICANNs', "@normal")])
    p.add_tools(hover)

    logbutton = Button(label="LOG", button_type="success", width=200)
    callback = CustomJS(
        args=dict(source=source, label=p.yaxis[0]),
        code="""
             if (source.data["type"][0] == "normal"){
             source.data["value"] = source.data["log"];
             source.data["type"][0] = "log";
             label.axis_label = "Log(# of HICANNs)"
             } else {
             source.data["value"] = source.data["normal"];
             source.data["type"][0] = "normal";
             label.axis_label = "# of HICANNs"
             }
             source.change.emit();
             label.trigger("change");
             """,
    )
    logbutton.js_on_click(callback)

    layout = grid([logbutton, p])
    return layout

def store_bokeh(name, bokeh_figure, filename):
    output_file(filename)
    save(bokeh_figure, title=name + " {0}".format(datetime.datetime.now()))
