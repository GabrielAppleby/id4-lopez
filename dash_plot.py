from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, ALL, State


app = Dash(__name__)

df = pd.read_csv('transformed_data.csv', index_col=0)
PARAM_START: int = 2
PARAM_STOP: int = 101
PARAM_STEP: int = 5
PARAMS: List[int] = list(range(PARAM_START, PARAM_STOP + PARAM_STEP, PARAM_STEP))
COLUMNS: List[str] = ['umap', 'verticalExcitationEnergy', 'homo', 'lumo', 'redoxPotentialS0',
                      'redoxPotentialS1', 'redoxPotentialT1', 'dipoleMomentS0', 'dipoleMomentS1',
                      'dipoleMomentT1', 'zero0s1', 'zero0t1', 'ionizationPotential', 'S', 'C', 'Cl', 'F', 'B',
                      'O', 'N']

histograms: List[dcc.Graph] = [dcc.Graph(id={'type': 'histogram', 'index': col}, figure=px.histogram(df, x=col),
                                         config={'displayModeBar': False}) for col in COLUMNS[1:]]

app.layout = html.Div(id='app-entry', className='flex column', children=[
    html.Div(id='controls', className='flex', children=[
        html.Div(id='scatter_color_selector_div', children=[
            html.H4(children='Color:'),
            dcc.Dropdown(id='scatter_color_selector',
                         options=[{'label': name, 'value': name} for name in COLUMNS[1:]],
                         multi=False,
                         clearable=False,
                         value=COLUMNS[-1])],
                 className='dropdown'),
        html.Div(id='scatter_x_axis_selector_div', children=[
            html.H4(children='X - Axis:'),
            dcc.Dropdown(id='x_axis_selector',
                         options=[{'label': name, 'value': name} for name in COLUMNS],
                         multi=False,
                         clearable=False,
                         value=COLUMNS[0])],
                 className='dropdown'),
        html.Div(id='scatter_y_axis_selector_div', children=[
            html.H4(children='Y - Axis:'),
            dcc.Dropdown(id='y_axis_selector',
                         options=[{'label': name, 'value': name} for name in COLUMNS],
                         multi=False,
                         clearable=False,
                         value=COLUMNS[0])],
                 className='dropdown'),

        html.Div(id='num_neighbors_slider_div', children=[
            html.H4(children='Number of neighbors:'),
            dcc.Slider(id='num_neighbors_slider', className='dcc-slider-cludge', min=PARAMS[0], max=PARAMS[-1], step=PARAM_STEP,
                       value=PARAMS[4])],
                 className='slider'),
    ]),
    html.Div(id='plots', className='flex', children=[
        html.Div(id='projection', children=[
            dcc.Graph(
                id='projection_scatter',
                config={'modeBarButtonsToRemove': ['select', 'lasso']}
            ),
            dcc.Tooltip(id="projection_tooltip")
        ]),
        html.Div(id='histograms', children=histograms)
    ]),
])


def get_figure(df: pd.DataFrame, x_col: str, y_col: str, color: str, selected_points: List[int]):
    fig = px.scatter(df, x=x_col, y=y_col, color=color,
                     hover_data={name: False for name in df.columns})
    fig.update_traces(selectedpoints=selected_points,
                      customdata=df.index,
                      mode='markers',
                      unselected={'marker': {'color': 'grey'}})

    return fig


def get_hist(inner_df, x_col, selectedpoints, selectedpoints_local):
    inner_df['selected'] = inner_df.index.isin(selectedpoints)
    fig = px.histogram(inner_df, x=x_col, color='selected', color_discrete_map={True: '#636efa', False: '#808080'})
    if selectedpoints_local and "range" in selectedpoints_local:
        ranges = selectedpoints_local["range"]
        selection_bounds = {
            "x0": ranges["x"][0],
            "x1": ranges["x"][1],
            "y0": ranges["y"][0],
            "y1": ranges["y"][1],
        }
        xref = None
        yref = None
        fig.add_shape(dict({'type': 'rect',
                            'line': {'width': 1, 'dash': 'dot', 'color': 'darkgrey'}},
                           xref=xref,
                           yref=yref,
                           **selection_bounds))

    fig.update_traces(customdata=inner_df.index)
    fig.update_layout(dragmode='select', hovermode=False)

    return fig


@app.callback(
    Output(component_id='projection_scatter', component_property='figure'),
    Input(component_id='scatter_color_selector', component_property='value'),
    Input(component_id='x_axis_selector', component_property='value'),
    Input(component_id='y_axis_selector', component_property='value'),
    Input(component_id='num_neighbors_slider', component_property='value'),
    Input({'type': 'histogram', 'index': ALL}, 'selectedData'))
def callback(color, x_axis, y_axis, neighbor, selection):
    neighbor = str(neighbor)
    selected_points = df.index
    for sd in selection:
        current_hist_points = []
        if sd is not None and sd['points'] is not None:
            for point_collection in sd['points']:
                if point_collection['customdata'] is not None and len(point_collection['customdata']) > 0:
                    current_hist_points.extend(point_collection['customdata'])
        if len(current_hist_points) > 0:
            selected_points = np.intersect1d(selected_points, np.array(current_hist_points))
    if x_axis == 'umap':
        x_axis = 'x_' + neighbor
    if y_axis == 'umap':
        y_axis = 'y_' + neighbor
    return get_figure(df, x_axis, y_axis, color, selected_points)


@app.callback(
    Output({'type': 'histogram', 'index': ALL}, component_property='figure'),
    Input({'type': 'histogram', 'index': ALL}, 'selectedData'),
    State({'type': 'histogram', 'index': ALL}, 'id'))
def callback(selection, columns):
    selected_points = df.index
    for sd in selection:
        current_hist_points = []
        if sd is not None and sd['points'] is not None:
            for point_collection in sd['points']:
                if point_collection['customdata'] is not None and len(point_collection['customdata']) > 0:
                    current_hist_points.extend(point_collection['customdata'])
        if len(current_hist_points) > 0:
            selected_points = np.intersect1d(selected_points, np.array(current_hist_points))
    hists = []
    for index, local_selection in zip([col['index'] for col in columns], selection):
        hists.append(get_hist(df, index, selected_points, local_selection))
    return hists


@app.callback(
    Output("projection_tooltip", "show"),
    Output("projection_tooltip", "bbox"),
    Output("projection_tooltip", "children"),
    Input("projection_scatter", "clickData"),
    Input('projection', 'n_clicks'),
)
def display_hover(hoverData, n_clicks):
    if hoverData is None:
        return False, None, None
    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    df_row = df.iloc[num]
    img_src = df_row['img']
    link = df_row['MolView_url']
    children = [
        html.Div(
            children=[
                html.Img(src=img_src, style={"width": "100%"}),
            ], style={'width': '200px', 'white-space': 'normal'})
    ]
    return True, bbox, children


@app.callback(Output("projection_scatter", "clickData"),
              State("projection_scatter", "clickData"),
              Input('projection', 'n_clicks'))
def reset_clickData(clickData, n_clicks):
    if n_clicks is not None and n_clicks % 2 == 0:
        return None
    return clickData


if __name__ == '__main__':
    app.run_server(debug=True)
