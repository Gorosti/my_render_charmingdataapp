import base64
import io
import dash
from dash.dependencies import Input, Output, State
import dash_mantine_components as dmc
from dash import dcc, html, dash_table
from dash.dash_table.Format import Format, Scheme, Symbol
import plotly.graph_objs as go
from collections import OrderedDict
import pandas as pd
import numpy as np
import math
import gc

# Import the AgeDating Class
from MEM import MEM

# #-----------------------------------------------#
# #            Initialize the Dash App            #
# #-----------------------------------------------#  

app = dash.Dash(__name__)
server = app.server

# #-----------------------------------------------#
# #               Define the Layout               #
# #-----------------------------------------------#  

app.layout = html.Div(
    children=[
        dmc.Grid(
            align='stretch',
            gutter=0,
            children=[
                dmc.Col(
                    span=12,
                    xl=6,
                    children=[
                        dmc.Col(
                            span=12,
                            xl=12,
                            children=[
                                dmc.Paper(
                                    p=8,
                                    m=4,
                                    withBorder=True,
                                    children=[
                                        html.Div('Data Selection'),
                                        dmc.Grid(
                                            align='stretch',
                                            gutter='xs',
                                            children=[
                                                dmc.Col(
                                                    span=12,
                                                    xl=11,
                                                    children=[
                                                        html.Div(
                                                            children=[
                                                                # Upload Button for the data file ###
                                                                dcc.Upload(
                                                                    id='upload-data',
                                                                    children=html.Button('Select Files')
                                                                ),
                                                                html.Div(id='upload-text-output'),
                                                                html.Br(),
                                                                # Checklist for selecting the sheet names ###
                                                                dcc.Checklist(id='sheet-checklist'),
                                                                html.Br(),
                                                                # Material selection ###
                                                                dcc.Dropdown(
                                                                    id='material-dropdown',
                                                                    style=dict(display='none'),
                                                                    options={
                                                                        'F': 'Feldspar',
                                                                        'Q': 'Quartz'
                                                                    },
                                                                    value='F'
                                                                ),
                                                                html.Br(),
                                                                # Table menu ###
                                                                dash_table.DataTable(
                                                                    id='option-menu',
                                                                    editable=True
                                                                ),
                                                            ],
                                                            style={
                                                                'padding': 10,
                                                                'flex': 1,
                                                                "width": "100%"
                                                            }
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        dmc.Col(
                            span=12,
                            xl=12,
                            children=[
                                dmc.Paper(
                                    p=8,
                                    m=4,
                                    withBorder=True,
                                    children=[
                                        html.Div('Fitted Profiles'),
                                        dmc.Grid(
                                            align='stretch',
                                            gutter='xs',
                                            children=[
                                                dmc.Col(
                                                    span=12,
                                                    xl=6,
                                                    children=[
                                                        html.Div(
                                                            children=[
                                                                dmc.Col(
                                                                    span=12,
                                                                    xl=6,
                                                                    children=[
                                                                        dcc.Loading(
                                                                            id="loading-1",
                                                                            children=[
                                                                                dcc.Graph(
                                                                                    id='fit-graph',
                                                                                    style=dict(display='none')
                                                                                )
                                                                            ],
                                                                            type="circle"
                                                                        ),
                                                                    ]
                                                                ),
                                                            ],
                                                            style={
                                                                'padding': 10,
                                                                'flex': 1,
                                                                "width": "100%"
                                                            }
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
                dmc.Col(
                    span=12,
                    xl=6,
                    children=[
                        dmc.Col(
                            span=12,
                            xl=12,
                            children=[
                                dmc.Paper(
                                    p=8,
                                    m=4,
                                    withBorder=True,
                                    children=[
                                        html.Div('Initial Guess'),
                                        dmc.Grid(
                                            align='stretch',
                                            gutter='xs',
                                            children=[
                                                dmc.Col(
                                                    span=12,
                                                    xl=6,
                                                    children=[
                                                        html.Div(
                                                            children=[
                                                                html.Div(
                                                                    id='radio-container',
                                                                    children=[
                                                                        dcc.RadioItems(
                                                                            id='radio-guess',
                                                                            inline=True
                                                                        ),
                                                                        html.Br(),
                                                                        html.Div(
                                                                            children=[
                                                                                html.Div(
                                                                                    id='order-container',
                                                                                    children=[
                                                                                        html.I("Order of the Model"),
                                                                                        html.Br(),
                                                                                        dcc.Input(
                                                                                            id="input-order",
                                                                                            type="number",
                                                                                            value=2.1,
                                                                                            min=1,
                                                                                            max=10,
                                                                                            placeholder="",
                                                                                            style={'marginRight': '10px'}
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                                html.Div(
                                                                                    id='mu-container',
                                                                                    children=[
                                                                                        html.I("Mu"),
                                                                                        html.Br(),
                                                                                        dcc.Input(
                                                                                            id="input-mu",
                                                                                            type="number",
                                                                                            value=1.38,
                                                                                            min=1e-6,
                                                                                            max=1e6,
                                                                                            placeholder="",
                                                                                            style={'marginRight': '10px'}
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ],
                                                                            style={
                                                                                'display': 'flex',
                                                                                'flex-direction': 'row'
                                                                            }
                                                                        ),
                                                                    ],
                                                                    style={'display': 'none'}
                                                                ),
                                                                html.Div(
                                                                    children=[
                                                                        html.Div(
                                                                            id='k1_slider-container',
                                                                            children=[
                                                                                dcc.Slider(
                                                                                    id='k1_slider',
                                                                                    marks={
                                                                                        0: {'label': '0'},
                                                                                        0.99: {'label': 'K1'}
                                                                                    },
                                                                                    tooltip={"placement": "right"},
                                                                                    min=0,
                                                                                    max=0.99,
                                                                                    value=0.5,
                                                                                    included=False,
                                                                                    vertical=True
                                                                                ),
                                                                            ],
                                                                            style={'display': 'none'}),
                                                                        html.Div(
                                                                            id='k2_slider-container',
                                                                            children=[
                                                                                dcc.Slider(
                                                                                    id='k2_slider',
                                                                                    marks={
                                                                                        0: {'label': '0'},
                                                                                        0.99: {'label': 'K2'}
                                                                                    },
                                                                                    tooltip={"placement": "right"},
                                                                                    min=0,
                                                                                    max=0.99,
                                                                                    value=0.1,
                                                                                    included=False,
                                                                                    vertical=True
                                                                                ),
                                                                            ],
                                                                            style={'display': 'none'}
                                                                        ),
                                                                        dcc.Graph(
                                                                            id='guess-graph',
                                                                            style=dict(display='none')
                                                                        ),
                                                                    ],
                                                                    style={'display': 'flex', 'flex-direction': 'row'}
                                                                ),
                                                                # Create Div to place a conditionally visible element inside
                                                                html.Div(
                                                                    id='x1_slider-container',
                                                                    children=[
                                                                        dcc.Slider(
                                                                            id='x1_slider',
                                                                            min=0,
                                                                            max=100,
                                                                            value=65,
                                                                            tooltip={"placement": "right"},
                                                                            included=False
                                                                        ),
                                                                    ],
                                                                    style={'display': 'none'}
                                                                ),
                                                                html.Div(
                                                                    id='x2_slider-container',
                                                                    children=[
                                                                        dcc.Slider(
                                                                            id='x2_slider',
                                                                            min=0,
                                                                            max=100,
                                                                            value=0,
                                                                            tooltip={"placement": "right"},
                                                                            included=False
                                                                        ),
                                                                    ],
                                                                    style={'display': 'none'}
                                                                ),
                                                                html.Div(
                                                                    id='x3_slider-container',
                                                                    children=[
                                                                        dcc.Slider(
                                                                            id='x3_slider',
                                                                            min=0,
                                                                            max=100,
                                                                            value=0,
                                                                            tooltip={"placement": "right"},
                                                                            included=False
                                                                        ),
                                                                    ],
                                                                    style={'display': 'none'}
                                                                ),
                                                                html.Div(
                                                                    id='table-container',
                                                                    children=[
                                                                        dash_table.DataTable(
                                                                            id='guess-modified',
                                                                            data=None,
                                                                            columns=None
                                                                        ),
                                                                    ],
                                                                    style={'display': 'none'}
                                                                ),
                                                            ],
                                                            style={
                                                                'padding': 10,
                                                                'flex': 1,
                                                                "width": "100%",
                                                                'backgroundColor': 'white'
                                                            }
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        dmc.Col(
                            span=12,
                            xl=12,
                            children=[
                                dmc.Paper(
                                    p=8,
                                    m=4,
                                    withBorder=True,
                                    children=[
                                        html.Div('ERC Graph'),
                                        dmc.Grid(
                                            align='stretch',
                                            gutter='xs',
                                            children=[
                                                dmc.Col(
                                                    span=12,
                                                    xl=6,
                                                    children=[
                                                        html.Div(
                                                            children=[
                                                                dmc.Col(
                                                                    span=12,
                                                                    xl=6,
                                                                    children=[
                                                                        dcc.Loading(
                                                                            id="loading-1",
                                                                            children=[
                                                                                dcc.Graph(
                                                                                    id='ERC-graph',
                                                                                    style=dict(display='none')
                                                                                ),
                                                                            ],
                                                                            type="circle"
                                                                        ),
                                                                    ]
                                                                ),
                                                            ],
                                                            style={
                                                                'padding': 10,
                                                                'flex': 1,
                                                                "width": "100%",
                                                                'backgroundColor': 'white'
                                                            }
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        ),
    ],
)


# #-----------------------------------------------#
# #              Define the Callbacks             #
# #-----------------------------------------------#  


@app.callback(
    Output('upload-text-output', 'children'),
    Output('sheet-checklist', 'options'),
    Output('sheet-checklist', 'value'),
    Output('material-dropdown', 'style'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
)
def read_sheet_names(contents, filename):
    """
    filename: name of the chosen file 
    content: content of that file
    """

    sheet_names = []
    return_string = ''
    style = dict(display='none')

    if contents is not None:

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        try:
            if 'xlsx' in filename:
                # Assume an excel file has bee chosen
                df_dict = pd.ExcelFile(io.BytesIO(decoded))
                sheet_names = df_dict.sheet_names
                return_string = ''
                style = dict()

            else:
                return_string = 'Not an excel file!!'
        except:
            return_string = 'Not able to read it!!'

    return return_string, sheet_names, sheet_names, style


@app.callback(
    Output('option-menu', 'data'),
    Output('option-menu', 'columns'),
    Output('option-menu', 'dropdown'),
    Output('radio-guess', 'options'),
    Output('radio-guess', 'value'),
    # Output('accept-menu-button', 'style'),
    Input('sheet-checklist', 'value')
)
def create_menu_table(selected_sheets):
    # Define menu options
    side_options = ['Front', 'End', 'Front', 'Front', 'Front']
    model_options = ['Exposure',
                     'Exp + Burial',
                     'Exp + Burial + Exp',
                     'Exp + Burial + Exp + Burial',
                     'Exp + Burial + Exp + Burial + Exp']

    df_options = pd.DataFrame(OrderedDict([
        ('side_options', side_options),
        ('model_options', model_options)
    ]))

    # We create the initial data for each column
    model = [model_options[0]] * len(selected_sheets)
    side = [side_options[0]] * len(selected_sheets)
    thickness = [50] * len(selected_sheets)

    known_t1 = ['Unknown'] * len(selected_sheets)
    known_t2 = known_t1
    known_t3 = known_t1

    df = pd.DataFrame(OrderedDict([
        ('Sample', selected_sheets),
        ('Model', model),
        ('Side', side),
        ('Thickness', thickness),
        ('Known t1', known_t1),
        ('Known t2', known_t2),
        ('Known t3', known_t3)
    ]))

    # Define the drop down menu

    dropdown = {
        'Model': {
            'options': [
                {'label': i, 'value': i}
                for i in df_options['model_options'].unique()
            ]
        },
        'Side': {
            'options': [
                {'label': i, 'value': i}
                for i in df_options['side_options'].unique()
            ]
        }
    }

    # Lets stick to the nomenclature expected by the dcc.DataTable
    data = df.to_dict('records')

    # columns = [{'name': col, 'id': col} for col in df.columns]
    columns = [
        {
            'id': 'Sample',
            'name': 'Sample',
            'type': 'text'
        }, {
            'id': 'Model',
            'name': 'Model',
            'presentation': 'dropdown'
        }, {
            'id': 'Side',
            'name': 'Side',
            'presentation': 'dropdown'
        }, {
            'id': 'Thickness',
            'name': 'Thickness',
            'type': 'numeric',
            'format': Format(
                precision=1,
                scheme=Scheme.fixed,
                symbol=Symbol.yes,
                symbol_suffix=' [mm]'),
        }, {
            'id': 'Known t1',
            'name': 'Known t1',
            'type': 'numeric',
            'format': Format(
                precision=3,
                scheme=Scheme.fixed,
                symbol=Symbol.yes,
                symbol_suffix=' days'),
        }, {
            'id': 'Known t2',
            'name': 'Known t2',
            'type': 'numeric',
            'format': Format(
                precision=3,
                scheme=Scheme.fixed,
                symbol=Symbol.yes,
                symbol_suffix=' days'),
        }, {
            'id': 'Known t3',
            'name': 'Known t3',
            'type': 'numeric',
            'format': Format(
                precision=3,
                scheme=Scheme.fixed,
                symbol=Symbol.yes,
                symbol_suffix=' days'),
        }
    ]

    radio_options = ['All'] + selected_sheets
    radio_value = 'All'

    if len(selected_sheets) == 0:
        data = None
        columns = None
        dropdown = None
        radio_options = ['a', 'b']
        radio_value = 'a'

    return data, columns, dropdown, radio_options, radio_value


@app.callback(
    Output('x1_slider', 'max'),
    Output('x2_slider', 'max'),
    Output('x3_slider', 'max'),
    Output('x1_slider', 'marks'),
    Output('x2_slider', 'marks'),
    Output('x3_slider', 'marks'),
    Output('radio-container', 'style'),
    Output('order-container', 'style'),
    # Input('accept-menu-button', 'n_clicks'),
    Input('option-menu', 'data'),
    Input('material-dropdown', 'value'),
    State('option-menu', 'columns'),
    State('upload-data', 'contents'),
    State('sheet-checklist', 'value'),

)
def create_guess_graph_frame(rows, material, columns, contents,
                             selected_sheets):
    if rows:

        # First we extract the menu choices ###

        # We build up the dataframe again
        df = pd.DataFrame(rows, columns=[c['name'] for c in columns])

        # Get the model
        model_idx = df['Model'].values

        # Convert the format of the model
        model_dict = {'Exposure': 1,
                      'Exp + Burial': 2,
                      'Exp + Burial + Exp': 3,
                      'Exp + Burial + Exp + Burial': 4,
                      'Exp + Burial + Exp + Burial + Exp': 5}

        model_idx = np.array([model_dict[letter] for letter in model_idx])

        # Also we Initialize the MEM object ###

        # Decode the full name
        content_type, content_string = contents.split(',')
        full_name = base64.b64decode(content_string)

        # Initialize
        FIT = MEM(full_name)

        # Read data
        weight = str('none')
        FIT.read_data_array(selected_sheets, weight)

        # Create initial guess
        initial_guess, df_data, df_columns = define_initial_guess(selected_sheets)

        residual = 0.01
        FIT.define_model(material, initial_guess, selected_sheets, model_idx, residual)
        FIT.multiple_fiting_array(selected_sheets, model_idx)

        xmax = max(FIT.xall[0]) + 0.5

        # We round it up 
        xmax = math.ceil(xmax)

        radio_style = {'display': 'block'}

        if material == 'Q':
            order_style = {'display': 'none'}
        else:
            order_style = {'display': 'block'}

    else:
        xmax = 1000
        radio_style = {'display': 'none'}
        order_style = {'display': 'block'}

    marks_xp1 = {
        0: {'label': '0'},
        xmax: {'label': 'xp1'}
    }
    marks_xp2 = {
        0: {'label': '0'},
        xmax: {'label': 'xp2'}
    }
    marks_xp3 = {
        0: {'label': '0'},
        xmax: {'label': 'xp3'}
    }
    return (xmax, xmax, xmax,
            marks_xp1, marks_xp2, marks_xp3,
            radio_style, order_style)


def define_initial_guess(sheet_names):
    # Initial guess values
    r = 2.1
    u = 1.38
    xp1 = 7
    K1 = 0.03
    xp2 = 30
    K2 = 1
    xp3 = 0.01

    guess = np.array([r] + [u, np.exp(xp1 * u), -np.log(1 - K1), np.exp(xp2 * u),
                            -np.log(1 - K1), np.exp(xp3 * u)] * len(sheet_names))

    params = [r, u, xp1, K1, xp2, K2, xp2]
    param_list = []
    for _ in sheet_names:
        param_list.append(params)

    df = pd.DataFrame(np.array(param_list),
                      index=sheet_names,
                      columns=['r', 'u', 'xp1', 'K1', 'xp2', 'K2', 'xp3'])

    df_data = df.to_dict('records')
    df_columns = [{"name": i, "id": i} for i in df.columns]

    return guess, df_data, df_columns


@app.callback(
    Output('x1_slider-container', 'style'),
    Output('x2_slider-container', 'style'),
    Output('x3_slider-container', 'style'),
    Output('k1_slider-container', 'style'),
    Output('k2_slider-container', 'style'),
    Output('mu-container', 'style'),
    Input('radio-guess', 'value'),
    Input('option-menu', 'data'),
    State('sheet-checklist', 'value'),
    State('option-menu', 'columns'),
)
def adapt_slider_visibility(sheet, menu_data, selected_sheets, menu_col):
    k1_style = {'display': 'none'}
    k2_style = {'display': 'none'}
    xp1_style = {'display': 'none'}
    xp2_style = {'display': 'none'}
    xp3_style = {'display': 'none'}
    mu_style = {'display': 'none'}

    if menu_data:

        if sheet in selected_sheets:

            df_menu = pd.DataFrame(menu_data,
                                   columns=[c['name'] for c in menu_col],
                                   index=selected_sheets)
            # Get the model
            model_idx = df_menu['Model'][sheet]
            # Convert the format
            model_dict = {'Exposure': 1,
                          'Exp + Burial': 2,
                          'Exp + Burial + Exp': 3,
                          'Exp + Burial + Exp + Burial': 4,
                          'Exp + Burial + Exp + Burial + Exp': 5}

            model_idx = model_dict.get(model_idx)

            mu_style = {'display': 'block'}

            if model_idx > 0:
                # Exp + Bur or more
                xp1_style = {
                    'display': 'block',
                    'height': '50px',
                    'width': '500px'
                }
            if model_idx > 1:
                # Exp + Bur or more
                k1_style = {'display': 'block',
                            'height': '500px',
                            'width': '60px'}

            if model_idx > 2:
                # Exp + Bur or more
                xp2_style = {'display': 'block',
                             'height': '50px',
                             'width': '500px'}

            if model_idx > 3:
                # Exp + Bur or more
                k2_style = {'display': 'block',
                            'height': '500px',
                            'width': '60px'}

            if model_idx > 4:
                # Exp + Bur or more
                xp3_style = {'display': 'block',
                             'height': '50px',
                             'width': '500px'}

    return xp1_style, xp2_style, xp3_style, k1_style, k2_style, mu_style


@app.callback(
    Output('guess-modified', 'data'),
    Output('guess-modified', 'columns'),
    Input('x1_slider', 'value'),
    Input('x2_slider', 'value'),
    Input('x3_slider', 'value'),
    Input('k1_slider', 'value'),
    Input('k2_slider', 'value'),
    Input('radio-guess', 'value'),
    Input('input-order', 'value'),
    Input('input-mu', 'value'),
    Input('option-menu', 'data'),
    State('guess-modified', 'data'),
    State('guess-modified', 'columns'),
    State('sheet-checklist', 'value'),
    State('sheet-checklist', 'options'),
)
def modify_guess(xp1, xp2, xp3, K1, K2, sheet, order, mu,
                 _, data, columns,
                 sheet_names, available_sheets):
    if available_sheets:

        if data is None:

            # Initial guess values
            r = 2.1
            u = 1.38
            xp1 = 7
            K1 = 0.03
            xp2 = 30
            K2 = 1
            xp3 = 0.01

            params = [r, u, xp1, K1, xp2, K2, xp3]
            param_list = []
            for _ in sheet_names:
                param_list.append(params)

            df = pd.DataFrame(np.array(param_list),
                              index=sheet_names,
                              columns=['r', 'u', 'xp1', 'K1', 'xp2', 'K2', 'xp3'])

        else:

            df = pd.DataFrame(data,
                              columns=[c['name'] for c in columns],
                              index=available_sheets)

            df['r'][available_sheets] = order
            df['u'][sheet] = mu
            df['xp1'][sheet] = xp1
            df['xp2'][sheet] = xp2
            df['xp3'][sheet] = xp3
            df['K1'][sheet] = K1
            df['K2'][sheet] = K2

        df_data = df.to_dict('records')
        df_columns = [{"name": i, "id": i} for i in df.columns]

    else:

        df_data = None
        df_columns = None

    return df_data, df_columns


@app.callback(
    Output('guess-graph', 'figure'),
    Output('guess-graph', 'style'),
    Input('guess-modified', 'data'),
    State('guess-modified', 'columns'),
    State('option-menu', 'data'),
    State('option-menu', 'columns'),
    State('upload-data', 'contents'),
    State('sheet-checklist', 'value'),
    State('sheet-checklist', 'options'),
    State('material-dropdown', 'value'),
    State('radio-guess', 'value'),
)
def create_guess_fig(rows_guess, columns_guess, rows_menu,
                     columns_menu,
                     contents, selected_sheets, available_sheets,
                     material, sheet):
    if rows_guess and rows_menu:

        # First we extract the menu choices ###
        # We build up the dataframe again
        df_guess = pd.DataFrame(rows_guess,
                                columns=[c['name'] for c in columns_guess],
                                index=available_sheets)
        df_guess = df_guess.loc[selected_sheets]
        df_menu = pd.DataFrame(rows_menu, columns=[c['name'] for c in columns_menu])
        # Get the model
        model_idx = df_menu['Model'].values
        # Convert the format
        model_dict = {'Exposure': 1,
                      'Exp + Burial': 2,
                      'Exp + Burial + Exp': 3,
                      'Exp + Burial + Exp + Burial': 4,
                      'Exp + Burial + Exp + Burial + Exp': 5}
        model_idx = np.array([model_dict[letter] for letter in model_idx])
        bla = f'Model idx: {model_idx}'

        # Also we Initialize the MEM object ###
        # Decode the full name
        content_type, content_string = contents.split(',')
        full_name = base64.b64decode(content_string)
        # Initialize
        FIT = MEM(full_name)
        # Read data
        weight = str('none')
        FIT.read_data_array(selected_sheets, weight)

        initial_guess = convert_params_to_guess(df_guess)

        residual = 0.01
        FIT.define_model(material, initial_guess, selected_sheets, model_idx, residual)
        FIT.multiple_fiting_array(selected_sheets, model_idx)

        plot_content = []
        fig = go.Figure()

        if sheet in selected_sheets:
            idx = selected_sheets.index(sheet)
            name = sheet
            plot_content.append(
                go.Scatter(
                    x=FIT.xall[idx],
                    y=FIT.yall[idx],
                    error_y=dict(
                        type='data',  # value of error bar given in data coordinates
                        array=FIT.errall[idx],
                        visible=True),
                    mode='markers',
                    marker_color=fig.layout['template']['layout']['colorway'][idx],
                    legendgroup=name,
                    legendgrouptitle_text=name,
                    name='Data'
                )
            )

            xmax = max(FIT.xall[idx]) + 0.5
            xi = np.linspace(0, xmax, 100)
            y_guess_temp = FIT.fun(xi, idx, model_idx[idx], *FIT.P0)

            plot_content.append(go.Scatter(x=xi,
                                           y=y_guess_temp,
                                           line=dict(color=fig.layout['template']['layout']['colorway'][idx]),
                                           legendgroup=name,
                                           name='Guess'))

        else:

            for idx, name in enumerate(selected_sheets):
                plot_content.append(go.Scatter(x=FIT.xall[idx],
                                               y=FIT.yall[idx],
                                               error_y=dict(
                                                   type='data',  # value of error bar given in data coordinates
                                                   array=FIT.errall[idx],
                                                   visible=True),
                                               mode='markers',
                                               marker_color=fig.layout['template']['layout']['colorway'][idx],
                                               legendgroup=name,
                                               legendgrouptitle_text=name,
                                               name='Data'))

                xmax = max(FIT.xall[idx]) + 0.5
                xi = np.linspace(0, xmax, 100)
                y_guess_temp = FIT.fun(xi, idx, model_idx[idx], *FIT.P0)

                plot_content.append(go.Scatter(x=xi,
                                               y=y_guess_temp,
                                               line=dict(color=fig.layout['template']['layout']['colorway'][idx]),
                                               legendgroup=name,
                                               name='Guess'))

        # # Get some values
        # i =0
        # xmax = max(FIT.xall[i])+0.5
        # xi = np.linspace(0,xmax,100)
        # y_guess_temp = FIT.fun(xi,i,model_idx[i],*FIT.P0)
        #     # plt.plot(xi, y_guess_temp, color = 'green', label='Guess')

        layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)',
                           title="Initial Guess",
                           height=500,
                           width=500)

        fig = go.Figure(data=plot_content,
                        layout=layout)

        graph_style = dict()

    else:

        fig = {}
        graph_style = dict(display='none')

    return fig, graph_style


def convert_params_to_guess(df_params):
    guess = []

    for index, row in df_params.iterrows():

        r = row['r']
        u = row['u']
        xp1 = row['xp1']
        K1 = row['K1']
        xp2 = row['xp2']
        K2 = row['K2']
        xp3 = row['xp3']

        if len(guess) == 0:
            guess.append(r)

        guess.append(u)
        guess.append(np.exp(xp1 * u))
        guess.append(-np.log(1 - K1))
        guess.append(np.exp(xp2 * u))
        guess.append(-np.log(1 - K2))
        guess.append(np.exp(xp3 * u))

    return np.array(guess)


@app.callback(
    Output('fit-graph', 'figure'),
    Output('fit-graph', 'style'),
    Output('ERC-graph', 'figure'),
    Output('ERC-graph', 'style'),
    Input('guess-modified', 'data'),
    Input('option-menu', 'data'),
    State('guess-modified', 'columns'),
    State('option-menu', 'columns'),
    State('upload-data', 'contents'),
    State('sheet-checklist', 'value'),
    State('material-dropdown', 'value'),
)
def create_mem(rows_guess, rows_menu, columns_guess, columns_menu,
               contents, selected_sheets, material):
    fig_fit = go.Figure()
    figure_erc = go.Figure(data=go.Scatter(
        x=[0, 1, 2, 3, 4, 5],
        y=[0, 1, 4, 9, 16, 25]
    ))
    graph_style_fit = dict(display='none')
    graph_style_erc = dict(display='none')

    if rows_guess and rows_menu:

        weight = str('none')
        wbtolerance = 0.05
        logid = str('y')
        plotpredictionbands = str('no')
        plotprev = str('yes')
        residual = 0.01  # Recidual relative to saturation level.

        # First we extract the menu choices ###
        # We build up the dataframe again
        df_guess = pd.DataFrame(rows_guess, columns=[c['name'] for c in columns_guess])
        df_menu = pd.DataFrame(rows_menu, columns=[c['name'] for c in columns_menu])

        # Get the model
        model_idx = df_menu['Model'].values
        thickness = df_menu['Thickness'].values
        site_idx = df_menu['Side'].values
        known_t1 = df_menu['Known t1'].values
        known_t2 = df_menu['Known t2'].values
        known_t3 = df_menu['Known t3'].values

        # Convert the format
        model_dict = {'Exposure': 1,
                      'Exp + Burial': 2,
                      'Exp + Burial + Exp': 3,
                      'Exp + Burial + Exp + Burial': 4,
                      'Exp + Burial + Exp + Burial + Exp': 5}
        model_idx = np.array([model_dict[letter] for letter in model_idx])
        site_idx = tuple(site_idx)

        coerced_known_t1 = []
        coerced_known_t2 = []
        coerced_known_t3 = []

        for t1, t2, t3 in zip(known_t1, known_t2, known_t3):

            try:
                coerced_t1 = float(t1)
            except:
                # No possible to coerce
                coerced_t1 = np.nan

            try:
                coerced_t2 = float(t2)
            except:
                # No possible to coerce
                coerced_t2 = np.nan

            try:
                coerced_t3 = float(t3)
            except:
                # No possible to coerce
                coerced_t3 = np.nan

            coerced_known_t1.append(coerced_t1)
            coerced_known_t2.append(coerced_t2)
            coerced_known_t3.append(coerced_t3)

        known_t1 = np.array(coerced_known_t1)
        known_t2 = np.array(coerced_known_t2)
        known_t3 = np.array(coerced_known_t3)

        # Also we Initialize the MEM object ###
        # Decode the full name
        content_type, content_string = contents.split(',')
        data = base64.b64decode(content_string)
        # Initialize

        FIT = MEM(data)

        figure_erc = go.Figure(data=go.Scatter(
            x=[0, 1, 2, 3, 4, 5],
            y=[0, 1, 1, 19, 1, 1]
        ))

        # Read data
        FIT.read_data_array(selected_sheets, weight)

        initial_guess = convert_params_to_guess(df_guess)

        FIT.define_model(material, initial_guess, selected_sheets, model_idx, residual)
        FIT.multiple_fiting_array(selected_sheets, model_idx)
        FIT.run_model_array()
        FIT.Parameterrresults(selected_sheets, model_idx)
        FIT.xp_depth(material, selected_sheets, model_idx)
        FIT.Well_bleached_depth(wbtolerance, model_idx)
        FIT.confidence_bands_array(logid, model_idx, plotpredictionbands,
                                   plotprev, site_idx, thickness)
        FIT.SingleCal(selected_sheets, model_idx, known_t1, known_t2, known_t3)
        FIT.ERC(known_t1, known_t2, known_t3, selected_sheets, model_idx)

        fig_fit = FIT.plotly_fig_fit
        graph_style_fit = dict()

        fig_erc = FIT.plotly_fig_ERC

        if isinstance(fig_erc, list):

            if len(fig_erc) > 0:
                fig_erc = fig_erc[0]
                graph_style_erc = dict()

            elif len(fig_erc) == 0:
                fig_erc = go.Figure(
                    data=go.Scatter(
                        x=[0, 1, 2, 3, 4, 5],
                        y=[20, 1, 1, 19, 1, 1]
                    )
                )

        figure_erc = fig_erc

        gc.collect()

    return fig_fit, graph_style_fit, figure_erc, graph_style_erc


if __name__ == '__main__':
    app.run_server(debug=True)

