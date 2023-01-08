
import base64
import io

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table
from dash.dash_table.Format import Format, Scheme, Symbol

import plotly.graph_objs as go
from collections import OrderedDict

import pandas as pd
import numpy as np
import math


#-----------------------------------------------#
#             Define AgeDating Class            #
#-----------------------------------------------# 

# from MEM import MEM


#-----------------------------------------------#
#            Initialize the Dash App            #
#-----------------------------------------------#  

app = dash.Dash(__name__)
server = app.server
# https://www.youtube.com/watch?v=XWJBJoV5yww

#-----------------------------------------------#
#               Define the Layout               #
#-----------------------------------------------#  

app.layout = html.Div([
    
    html.Div(children=[

        ### Upload Button for the data file ###
        dcc.Upload(id='upload-data',            
                   children=html.Button('Select Files')),
        html.Div(id='upload-text-output'),
        
        ### Checklist for selecting the sheet names ###
        dcc.Checklist(id='sheet-checklist'),

        ### Material selection ###
        dcc.Dropdown(id='material-dropdown',
                   options={
                        'F' : 'Feldspar',
                        'Q'   : 'Quartz'
                   },
                   value='F'
        ),
    ], style={'padding': 10, 
              'flex': 1,
              "width": "20%"}),
              
    
    # html.Div(children=[

    #     ### Table menu ###
    #     dash_table.DataTable(id='option-menu',
    #                          editable=True),
        
    #     dcc.Graph(id='fit-graph',
    #               style = dict(display='none')),
        
    # ], style={'padding': 10, 'flex': 1}),
    
    # ### Accept button ###
    # html.Button('Accept',
    #             id='accept-menu-button',
    #             n_clicks=0,
    #             style = dict(display='none')),
    
    # ### Guess Plot ### 
    # html.Div(children=[
        
    #     html.Div(children=[
            
    #         html.Div(id='k1_slider-container', children=[
            
    #             dcc.Slider(id='k1_slider',
    #                        marks={
    #                            0:{'label': '0'},
    #                            0.99:{'label':'K1'}
    #                            },
    #                        tooltip={"placement": "right"},
    #                        min=0, 
    #                        max=0.99,
    #                        value=0.5,
    #                        included=False,
    #                        vertical=True
    #             ),
    #         ], style= {'display': 'none'}),
            
    #         html.Div(id='k2_slider-container', children=[
            
    #             dcc.Slider(id='k2_slider',
    #                        marks={
    #                            0:{'label': '0'},
    #                            0.99:{'label':'K2'}
    #                            },
    #                        tooltip={"placement": "right"},
    #                        min=0, 
    #                        max=0.99,
    #                        value=0.5,
    #                        included=False,
    #                        vertical=True
    #             ),
    #         ], style= {'display': 'none'}),
    
    #         dcc.Graph(id='guess-graph',
    #                   style = dict(display='none')),
            
    #         dcc.Graph(id='ERC-graph',
    #                   style = dict(display='none')),
            
    #     ], style={'display': 'flex', 'flex-direction': 'row'}),
        
    #     # Create Div to place a conditionally visible element inside
    #     html.Div(id='x1_slider-container', children=[
        
    #         dcc.Slider(id='x1_slider',
    #                    min=0, 
    #                    max=100,
    #                    value=65,
    #                     marks={
    #                         0:{'label': '0'},
    #                         1:{'label':'xp1'}
    #                         },
    #                     tooltip={"placement": "right"},
    #                    included=False
    #         ),
    #     ], style= {'display': 'none'}),
        
        
    #     html.Div(id='x2_slider-container', children=[
        
    #         dcc.Slider(id='x2_slider',
    #                    marks={'label':'xp2'},
    #                    min=0, 
    #                    max=100,
    #                    value=65,
    #                    included=False
    #         ),
    #     ], style= {'display': 'none'}),
        
    #     html.Div(id='x3_slider-container', children=[
        
    #         dcc.Slider(id='x3_slider',
    #                    marks={'label':'xp3'},
    #                    min=0, 
    #                    max=100,
    #                    value=65,
    #                    included=False
    #         ),
    #     ], style= {'display': 'none'}),
        
    #     dash_table.DataTable(id='guess-modified',
    #                          data=None,
    #                          columns=None),
        
    # ], style={'padding': 10, 
    #           'flex': 1,
    #           "width": "20%"}),
    

], style={'display': 'flex', 'flex-direction': 'row'})

              
              
              
              
#-----------------------------------------------#
#              Define the Callbacks             #
#-----------------------------------------------#  


@app.callback(
    Output('upload-text-output', 'children'),
    Output('sheet-checklist', 'options'),
    Output('sheet-checklist', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def read_sheet_names(contents, filename):
    
    """
    filename: name of the chosen file 
    content: content of that file
    """

    sheet_names = []
    return_string='Content is none'
    
    if contents is not None:
    
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if 'xlsx' in filename:
            # Assume an excel file has bee chosen
            try:
                df_dict = pd.ExcelFile(io.BytesIO(decoded))
                sheet_names = df_dict.sheet_names
                return_string = 'All Good'
            except:
                return_string = 'Can not open but it is an Excel'

        else:
            return_string= 'Not an excel file!!'

    return return_string, sheet_names, sheet_names


# @app.callback(
#     Output('option-menu', 'data'),
#     Output('option-menu', 'columns'),
#     Output('option-menu', 'dropdown'),
#     Output('accept-menu-button', 'style'),
#     Input('sheet-checklist', 'value')
# )

# def create_menu_table(selectd_sheets):
    
#     # Define menu options
#     side_options = ['Front', 'End', 'Front', 'Front', 'Front']
#     model_options = ['Exposure', 
#                      'Exp + Burial',
#                      'Exp + Burial + Exp',
#                      'Exp + Burial + Exp + Burial',
#                      'Exp + Burial + Exp + Burial + Exp']
    
#     df_options = pd.DataFrame(OrderedDict([
#                  ('side_options', side_options),
#                  ('model_options', model_options)
#     ]))
    
#     # We create the initial data for each column
#     model = [model_options[0]] * len(selectd_sheets) 
#     side = [side_options[0]] * len(selectd_sheets) 
#     thickness = [50] * len(selectd_sheets) 
        
#     known_t1 = ['Unknown'] * len(selectd_sheets) 
#     known_t2 = known_t1
#     known_t3 = known_t1
    
#     df = pd.DataFrame(OrderedDict([
#          ('Sample', selectd_sheets),
#          ('Model', model),
#          ('Side', side),
#          ('Thickness', thickness),
#          ('Known t1', known_t1),
#          ('Known t2', known_t2),
#          ('Known t3', known_t3)
#     ]))
    
#     # Define the drop down menu
    
#     dropdown={
#         'Model': {
#             'options': [
#                     {'label': i, 'value': i}
#                     for i in df_options['model_options'].unique()
#                 ]
#         },
#         'Side': {
#              'options': [
#                     {'label': i, 'value': i}
#                     for i in df_options['side_options'].unique()
#                 ]
#         }
#     }
    

#     # Lets stick to the nomenclature expected by the dcc.DataTable
#     data = df.to_dict('records')
    
#     # columns = [{'name': col, 'id': col} for col in df.columns]
#     columns=[
#         {
#             'id': 'Sample',
#             'name': 'Sample',
#             'type': 'text'
#         }, {
#             'id': 'Model',
#             'name': 'Model', 
#             'presentation': 'dropdown'
#         }, {
#         'id': 'Side',
#         'name': 'Side',
#         'presentation': 'dropdown'
#         }, {
#         'id': 'Thickness', 
#         'name': 'Thickness',
#         'type': 'numeric',
#         'format': Format(
#             precision=1,
#             scheme=Scheme.fixed,
#             symbol=Symbol.yes,
#             symbol_suffix= ' [mm]'),
#         }, {
#         'id': 'Known t1', 
#         'name': 'Known t1',
#         'type': 'numeric',
#         'format': Format(
#             precision=3,
#             scheme=Scheme.fixed,
#             symbol=Symbol.yes,
#             symbol_suffix= ' days'),
#         }, {
#         'id': 'Known t2', 
#         'name': 'Known t2',
#         'type': 'numeric',
#         'format': Format(
#             precision=3,
#             scheme=Scheme.fixed,
#             symbol=Symbol.yes,
#             symbol_suffix= ' days'),
#         }, {
#         'id': 'Known t3', 
#         'name': 'Known t3',
#         'type': 'numeric',
#         'format': Format(
#             precision=3,
#             scheme=Scheme.fixed,
#             symbol=Symbol.yes,
#             symbol_suffix= ' days'),
#         }
#     ]
            
#     button_Style = dict()
    
#     if len(selectd_sheets) == 0:
        
#         data = None
#         columns = None
#         dropdown = None
#         button_Style = dict(display='none')
    
#     return data, columns, dropdown, button_Style
    

# @app.callback(
#     Output('x1_slider', 'max'),
#     Output('x1_slider-container', 'style'),
#     Output('x2_slider', 'max'),
#     Output('x2_slider-container', 'style'),
#     Output('x3_slider', 'max'),
#     Output('x3_slider-container', 'style'),
#     Output('k1_slider-container', 'style'),
#     Output('k2_slider-container', 'style'),
#     Input('accept-menu-button', 'n_clicks'),
#     State('option-menu', 'data'),
#     State('option-menu', 'columns'),
#     State('upload-data', 'contents'),
#     State('sheet-checklist', 'value'),
#     State('material-dropdown', 'value')
# )
# def create_guess_graph_frame(n_clicks, rows, columns, contents,
#                              selected_sheets, material):
    
#     if n_clicks > 0:
        
#         ### First we extract the menu choices ###
        
#         # We build up the dataframe again
#         df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
        
#         # Get the model
#         model_idx = df['Model'].values
        
#         # Convert the format of the model
#         model_dict = {'Exposure' : 1, 
#                       'Exp + Burial' : 2, 
#                       'Exp + Burial + Exp' : 3, 
#                       'Exp + Burial + Exp + Burial' : 4, 
#                       'Exp + Burial + Exp + Burial + Exp' : 5 }
        
#         model_idx = np.array([model_dict[letter] for letter in model_idx])        
        
#         ### Also we Initialize the MEM object ###
        
#         # Decode the full name
#         content_type, content_string = contents.split(',')
#         full_name = base64.b64decode(content_string)
        
#         # Initialize
#         FIT = MEM(full_name)
        
#         # Read data
#         weight= str('none')
#         FIT.read_data_array(selected_sheets, weight)
        
#         # Create initial guess
#         initial_guess, df_data, df_columns = define_initial_guess(selected_sheets)
        
#         residual =0.01
#         FIT.define_model(material, initial_guess, selected_sheets, model_idx, residual)
#         FIT.multiple_fiting_array(selected_sheets, model_idx)
        
#         xmax = max(FIT.xall[0])+0.5
        
#         # We round it up 
#         xmax = math.ceil(xmax)
          
#         horiz_slider_style= {'display': 'block',
#                              'height': '20px',
#                              'width': '700px'}
        
#         vert_slider_style= {'display': 'block',
#                              'height': '700px',
#                              'width': '40px'}

#     else:
        
#         xmax=1000
#         horiz_slider_style = {'display': 'none'}
#         vert_slider_style = {'display': 'none'}
#         initial_guess = []
    
#     return (xmax, horiz_slider_style, xmax, horiz_slider_style, 
#             xmax, horiz_slider_style, vert_slider_style, vert_slider_style)



# def define_initial_guess(sheet_names):
    
#     # Initial guess values
#     r=2.1
#     u=1.38
#     xp1=7
#     K1=0.03
#     xp2=30
#     K2=1
#     xp3=0.01
    
#     guess = np.array([r] + [u, np.exp(xp1*u), -np.log(1-K1), np.exp(xp2*u),
#                      -np.log(1-K1), np.exp(xp3*u)] * len(sheet_names))
    
#     params = [r, u, xp1, K1, xp2, K2, xp2]
#     param_list = []
#     for i, name in enumerate(sheet_names):
#         param_list.append(params)
        
#     df = pd.DataFrame(np.array(param_list),
#                       index=sheet_names,
#                       columns=['r', 'u', 'xp1', 'K1', 'xp2', 'K2', 'xp3'])
    
#     df_data = df.to_dict('records')
#     df_columns = [{"name": i, "id": i} for i in df.columns]

#     return guess, df_data, df_columns

    
# @app.callback(
#     Output('guess-modified', 'data'),
#     Output('guess-modified', 'columns'),
#     Input('x1_slider', 'value'),
#     Input('x2_slider', 'value'),
#     Input('x3_slider', 'value'),
#     Input('k1_slider', 'value'),
#     Input('k2_slider', 'value'),
#     State('guess-modified', 'data'),
#     State('guess-modified', 'columns'),
#     State('sheet-checklist', 'value'),
# )
# def modify_guess(xp1, xp2, xp3, K1, K2, data, columns, sheet_names):
    
#     if sheet_names is not None:
    
#         if data is None:
        
#             # Initial guess values
#             r=2.1
#             u=1.38
#             xp1=7
#             K1=0.03
#             xp2=30
#             K2=1
#             xp3=0.01
        
#             params = [r, u, xp1, K1, xp2, K2, xp3]
#             param_list = []
#             for i, name in enumerate(sheet_names):
#                 param_list.append(params)
                
#             df = pd.DataFrame(np.array(param_list),
#                               index=sheet_names,
#                               columns=['r', 'u', 'xp1', 'K1', 'xp2', 'K2', 'xp3'])
            
#         else:
            
#             df = pd.DataFrame(data,
#                               columns=[c['name'] for c in columns],
#                               index=sheet_names)
            
#             df['xp1'][sheet_names[0]] = xp1
#             df['xp2'][sheet_names[0]] = xp2
#             df['xp3'][sheet_names[0]] = xp3
#             df['K1'][sheet_names[0]] = K1
#             df['K2'][sheet_names[0]] = K2
            
#         df_data = df.to_dict('records')
#         df_columns = [{"name": i, "id": i} for i in df.columns]
        
#     else:
        
#         df_data=None
#         df_columns=None
        
#     return df_data, df_columns
    
# @app.callback(
#     Output('guess-graph', 'figure'),
#     Output('guess-graph', 'style'),
#     Input('guess-modified', 'data'),
#     State('guess-modified', 'columns'),
#     State('option-menu', 'data'),
#     State('option-menu', 'columns'),
#     State('upload-data',  'contents'),
#     State('sheet-checklist', 'value'),
#     State('material-dropdown', 'value'),
#     State('accept-menu-button', 'n_clicks')
# )
# def create_guess_fig(rows_guess, columns_guess, rows_menu, columns_menu,
#                      contents, selected_sheets, material, n_clicks):
    
#     if n_clicks > 0:
        
#         ### First we extract the menu choices ###
#         # We build up the dataframe again
#         df_guess = pd.DataFrame(rows_guess, columns=[c['name'] for c in columns_guess])
#         df_menu = pd.DataFrame(rows_menu, columns=[c['name'] for c in columns_menu])
#         # Get the model
#         model_idx = df_menu['Model'].values
#         # Convert the format
#         model_dict = {'Exposure' : 1, 
#                       'Exp + Burial' : 2, 
#                       'Exp + Burial + Exp' : 3, 
#                       'Exp + Burial + Exp + Burial' : 4, 
#                       'Exp + Burial + Exp + Burial + Exp' : 5 }
#         model_idx = np.array([model_dict[letter] for letter in model_idx])
#         bla = f'Model idx: {model_idx}'
        
#         ### Also we Initialize the MEM object ###
#         # Decode the full name
#         content_type, content_string = contents.split(',')
#         full_name = base64.b64decode(content_string)
#         # Initialize
#         FIT = MEM(full_name)
#         # Read data
#         weight= str('none')
#         FIT.read_data_array(selected_sheets, weight)
        
#         initial_guess = convert_params_to_guess(df_guess)
        
#         residual =0.01
#         FIT.define_model(material, initial_guess, selected_sheets, model_idx, residual)
#         FIT.multiple_fiting_array(selected_sheets, model_idx)
    
#         plot_content = []
#         for idx, name in enumerate(selected_sheets):
#             plot_content.append(go.Scatter(x=FIT.xall[idx],
#                                            y=FIT.yall[idx],
#                                            error_y=dict(
#                                                 type='data', # value of error bar given in data coordinates
#                                                 array=FIT.errall[idx],
#                                                 visible=True),
#                                            mode='markers',
#                                            name=name))
           
#         # Get some values
#         i =0
#         xmax = max(FIT.xall[i])+0.5
#         xi = np.linspace(0,xmax,100)
#         y_guess_temp = FIT.fun(xi,i,model_idx[i],*FIT.P0)
#             # plt.plot(xi, y_guess_temp, color = 'green', label='Guess')
        
#         plot_content.append(go.Scatter(x=xi,
#                                        y=y_guess_temp,
#                                        name='blu'))
        
#         layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)',
#                            plot_bgcolor='rgba(0,0,0,0)',
#                            title="Initial Guess",
#                            height=500,
#                            width=500)
        
#         fig = go.Figure(data=plot_content,
#                         layout=layout)
        
#         graph_Style = dict()    

#     else:
        
#         fig = {}
#         graph_Style = dict(display='none')

#     return fig, graph_Style


# def convert_params_to_guess(df_params):
    
#     guess = []
    
#     for index, row in df_params.iterrows():
        
#         r = row['r']
#         u = row['u']
#         xp1 = row['xp1']
#         K1 = row['K1']
#         xp2 = row['xp2']
#         K2 = row['K2']
#         xp3 = row['xp3']
        
#         if len(guess) == 0:
#             guess.append(r)
            
#         guess.append(u)
#         guess.append(np.exp(xp1*u))
#         guess.append(-np.log(1-K1))
#         guess.append(np.exp(xp2*u))
#         guess.append(-np.log(1-K2))
#         guess.append(np.exp(xp3*u))
        
#     return np.array(guess)


# @app.callback(
#     Output('fit-graph', 'figure'),
#     Output('fit-graph', 'style'),
#     Output('ERC-graph', 'figure'),
#     Output('ERC-graph', 'style'),
#     Input('guess-modified', 'data'),
#     Input('option-menu', 'data'),
#     State('guess-modified', 'columns'),
#     State('option-menu', 'columns'),
#     State('upload-data',  'contents'),
#     State('sheet-checklist', 'value'),
#     State('material-dropdown', 'value'),
#     State('accept-menu-button', 'n_clicks')
# )        
# def create_MEM(rows_guess, rows_menu, columns_guess, columns_menu,
#                contents, selected_sheets, material, n_clicks):
    
#     if n_clicks > 0:
    
#         weight= str('none')
#         WBtolerance = 0.05
#         logID = str('y')
#         PlotPredictionBands = str('no')
#         PlotPrev = str('yes')
#         residual=0.01 #Recidual relative to saturation level. 
        
#         ### First we extract the menu choices ###
#         # We build up the dataframe again
#         df_guess = pd.DataFrame(rows_guess, columns=[c['name'] for c in columns_guess])
#         df_menu = pd.DataFrame(rows_menu, columns=[c['name'] for c in columns_menu])
        
#         # Get the model
#         model_idx = df_menu['Model'].values
#         Thickness = df_menu['Thickness'].values
#         Site_idx = df_menu['Side'].values
#         known_t1 = df_menu['Known t1'].values
#         known_t2 = df_menu['Known t2'].values
#         known_t3 = df_menu['Known t3'].values

#         # Convert the format
#         model_dict = {'Exposure' : 1, 
#                       'Exp + Burial' : 2, 
#                       'Exp + Burial + Exp' : 3, 
#                       'Exp + Burial + Exp + Burial' : 4, 
#                       'Exp + Burial + Exp + Burial + Exp' : 5 }
#         model_idx = np.array([model_dict[letter] for letter in model_idx])
#         Site_idx = tuple(Site_idx)
        
#         coerced_known_t1 = []
#         coerced_known_t2 = []
#         coerced_known_t3 = []
        
#         for t1, t2, t3 in zip(known_t1, known_t2, known_t3):
            
#             try:
#                 coerced_t1 = float(t1)
#             except:
#                 # No possible to coerce
#                 coerced_t1 = np.nan
                
#             try:
#                 coerced_t2 = float(t2)
#             except:
#                 # No possible to coerce
#                 coerced_t2 = np.nan
                    
#             try:
#                 coerced_t3 = float(t3)
#             except:
#                 # No possible to coerce
#                 coerced_t3 = np.nan
                
#             coerced_known_t1.append(coerced_t1)
#             coerced_known_t2.append(coerced_t2)
#             coerced_known_t3.append(coerced_t3)
    
#         known_t1 = np.array(coerced_known_t1)
#         known_t2 = np.array(coerced_known_t2)
#         known_t3 = np.array(coerced_known_t3)
    
#         ### Also we Initialize the MEM object ###
#         # Decode the full name
#         content_type, content_string = contents.split(',')
#         full_name = base64.b64decode(content_string)
#         # Initialize
        
#         FIT = MEM(full_name)
        
#         # Read data
#         FIT.read_data_array(selected_sheets, weight)
        
#         initial_guess = convert_params_to_guess(df_guess)
        
#         FIT.define_model(material, initial_guess, selected_sheets, model_idx, residual)
#         FIT.multiple_fiting_array(selected_sheets, model_idx)
#         FIT.run_model_array()
#         FIT.Parameterrresults(selected_sheets, model_idx)
#         FIT.xp_depth(material,selected_sheets,model_idx)
#         FIT.Well_bleached_depth(WBtolerance,model_idx)
#         FIT.confidence_bands_array(logID, model_idx, PlotPredictionBands,
#                                    PlotPrev, Site_idx, Thickness)
#         FIT.SingleCal(selected_sheets,model_idx,known_t1,known_t2, known_t3)
#         FIT.ERC(known_t1,known_t2,known_t3,selected_sheets,model_idx)
        
        
#         fig_fit = FIT.plotly_fig_fit
#         graph_Style_fit = dict()
        
#         try:
#             fig_ERC = FIT.plotly_fig_ERC
#             graph_Style_ERC = dict()
#         except:
#             fig_ERC=None
#             graph_Style_ERC = dict(display='none')

#     else:
        
#         fig_fit=None
#         fig_ERC=None
#         graph_Style_fit = dict(display='none')
#         graph_Style_ERC = dict(display='none')
    
#     return fig_fit, graph_Style_fit, fig_ERC, graph_Style_ERC


if __name__ == '__main__':
    app.run_server(debug=True)
