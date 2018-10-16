##
## Imports
import pprint
import utils
import markdown_text
import os
import json
from palettable.colorbrewer import qualitative as colors
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import logging
from matplotlib import cm
from textwrap import wrap

##
## Initial setup

# set the logging level
logging.basicConfig(level=logging.DEBUG)

# Colors!
def getColorScaleCont(colorCode, colorrange, nbins=50):
    cmap = cm.get_cmap(colorCode)
    gradient = np.linspace(colorrange[0], colorrange[1], nbins)
    colors = ['rgb(%i, %i, %i)' % (x[0],x[1],x[2]) for x in cmap(gradient)[:,:3]*256]
    scale = [[x, y] for x, y in zip(gradient, colors)]

    return scale

def getColorScaleCat(colorCode, colorvals):
    cmap = cm.get_cmap(colorCode)
    colorvals=np.asarray(colorvals)
    gradient = (colorvals-np.min(colorvals)) / (np.max(colorvals) - np.min(colorvals))
    colors = ['rgb(%i, %i, %i)' % (x[0],x[1],x[2]) for x in cmap(gradient)[:,:3]*256]
    scale = [[x, y] for x, y in zip(gradient, colors)]

    return scale

colorList = [colors.Dark2_8.colors,
             colors.Paired_12.colors,
             colors.Set2_8.colors,
             colors.Set3_12.colors]

colorInfo = {
    'topic': ['rgb(' + str(item)[1:-1] + ')' for sublist in colorList for item in sublist],
    'sentiment': 'RdBu',
    'emotion': {
        'anger': {'data': 'watson_emotion_anger', 'color': 'Reds'},
        'disgust': {'data': 'watson_emotion_disgust', 'color': 'Greens'},
        'fear': {'data': 'watson_emotion_fear', 'color': 'Purples'},
        'joy': {'data': 'watson_emotion_joy', 'color': 'YlOrBr'},
        'sadness': {'data': 'watson_emotion_sadness', 'color': 'Blues'},
    }
}

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

table_cols = ['_text',
              'topic_code',
              'watson_sentiment',
              'watson_emotion_anger',
              'watson_emotion_disgust',
              'watson_emotion_fear',
              'watson_emotion_joy',
              'watson_emotion_sadness']

table_nice_names = ['text',
                    'topic',
                    'sentiment',
                    'anger',
                    'disgust',
                    'fear',
                    'joy',
                    'sadness']

##
## Data preparation

#  Prep the data
def configure_data(data):
    df_groups = data[['topic_code', 'cx', 'cy', 'cz']].groupby('topic_code').agg(np.mean).reset_index()
    df_groups['colors'] = df_groups.topic_code.apply(lambda x: colorInfo['topic'][int(x % len(colorInfo['topic']))])

    colorMap = [[x.topic_code / float(max(df_groups.topic_code)), x.colors] for _, x in df_groups.iterrows()]

    data = data.merge(df_groups.drop(['cx', 'cy', 'cz'], axis=1), on='topic_code')

    ## ENRICHMENTS

    # line wrapping
    data['hovertext'] = data.apply(lambda x: '<br>'.join(wrap(x._text.replace('\n', ''), 45)),
                                     axis=1)

    # sentiment and emotion scores
    data['hovertext'] = data.apply(lambda x: x.hovertext + \
                                               '<br><b>Watson Sentiment:</b> ' + \
                                               utils.get_sentiment_string(x) + \
                                               '<br><b>Watson Top Emotion:</b> ' + \
                                               utils.get_top_emotion_string(x),
                                     axis=1)

    return data, colorMap

def get_data_elements(data, colorMap):

    # get all of the data elements
    x, y, z, = np.asarray(data[['cx', 'cy', 'cz']]).transpose()
    topic_code = data['topic_code'].tolist()
    topic_colors = data['colors'].tolist()
    hovertext = np.asarray(data['hovertext']).transpose()
    sentiment = data['watson_sentiment'].tolist()
    anger, disgust, fear, joy, sadness = np.asarray(data[['watson_emotion_anger',
                                                          'watson_emotion_disgust',
                                                          'watson_emotion_fear',
                                                          'watson_emotion_joy',
                                                          'watson_emotion_sadness']]).transpose()


    elements = dict(
        coords = dict(x=x, y=y, z=z),
        text = hovertext,
        topic = dict(code=topic_code, color=topic_colors),
        watson = dict(sentiment=sentiment,
                      emotion=dict(anger=anger,
                                    disgust=disgust,
                                    fear=fear,
                                    joy=joy,
                                    sadness=sadness)
                      ),
        colorMap = colorMap
    )

    return elements



##
## Globally define data elements
dataPath = 'fb_posts_tsne_3d.csv'
data = pd.read_csv(dataPath, encoding='utf-8', sep=',')

# prepare the data
data, colorMap = configure_data(data)
data_elements = get_data_elements(data, colorMap)

# Get defaults for data
default_camera = {'eye':
                      {'y': 0.53,
                       'x': 0.75,
                       'z': 0.7},
                  'up': {'y': 0,
                         'x': 0,
                         'z': 1},
                  'center': {'y': 0,
                             'x': 0,
                             'z': 0}
                  }


# default scene
default_scene = dict(
    camera=default_camera,
    xaxis=dict(
        range=[min(data["cx"]), max(data["cx"])],
        autorange=False
    ),
    yaxis=dict(
        range=[min(data["cy"]), max(data["cy"])],
        autorange=False
    ),
    zaxis=dict(
        range=[min(data["cz"]), max(data["cz"])],
        autorange=False
    )
)

##
## Plot functions

# 3d scatterplot
def plot3dscatter(x, y, z, color_var, color_range, hover_text, label_colors, color_map, reversescale=False):

    # Create the plot
    plt = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=3,
            cmin=color_range[0],
            cmax=color_range[1],
            color=color_var,  # set color to an array/list of desired values
            colorscale=color_map,  # choose a colorscale
            reversescale=reversescale,
            opacity=0.8
        ),
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor=label_colors
        ),
        text=hover_text,
        textposition="top center"
    )

    return plt


# histogram
def plotHistogram(x, color_var, color_range, colorscale, nbins):
    plt = go.Histogram(x=x,
                       marker=dict(
                           cmin=color_range[0],
                           cmax=color_range[1],
                           color=color_var,
                           colorscale=colorscale,  # choose a colorscale
                           autocolorscale=False
                           ),
                       xbins=dict(
                           start=color_range[0],
                           end=color_range[1],
                           size=(color_range[1]-color_range[0])/float(nbins)
                           ),
                       autobinx=False
                       )

    return plt

# bar graph
def plotBar(x, y, color_var, color_range, colorscale, label_colors, hover_text, selectedTopics=None):
    plt = go.Bar(x=x, y=y,
                 marker=dict(
                     cmin=color_range[0],
                     cmax=color_range[1],
                     color=color_var,
                     colorscale=colorscale,  # choose a colorscale
                     autocolorscale=False
                 ),
                 hoverinfo='text',
                 hoverlabel=dict(
                     bgcolor=label_colors
                 ),
                 text=hover_text,
                 orientation='h',
                 selectedpoints=selectedTopics
                 )

    return plt

##
## Build the 'figure' elements for the dashboard (plot + layout specification)

# 3D figures
def get3dfigure(dell, type=1, subtype=1, camera = default_camera):
    # logic for choosing variables
    if type == 1: #topic
        color_var = dell['topic']['code']
        color_range = [0,29]#min(dell['topic']['code']), max(dell['topic']['code'])]
        label_colors = dell['topic']['color']
        color_map = dell['colorMap']
        reversescale=False
    elif type == 2: #sentiment
        color_var = dell['watson']['sentiment']
        color_range = [-1,1]
        label_colors = 'aliceblue'
        color_map = colorInfo['sentiment']
        reversescale = True
    elif type == 3: #emotion
        color_range = [0,1]
        color_bins=500
        label_colors = 'aliceblue'
        reversescale=False
        # different colors for different sub-types
        if subtype==1:
            color_var = dell['watson']['emotion']['anger']
            color_map = getColorScaleCont(colorInfo['emotion']['anger']['color'], color_range, nbins=color_bins)
        elif subtype == 2:
            color_var = dell['watson']['emotion']['disgust']
            color_map = getColorScaleCont(colorInfo['emotion']['disgust']['color'], color_range, nbins=color_bins)
        elif subtype == 3:
            color_var = dell['watson']['emotion']['fear']
            color_map = getColorScaleCont(colorInfo['emotion']['fear']['color'], color_range, nbins=color_bins)
        elif subtype == 4:
            color_var = dell['watson']['emotion']['joy']
            color_map = getColorScaleCont(colorInfo['emotion']['joy']['color'], color_range, nbins=color_bins)
        elif subtype == 5:
            color_var = dell['watson']['emotion']['sadness']
            color_map = getColorScaleCont(colorInfo['emotion']['sadness']['color'], color_range, nbins=color_bins)
        else:
            logging.error("Invalid 3d plot sub-type: %s", subtype)
            raise Exception

    else:
        logging.error("Invalid 3d plot type: %s", type)
        raise Exception

    # build the figure
    fig = {
        'data': [plot3dscatter(x=dell['coords']['x'],
                               y=dell['coords']['y'],
                               z=dell['coords']['z'],
                               color_var = color_var,
                               color_range = color_range,
                               hover_text=dell['text'],
                               label_colors = label_colors,
                               color_map = color_map,
                               reversescale=reversescale)],
        'layout': go.Layout(
            height=900,
            margin={'l': 0, 'b': 10, 't': 25, 'r': 10},
            hovermode='closest',
            scene=dict(
                camera=camera,
                xaxis=default_scene['xaxis'],
                yaxis=default_scene['yaxis'],
                zaxis=default_scene['zaxis'],
                aspectmode='manual',
                aspectratio=dict(
                    x=1,
                    y=1,
                    z=1
                )
            ),
        ),
    }

    return fig


# Histogram figure
def getHistFigure(dell, type, subtype=None):
    # logic for choosing plot characteristics based on type and subtype

    # SENTIMENT
    if type == 'sentiment':
        x = dell['watson']['sentiment']
        color_range = [-1, 1]
        colorscale = colorInfo['sentiment']
        nbins = 40
        reversescale = False
        xaxis = dict(
            tickvals=[-1,0, 1],
            range = [-1,1],
            ticktext=["Negative", "Neutral", "Positive"]
        )

    # EMOTION
    elif type == 'emotion':
        # settings for all emotion-type histograms
        color_range = [0, 1]
        nbins = 40
        xaxis = dict(
            tickvals=[0,1],
            range=[0,1],
            ticktext=["",""]
        )
        reversescale=True

        # sub-type differences
        if subtype ==1:
            x = dell['watson']['emotion']['anger']
            colorscale = getColorScaleCont(colorInfo['emotion']['anger']['color'], color_range, nbins=nbins)
            xaxis['title']='Anger'
        elif subtype==2:
            x = dell['watson']['emotion']['disgust']
            colorscale = getColorScaleCont(colorInfo['emotion']['disgust']['color'], color_range, nbins=nbins)
            xaxis['title'] = 'Disgust'
        elif subtype==3:
            x = dell['watson']['emotion']['fear']
            colorscale = getColorScaleCont(colorInfo['emotion']['fear']['color'], color_range, nbins=nbins)
            xaxis['title'] = 'Fear'
        elif subtype==4:
            x = dell['watson']['emotion']['joy']
            colorscale = getColorScaleCont(colorInfo['emotion']['joy']['color'], color_range, nbins=nbins)
            xaxis['title'] = 'Joy'
        elif subtype==5:
            x = dell['watson']['emotion']['sadness']
            colorscale = getColorScaleCont(colorInfo['emotion']['sadness']['color'], color_range, nbins=nbins)
            xaxis['title'] = 'Sadness'
        else:
            logging.error("Invalid 3d plot sub-type: %s", subtype)
            raise Exception

    else:
        logging.error("Invalid 3d plot type: %s", type)
        raise Exception

    # Handle reversescale manually
    if reversescale:
        color_var = np.linspace(color_range[0], color_range[1], nbins)
    else:
        color_var = np.linspace(color_range[1], color_range[0], nbins)

    fig = {
        'data': [plotHistogram(x, color_var, color_range, colorscale, nbins)],
        'layout': go.Layout(
            height=130.,
            margin={'l': 50, 'b': 25, 't': 0, 'r': 25},
            dragmode='select',
            selectdirection='h',
            xaxis=xaxis,
            yaxis=dict(
                title='Count',
                type='log',
                autorange=True
            )
        )
    }

    return fig


# Bar Figure
def getBarFigure(df, type=1, subtype=1, selectedTopics = None):

    # group the dataframe and calculate topic-level statistics for visualization
    df_grouped = df.groupby(['topic_code']).\
        apply(lambda x: utils.get_topic_stats(x)).\
        reset_index().\
        drop('level_1', axis=1)

    # format the plot based on the type and subtype
    if type == 1:  # topic
        sort_idx = np.argsort(df_grouped['count'])
        df_grouped = df_grouped.iloc[sort_idx]
        color_var = df_grouped['topic_code']
        color_map = colorMap
    elif type==2: # sentiment
        sort_idx = np.argsort(df_grouped['mean_sentiment'])
        df_grouped = df_grouped.iloc[sort_idx]
        color_var = df_grouped['mean_sentiment']
        color_map = getColorScaleCat(colorInfo['sentiment'], color_var)
    elif type==3: # emotion
        if subtype == 1:
            sort_idx = np.argsort(df_grouped['mean_emotion_anger'])
            df_grouped = df_grouped.iloc[sort_idx]
            color_var = df_grouped['mean_emotion_anger']
            color_map = getColorScaleCat(colorInfo['emotion']['anger']['color'], color_var)
        elif subtype == 2:
            sort_idx = np.argsort(df_grouped['mean_emotion_disgust'])
            df_grouped = df_grouped.iloc[sort_idx]
            color_var = df_grouped['mean_emotion_disgust']
            color_map = getColorScaleCat(colorInfo['emotion']['disgust']['color'], color_var)
        elif subtype == 3:
            sort_idx = np.argsort(df_grouped['mean_emotion_fear'])
            df_grouped = df_grouped.iloc[sort_idx]
            color_var = df_grouped['mean_emotion_fear']
            color_map = getColorScaleCat(colorInfo['emotion']['fear']['color'], color_var)
        elif subtype == 4:
            sort_idx = np.argsort(df_grouped['mean_emotion_joy'])
            df_grouped = df_grouped.iloc[sort_idx]
            color_var = df_grouped['mean_emotion_joy']
            color_map = getColorScaleCat(colorInfo['emotion']['joy']['color'], color_var)
        elif subtype == 5:
            sort_idx = np.argsort(df_grouped['mean_emotion_sadness'])
            df_grouped = df_grouped.iloc[sort_idx]
            color_var = df_grouped['mean_emotion_sadness']
            color_map = getColorScaleCat(colorInfo['emotion']['sadness']['color'], color_var)
        else:
            logging.error("Invalid bar plot sub-type: %s", subtype)
            raise Exception
    else:
        logging.error("Invalid bar plot type: %s", type)
        raise Exception

    df_grouped.reset_index(inplace=True)

    # get indices for selected topics
    if selectedTopics is not None and len(selectedTopics)>0:
        selIdx = df_grouped.index[df_grouped['topic_code'].isin(selectedTopics)]
    else:
        selIdx = None

    # settings that apply to all
    y = df_grouped['topic_code']
    x = df_grouped['count']
    color_range = [min(color_var), max(color_var)]
    label_colors = 'aliceblue'
    hover_text = y

    fig = {
        'data': [plotBar(x, y, color_var, color_range, color_map, label_colors, hover_text, selectedTopics=selIdx)],
        'layout': go.Layout(
            height=900.,
            margin={'l': 50, 'b': 40, 't': 25, 'r': 25},
            clickmode='event',
            dragmode='zoom',
            yaxis = dict(
                fixedrange=True,
                type="category",
                title="Topic"
            ),
            xaxis=dict(
                fixedrange=True,
                title='Count',
                autorange=True
            )
        )
    }

    return fig

def configure_dashboard(dell):

    # INITIALIZE THE DASHBOARD
    app = dash.Dash(__name__)

    # LAYOUT THE DASHBOARD -- main div
    app.layout = html.Div(children=[

        ##
        ## Title
        html.Div([
            dcc.Markdown(children=markdown_text.mrk_title),
            html.Hr(),
        ], style={'width': '100%', 'display': 'inline-block'}),

        ##
        ## main widgets
        html.Div([

            # top row with controls and sentiment histogram
            html.Div([

                html.Div([
                    # text tips
                    dcc.Markdown(markdown_text.tips),
                    html.Hr(),
                ], className='twelve columns'),

            ], className='twelve columns'),


            # Middle area with emotion hists and 3d graph
            html.Div([

                # left column
                html.Div([
                    # histograms
                    dcc.Graph(id='hist-sentiment', figure=getHistFigure(dell, 'sentiment'), config={'displayModeBar': False}),
                    dcc.Graph(id='hist-emotion-anger', figure=getHistFigure(dell, 'emotion', 1), config={'displayModeBar': False}),
                    dcc.Graph(id='hist-emotion-disgust', figure=getHistFigure(dell, 'emotion', 2), config={'displayModeBar': False}),
                    dcc.Graph(id='hist-emotion-fear', figure=getHistFigure(dell, 'emotion', 3), config={'displayModeBar': False}),
                    dcc.Graph(id='hist-emotion-joy', figure=getHistFigure(dell, 'emotion', 4), config={'displayModeBar': False}),
                    dcc.Graph(id='hist-emotion-sadness', figure=getHistFigure(dell, 'emotion', 5), config={'displayModeBar': False}),
                ], className='three columns'),


                # middle and right columns
                html.Div([

                    # middle column
                    html.Div([

                        # Color menu dropdowns
                        html.Div([

                            html.Div([
                                dcc.Dropdown(
                                    options=[
                                        {'label': 'Topic', 'value': 1, 'style': {'font-size': 18}},
                                        {'label': 'Sentiment', 'value': 2, 'style': {'font-size': 18}},
                                        {'label': 'Emotion', 'value': 3, 'style': {'font-size': 18}},
                                    ],
                                    value=1,
                                    clearable=False,
                                    placeholder="Color by",
                                    id='color-selection-dropdown'
                                ),
                            ], className='three columns'),
                            html.Div([
                                dcc.Dropdown(
                                    options=[
                                        {'label': 'Anger', 'value': 1, 'style': {'font-size': 18}},
                                        {'label': 'Disgust', 'value': 2, 'style': {'font-size': 18}},
                                        {'label': 'Fear', 'value': 3, 'style': {'font-size': 18}},
                                        {'label': 'Joy', 'value': 4, 'style': {'font-size': 18}},
                                        {'label': 'Sadness', 'value': 5, 'style': {'font-size': 18}}
                                    ],
                                    value=1,
                                    clearable=False,
                                    placeholder="Select emotion",
                                    disabled=True,
                                    id='emotion-color-dropdown'),
                            ], className='three columns'),

                        ], className='twelve columns'),

                        # 3d graph
                        html.Div([
                            dcc.Graph(id='scatter-3d', figure=get3dfigure(dell)),
                        ], className='twelve columns'),

                    ], className='nine columns'),

                    # right column
                    html.Div([

                        # Sort menu dropdowns
                        html.Div([

                            # topic/sentiment/emotion
                            html.Div([
                                dcc.Dropdown(
                                    options=[
                                        {'label': 'Topic', 'value': 1, 'style': {'font-size': 18}},
                                        {'label': 'Sentiment', 'value': 2, 'style': {'font-size': 18}},
                                        {'label': 'Emotion', 'value': 3, 'style': {'font-size': 18}},
                                    ],
                                    value=1,
                                    clearable=False,
                                    placeholder="Sort by",
                                    id='sort-selection-dropdown'
                                ),
                            ], className='six columns'),

                            # emotion
                            html.Div([
                                dcc.Dropdown(
                                    options=[
                                        {'label': 'Anger', 'value': 1, 'style': {'font-size': 18}},
                                        {'label': 'Disgust', 'value': 2, 'style': {'font-size': 18}},
                                        {'label': 'Fear', 'value': 3, 'style': {'font-size': 18}},
                                        {'label': 'Joy', 'value': 4, 'style': {'font-size': 18}},
                                        {'label': 'Sadness', 'value': 5, 'style': {'font-size': 18}}
                                    ],
                                    value=1,
                                    clearable=False,
                                    placeholder="Select emotion",
                                    disabled=True,
                                    id='emotion-sort-dropdown'),
                            ], className='six columns'),

                        ]),

                        # bar graph
                        html.Div([
                            dcc.Graph(id='bar-topic',
                                      figure=getBarFigure(data),
                                      config = {
                                          'displayModeBar': False,
                                          'showAxisDragHandles': False,
                                          'showAxisRangeEntryBoxes': False
                                      }),
                        ], className='twelve columns'),
                    ], className='three columns'),
                ], className='nine columns'),

            ], style={"width": '100%', 'display': 'inline-block'})
            # Horisontal rule

        ], style={'width': '100%', 'display': 'inline-block'}),

        # data table
        html.Div([
            dt.DataTable(
                # Initialise the rows
                rows=[{}],
                columns = table_nice_names,
                row_selectable=False,
                editable=False,
                filterable=True,
                sortable=True,
                selected_row_indices=[],
                id='table'
            ),
            html.Div(id='data-selected-indices')
        ], style={'width': '100%', 'display': 'inline-block'}),

        # Hidden div inside the app that stores intermediate data values
        html.Div(id='data-filtered-by-all', style={'display': 'none'}),
        html.Div('{"selected_topics": []}', id='data-selected-topics', style={'display': 'none'})

    ], style={'width': '90%', 'margin-left': 'auto', 'margin-right': 'auto'})

    ##
    ## Callback functions

    @app.callback(Output('table', 'rows'), [Input('data-filtered-by-all', 'children')])
    def update_table(json_data):

        df = pd.read_json(json_data, orient='split')[table_cols]
        df.columns = table_nice_names

        return df.to_dict('records')

    # disable/enable emotion dropdown
    @app.callback(
        Output(component_id='emotion-color-dropdown', component_property='disabled'),
        [Input(component_id='color-selection-dropdown', component_property='value')])
    def show_hide_emotion_color_dropdown(color_variable):
        if color_variable == 3:
            return False
        else:
            return True

    @app.callback(
        Output(component_id='emotion-sort-dropdown', component_property='disabled'),
        [Input(component_id='sort-selection-dropdown', component_property='value')])
    def show_hide_emotion_sort_dropdown(color_variable):
        if color_variable == 3:
            return False
        else:
            return True

    # filter operation
    @app.callback(
        Output('data-filtered-by-all', 'children'),
        [Input('hist-sentiment', 'selectedData'),
         Input('hist-emotion-anger', 'selectedData'),
         Input('hist-emotion-disgust', 'selectedData'),
         Input('hist-emotion-fear', 'selectedData'),
         Input('hist-emotion-joy', 'selectedData'),
         Input('hist-emotion-sadness', 'selectedData'),
         Input('data-selected-topics', 'children')]
    )
    def filter_data(selSent, selAng, selDgst, selFear, selJoy, selSad, jsonSelectedTopics):

        # get the necc data from the input selections and color dropdown
        rangeSent = selSent["range"]["x"] if selSent is not None else [-1,1]
        rangeAng = selAng["range"]["x"] if selAng is not None else [0,1]
        rangeDgst = selDgst["range"]["x"] if selDgst is not None else [0,1]
        rangeFear = selFear["range"]["x"] if selFear is not None else [0,1]
        rangeJoy = selJoy["range"]["x"] if selJoy is not None else [0,1]
        rangeSad = selSad["range"]["x"] if selSad is not None else [0,1]

        # filter the data
        isSent = data["watson_sentiment"].between(rangeSent[0], rangeSent[1])
        isAng = data['watson_emotion_anger'].between(rangeAng[0], rangeAng[1])
        isDgst = data['watson_emotion_disgust'].between(rangeDgst[0], rangeDgst[1])
        isFear = data['watson_emotion_fear'].between(rangeFear[0], rangeFear[1])
        isJoy = data['watson_emotion_joy'].between(rangeJoy[0], rangeJoy[1])
        isSad = data['watson_emotion_sadness'].between(rangeSad[0], rangeSad[1])

        # unpack the json data
        selTpcs = json.loads(jsonSelectedTopics)['selected_topics']
        if selTpcs is not None and len(selTpcs) > 0:
            isTpc = data['topic_code'].isin(selTpcs)
        else:
            isTpc = True

        data_f = data[isSent & isAng & isDgst & isFear & isJoy & isSad & isTpc]

        # return the figure
        return data_f.to_json(orient='split', date_format='iso')

    # update 3d plot;
    @app.callback(
        Output('scatter-3d', 'figure'),
        [Input('data-filtered-by-all', 'children'),
         Input('color-selection-dropdown', 'value'),
         Input('emotion-color-dropdown', 'value'),
         Input('data-selected-topics', 'children')],
        [State('scatter-3d', 'relayoutData')],
    )
    def update3dScatter(json_data, colorVal, emotionVal, jsonSelectedTopics, relayout):

        # get figure properties to pass along to new figure
        camera = relayout['scene.camera'] if relayout is not None and 'scene.camera' in relayout.keys() else default_camera

        # unpack the json data
        data_f = pd.read_json(json_data, orient='split')



        #  Prep the data
        dell = get_data_elements(data_f, colorMap)

        # return the figure
        return get3dfigure(dell, colorVal, emotionVal, camera)

    # update topic bar
    @app.callback(
        Output('bar-topic', 'figure'),
        [Input('sort-selection-dropdown', 'value'),
         Input('emotion-sort-dropdown', 'value'),
         Input('data-selected-topics', 'children')]
    )
    def updateTopicBar(sortVar, sortEmot, jsonSelectedTopics):
        # unpack the json data
        selTpcs = json.loads(jsonSelectedTopics)['selected_topics']

        # return the figure
        return getBarFigure(data, type=sortVar, subtype=sortEmot, selectedTopics=selTpcs)

    # select topics
    @app.callback(
        Output('data-selected-topics', 'children'),
        [Input('bar-topic', 'clickData')],
        [State('data-selected-topics', 'children')]
    )
    def addSelectedTopic(clickData, jsonSelectedTopics):

        # unpack the json data
        oldTpcs = json.loads(jsonSelectedTopics)['selected_topics']
        newTpcs = [bar['y'] for bar in clickData['points']] if clickData is not None else []
        oldTpcsKeep = [tpc for tpc in oldTpcs if tpc not in newTpcs]
        newTpcsKeep = [tpc for tpc in newTpcs if tpc not in oldTpcs]

        newSelTpcs = oldTpcsKeep + newTpcsKeep

        # return the updated list
        return json.dumps({"selected_topics": newSelTpcs})



    return app







if __name__ == '__main__':
    # build the dashboard
    app = configure_dashboard(data_elements)

    # the port
    port = int(os.getenv('PORT', 8000))

    # run the server
    app.run_server(host='0.0.0.0', port=port, debug=True)