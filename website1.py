
import dash
from dash import dcc
from dash import html,dash_table
import feffery_antd_components as fac
from dash.dependencies import Input, Output,State
import plotly.graph_objs as go
from tabulate import tabulate
import plotly_express as px
from prettytable import PrettyTable
import plotly.offline as pyo
from datetime import datetime
import pandas as pd

#%%

# 加载数据
def back_trader(df,tick,scale):
    """

    :param df:  the column order should: time, adj_price, my_adj_price
    :return:
    """
    opening_option = []
    pnl_accum_final1 = 0  # the profit of my price
    pnl_accum_final2 = 0  # the profit of adj price
    pnl_delta = 0


    columns_name = ['time', 'trade_judge', 'pnl_calculate', 'pnl_accum_final1', 'pnl_accum_final2', 'pnl_delta', 'adj_P', 'my_adj_P','adj_state','my_adj_state']
    res = pd.DataFrame(columns=columns_name)
    threshold = tick*scale
    for index,row in df.iterrows():

        if len(opening_option) == 0: # judge the state of opening option,we assume just one option opening at the same time
            pnl_calculate = 0
            state = []
            if  row.iloc[1]-row.iloc[2] >threshold:  #Meet the opening conditions
                state.append(-1) # open
                state.append(pnl_calculate)
                state.append(pnl_accum_final1)
                state.append(pnl_accum_final2)
                state.append(pnl_delta)
                state = state+list(row[1:3])
                state = [row.iloc[0]]+state
                state.append('开1')
                state.append('开-1')
                opening_option.append(state)
            if row.iloc[2] - row.iloc[1] > threshold:
                state.append(1)
                state.append(pnl_calculate)
                state.append(pnl_accum_final1)
                state.append(pnl_accum_final2)
                state.append(pnl_delta)
                state = state + list(row[1:3])
                state = [row.iloc[0]] + state
                state.append('开-1')
                state.append('开1')
                opening_option.append(state)
        elif len(opening_option) == 1: # close

            if opening_option[0][1] ==1 :
                state = []

                # if (row.iloc[2] < row.iloc[1]) or (index == df.shape[0]-1) :
                ## 用下一幅adj平仓
                exercise_price = row.iloc[1]
                #exercise_price = (row.iloc[1]+row.iloc[2])/2
                pnl_calculate = exercise_price-(opening_option[0][7] +opening_option[0][6])/2 # long at my price
                pnl_accum_final1 += pnl_calculate
                pnl_accum_final2 += ((opening_option[0][7] +opening_option[0][6])/2-exercise_price) # short at adj_price
                pnl_delta = pnl_accum_final1-pnl_accum_final2
                state.append(-1)
                state.append(pnl_calculate)
                state.append(pnl_accum_final1)
                state.append(pnl_accum_final2)

                state.append(pnl_delta)

                state = state + list(row[1:3])
                state = [row.iloc[0]] + state
                state.append('平1')
                state.append('平-1')
                opening_option.append(state)
            if opening_option[0][1] == -1:
                state = []
                # if (row.iloc[2] > row.iloc[1]) or (index == df.shape[0]-1) :
                # 用下一幅adj_p平
                exercise_price = row.iloc[1]
                #exercise_price = (row.iloc[1] + row.iloc[2]) / 2
                pnl_calculate =  (opening_option[0][7] +opening_option[0][6])/2-exercise_price # short at my price
                pnl_accum_final1 += pnl_calculate
                pnl_accum_final2 += (exercise_price - (opening_option[0][7] +opening_option[0][6])/2)
                pnl_delta = pnl_accum_final1 - pnl_accum_final2
                state.append(1)
                state.append(pnl_calculate)
                state.append(pnl_accum_final1)
                state.append(pnl_accum_final2)
                state.append(pnl_delta)
                state = state + list(row[1:3])
                state = [row.iloc[0]] + state
                state.append('平-1')
                state.append('平1')
                opening_option.append(state)

        if len(opening_option) ==2 :
           #print(opening_option)
            df1 = pd.DataFrame(opening_option,columns=columns_name)
            res =pd.concat([res,df1])
            opening_option =[]
    a = res[res.pnl_calculate>0].shape[0]
    b = res[res.pnl_calculate<0].shape[0]
    print('win_num',a)
    print('lose_num',b)
    print('win ratio',a/(b+a) if (b!=0 or a!=0) else 0)
    return res,(a/(b+a) if (a!=0 or b!=0) else 0),a,b
def back_test_plot(s_time,s_adj_price,s_pred_price,tick,scale):
    """

    :param s_time: (series)
    :param s_adj_price: (series) true_price
    :param s_pred_price: (series) predicted price
    :param tick: (int) ticksize
    :param scale: (float) 开仓阈值= scale*ticksize

    :return:
    """
    column_name = ['time', 'adj_price', 'pred_price']
    dict_df = {

        column_name[0]: s_time,
        column_name[1]: s_adj_price,
        column_name[2]: s_pred_price
    }
    data_combined = pd.DataFrame(dict_df)
    #data_combined['time'] = data_combined['time'].astype(object)
    #data_combined['time'] = data_combined['time'].apply(lambda x: datetime.strptime(x, '%Y%m%d %H:%M:%S.%f'))

    win_ratio_res = pd.DataFrame(columns=['time', 'win_num', 'loss_num', 'win_ratio'])
    daily_gorups = data_combined.groupby(data_combined['time'].dt.date)
    for date, group in daily_gorups:
        df_daily_night2 = group[(group['time'].dt.hour>=0) & (group['time'].dt.hour<9)]
        df_daily_night1 = group[(group['time'].dt.hour >= 21) & (group['time'].dt.time <pd.to_datetime('23:59:59').time())]
        df_daily_afternoon = group[(group['time'].dt.hour >= 13) & (group['time'].dt.hour < 15)]
        df_daily_morning = group[
            (group['time'].dt.time <= pd.to_datetime('10:15:00').time()) & (group['time'].dt.hour >= 9)]
        df_daily_beforenoon = group[
            (group['time'].dt.time > pd.to_datetime('10:30:00').time()) & (group['time'].dt.hour <= 12)]
        df_list = [df_daily_morning,df_daily_beforenoon,df_daily_afternoon,df_daily_night1,df_daily_night2]
        time_label_en = ['morning','beforenoon','afternoon', 'night1','night2']
        time_label_cn = ['早盘','上午盘','下午盘','夜盘1','夜盘2']
        fig_list = []
        back_trader_list =[]
        for i in range(len(df_list)):
            df_1= df_list[i]
            if i == 0 :
                label_en = time_label_en[i]
                label_cn = time_label_cn[i]
            elif i == 1 :
                label_en = time_label_en[i]
                label_cn = time_label_cn[i]
            elif i == 2 :
                label_en = time_label_en[i]
                label_cn = time_label_cn[i]
            elif i == 3 :
                label_en = time_label_en[i]
                label_cn = time_label_cn[i]
            elif i == 4 :
                label_en = time_label_en[i]
                label_cn = time_label_cn[i]
            back_trader_df,win_ratio,win_num,loss_num = back_trader(df_1, tick, scale)
            back_trader_list.append(back_trader_df)
            win_ratio_res.loc[len(win_ratio_res)] = [str(date) + str(label_en), win_num, loss_num, win_ratio]
            if back_trader_df.shape[0] !=0:
                back_tarder_price_series = pd.merge(df_1, back_trader_df, how='left', on='time')

            else:
                back_tarder_price_series = pd.concat([df_1, back_trader_df], axis=1)


            # print(back_tarder_price_series)
            x_value = back_tarder_price_series['time']
            y_adj_price = back_tarder_price_series['adj_price']
            y_my_price = back_tarder_price_series['pred_price']
            text_adj = back_tarder_price_series['adj_state']
            text_my_adj = back_tarder_price_series['my_adj_state']
            trace1 = go.Scatter(x=x_value, y=y_my_price, text=text_my_adj, mode='lines+text', name='my_adj_price')
            trace2 = go.Scatter(x=x_value, y=y_adj_price, text=text_adj, mode='lines+text', name='adj_price')

            layout = go.Layout(title=f'{str(date)}'+str(label_cn)+'价格曲线',
                               xaxis=dict(title='time'),
                               yaxis=dict(title='Price',exponentformat='none')
                               )
            # 将图表数据和布局合并为一个图表对象
            fig = go.Figure(data=[trace1, trace2], layout=layout)

            # 将图表保存为 HTML 文件
            pyo.plot(fig, filename=f'./output/b回测价格序列{str(date)}' +str(label_cn)+ '.html', auto_open=False)
            fig_list.append(fig)



    print(tabulate(win_ratio_res, headers='keys', tablefmt='fancy_grid'))
    return fig_list,back_trader_list

df1 = pd.read_csv('dash/df.csv')
tick = min(df1['ask1']-df1['bid1'])
df1['time'] = df1['time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))

df1['time'].dt.date.unique()


lob = df1.copy()
#%%
# 将数据转换为DataFrame


app = dash.Dash(__name__)

app.layout = html.Div(style={'display': 'flex', 'flexDirection': 'column'}, children=[
    html.Div(
        [

            fac.AntdDivider(isDashed=True),
            fac.AntdForm(
                [
                    fac.AntdFormItem(
                        fac.AntdSelect(
                            id='field1-range',
                            options=[
                                {
                                    'label': x,
                                    'value': x
                                }
                                for x in df1['time'].dt.date.unique()
                            ],
                            # mode='multiple',
                            maxTagCount='responsive',
                            style={
                                'width': 200
                            }
                        ),
                        label='选择日期'
                    ),
                    fac.AntdFormItem(
                        fac.AntdSelect(
                            id='field2-range',
                            options=[
                                {'label': '早盘', 'value': 0},
                                {'label': '上午盘', 'value': 1},
                                {'label': '下午盘', 'value': 2},
                                {'label': '夜盘', 'value': 3}
                            ],
                            # mode='multiple',
                            maxTagCount='responsive',
                            style={
                                'width': 200
                            }
                        ),
                        label='选择时段'
                    ),
                    fac.AntdInput(
                        placeholder='输入开仓阈值（阈值=input*ticksize）',
                        id = 'field3-range',
                        style={
                            'width': 200
                        },

                    ),
                    fac.AntdButton(
                        '查询',
                        id='execute-query',
                        icon=fac.AntdIcon(
                            icon='antd-search'
                        ),
                        type='primary'
                    )
                ],
                layout='inline',
                style={
                    'marginBottom': 15
                }
            ),
            html.Div(id ='table-result-container',style={'display': 'flex'}),
        ],
    ),


    html.Div(style={'marginTop': '5px'}, children=[
        dcc.Graph(id='lob-chart', style={'height': '300px'}),
        html.H4('点击订单簿前后10幅数据'),
        html.Div(id='all-order-book'),
        html.H4('所有回测成交记录'),
        html.Div(id= 'back-trader')
    ])

])

@app.callback(
    Output('table-result-container', 'children'),
    Input('execute-query', 'nClicks'),
    [State('field1-range', 'value'),
     State('field2-range', 'value'),
     State('field3-range', 'value'),
     ]
)
def update_table_data(nClicks,field1_range, field2_range,field3_range):

    df_tem = df1.copy()
    if field1_range and field2_range and field3_range :
        print('para:',field1_range, field2_range, field3_range)
        df_tem = df_tem[df_tem['time'].dt.date == pd.to_datetime(str(field1_range)).date()]
        # print('fie1:',pd.to_datetime(str(field1_range)).date)
        # print('df:',df_tem.head())
        fig_list,back_trade_list = back_test_plot(df_tem['time'],df_tem['adjusted_price'],df_tem['pred_y'],tick,float(field3_range))
        # print(fig_list)
        fig = fig_list[field2_range]
    fig_out = [html.Div(style={'flex': '70%'}, children=[
        dcc.Graph(
            id='price-chart',
            figure= fig
        ),
    ]),

    html.Div(style={'flex': '30%','marginTop': '200px'}, children=[
        html.Button(
            'Last tick',
            id = 'last-tick',
            n_clicks=0,

        ),
        html.Button(
            'Next tick',
            id = 'next-tick',
            n_clicks=0,
        ),
        html.Div(id='order-book', children='点击图中的点以查看订单簿数据')
    ])]
    global back_trade_df
    back_trade_df = back_trade_list[field2_range]

    return fig_out

clickData_list  = []

@app.callback(
    [Output('last-tick', 'n_clicks'),
    Output('next-tick', 'n_clicks')],
    Input('price-chart', 'clickData'),
)

def reset(clickData):
    if clickData is not None:
        return 0,0

@app.callback(
    Output('order-book', 'children'),
    [Input('price-chart', 'clickData'),
     Input('last-tick', 'n_clicks'),
     Input('next-tick', 'n_clicks'),
     ]
)
def display_order_book(clickData,Last_tick,next_tick):
    print(clickData)
    df_tem = df1.copy()
    if clickData is None:
        return '点击图中的点以查看订单簿数据'
    # 用时间作为index
    point_index = clickData['points'][0]['x']
    #准备数据
    # 由于 point_index 的时间是四位小数（不知道为什么），而df中时间是六位小数，所以只能用大于小于筛选，点击时刻在lob_after第一行
    idx=abs(df_tem['time']- pd.to_datetime(point_index)).idxmin()
    lob_after = df_tem.iloc[idx:,:]
    lob_before = df_tem.iloc[:idx,:]
    # 判断是否点击clickdate ,若点击则清空last 和next
    # clickData_list.append(clickData)
    # if len(clickData_list) >= 2:
    #     if clickData_list[-1] != clickData_list[-2]:
    #         Last_tick =0
    #         next_tick = 0

    # 提取当前带点击的信息
    act_num = next_tick - Last_tick
    if act_num >=0:
        one_lob = lob_after.iloc[act_num,:]
    else:
        one_lob = lob_before.iloc[act_num,:]

    # print(one_lob)
    col_name = ['level','bid','b_vol','ask','a_vol']
    levels = []
    bids = []
    bids_volumes = []
    asks= []
    asks_volumes = []
    for i in range(1,6):
        levels.append(i)
        bids.append(one_lob[f'bid{i}'])
        bids_volumes.append(one_lob[f'bid_vol{i}'])
        asks.append(one_lob[f'ask{i}'])
        asks_volumes.append(one_lob[f'ask_vol{i}'])
    # 做成dict
    lob_dict = {
        col_name[0]: levels,
        col_name[1]: bids,
        col_name[2]: bids_volumes,
        col_name[3]: asks,
        col_name[4]: asks_volumes
    }

    df = pd.DataFrame(lob_dict)
    # print(lob_dict)
    order_book = [
        dash_table.DataTable(
            df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]
        ),
        # 可以加按条件显示单元格

    ]
    return order_book



@app.callback(
    [Output('lob-chart', 'figure'),
     Output('all-order-book', 'children'),
     Output('back-trader', 'children')],
    [Input('price-chart', 'clickData'),
     Input('last-tick', 'n_clicks'),
     Input('next-tick', 'n_clicks'),
     ]
)
def update_order_book_and_table(clickData,Last_tick,next_tick):
    df_tem = df1.copy()
    if clickData is None:
        return '点击图中的点以查看订单簿数据'
    # 用时间作为index
    point_index = clickData['points'][0]['x']
    # 准备数据
    # 由于 point_index 的时间是四位小数（不知道为什么），而df中时间是六位小数，所以只能用大于小于筛选，点击时刻在lob_after第一行
    idx=abs(df_tem['time']- pd.to_datetime(point_index)).idxmin()
    lob_after = df_tem.iloc[idx:,:]
    lob_before = df_tem.iloc[:idx,:]
    # clickData_list.append(clickData)
    # if len(clickData_list) >= 2:
    #     if clickData_list[-1] != clickData_list[-2]:
    #         Last_tick =0
    #         next_tick = 0
    # 提取当前带点击的信息
    act_num = next_tick - Last_tick
    if act_num >=0:
        one_lob = lob_after.iloc[act_num,:]
    else:
        one_lob = lob_before.iloc[act_num,:]

    lob_fig_1 = df_tem[df_tem.index == one_lob.name]
    lob_fig_2 = df_tem[df_tem.index == one_lob.name+1]
    lob_fig_0 = df_tem[df_tem.index == one_lob.name-1]
    lob_fig_tem = pd.concat([lob_fig_0,lob_fig_1])
    lob_fig = pd.concat([lob_fig_tem,lob_fig_2])
    # lob_fig = pd.concat([lob_before.iloc[-1:, :], lob_after.iloc[:2, :]])
    lob_fig.reset_index(inplace=True)
    col_name = ['time','price','volume','type']
    time_list = []
    price_list = []
    volume_list = []
    type = []
    for index, row in lob_fig.iterrows():

        for i in range(1, 6):
            if index ==0:
                time_list.append('前一')
                time_list.append('前一')
            elif index ==1:
                time_list.append('当前')
                time_list.append('当前')
            else:
                time_list.append('后一')
                time_list.append('后一')
            price_list.append(row[f'bid{i}'])
            type.append('bid')
            price_list.append(row[f'ask{i}'])
            type.append('ask')
            volume_list.append(row[f'bid_vol{i}'])
            volume_list.append(row[f'ask_vol{i}'])
    bar_dict = {
        col_name[0]: time_list,
        col_name[1]: price_list,
        col_name[2]: volume_list,
        col_name[3]: type
    }
    df_bar = pd.DataFrame(bar_dict)

    fig = px.bar(df_bar,
                 x="price",
                 y="volume",
                 color="type",
                 facet_row="time"  # 行方向切面字段：
                 )
    fig.update_layout(xaxis_tickformat = '%d')

    lob_10 = df_tem[(df_tem.index <= one_lob.name+10) & (df_tem.index >= one_lob.name-10)]
    table = dash_table.DataTable(
                data=lob_10.to_dict('records'),
                columns=[
                    {"name": i, "id": i} for i in lob_10.columns
                ],
                style_data_conditional=[
                    {
                        'if': {
                            'filter_query': '{{time}} = {}'.format(str(one_lob['time']).replace(' ','T')),

                        },
                        'backgroundColor': '#FF4136',
                        'color': 'white'
                    },
                ]
            )
    back_trade_table = dash_table.DataTable(
                data=back_trade_df.to_dict('records'),
                columns=[
                    {"name": i, "id": i} for i in back_trade_df.columns
                ],
                style_data_conditional=[
                    {
                        'if': {
                            'filter_query': '{{time}} = {}'.format(str(one_lob['time']).replace(' ','T')),

                        },
                        'backgroundColor': '#FF4136',
                        'color': 'white'
                    },
                ]
            )

    # rows = []
    # for i in range(len(df)):
    #     row_style = {'backgroundColor': '#FFDDC1'} if i == point_index else {}
    #     row = html.Tr(style=row_style, children=[
    #         html.Td(df['price'][i]),
    #         html.Td(lob_data['level1'][i][0]), html.Td(lob_data['level1'][i][1]),
    #         html.Td(lob_data['level2'][i][0]), html.Td(lob_data['level2'][i][1]),
    #         html.Td(lob_data['level3'][i][0]), html.Td(lob_data['level3'][i][1]),
    #         html.Td(lob_data['level4'][i][0]), html.Td(lob_data['level4'][i][1]),
    #         html.Td(lob_data['level5'][i][0]), html.Td(lob_data['level5'][i][1])
    #     ])
    #     rows.append(row)
    #
    # header = html.Tr([
    #     html.Th('价格'),
    #     html.Th('Level1 价格'), html.Th('Level1 数量'),
    #     html.Th('Level2 价格'), html.Th('Level2 数量'),
    #     html.Th('Level3 价格'), html.Th('Level3 数量'),
    #     html.Th('Level4 价格'), html.Th('Level4 数量'),
    #     html.Th('Level5 价格'), html.Th('Level5 数量')
    # ])

    return fig, table,back_trade_table


if __name__ == '__main__':
    app.run_server(debug=True)


#%%
df_tem = pd.read_csv('dash/df.csv')
point_index = '2024-05-23 11:14:38.7972'
df_tem['time']=df_tem['time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
a=abs(df_tem['time']- pd.to_datetime(point_index)).idxmin()
lob_after = df_tem[df_tem['time'] > pd.to_datetime(point_index)]
lob_before = df_tem[df_tem['time'] < pd.to_datetime(point_index)]
# one_lob = lob_after.iloc[0, :]
# lob_fig = pd.concat([lob_before.iloc[-1:,:],lob_after.iloc[:2,:]])
# col_name = ['time','price','volume','type']
# time_list = []
# price_list = []
# volume_list = []
# type = []
# for index,row in lob_fig.iterrows():
#     for i in range(1,6):
#         time_list.append(row['time'])
#         time_list.append(row['time'])
#         price_list.append(row[f'bid{i}'])
#         type.append('bid')
#         price_list.append(row[f'ask{i}'])
#         type.append('ask')
#         volume_list.append(row[f'bid_vol{i}'])
#         volume_list.append(row[f'ask_vol{i}'])
# bar_dict = {
#     col_name[0]: time_list,
#     col_name[1]: price_list,
#     col_name[2]: volume_list,
#     col_name[3]: type
# }
# df_bar = pd.DataFrame(bar_dict)
# fig = px.bar(df_bar,
#              x="price",
#              y="volume",
#              color="type",
#              facet_row="time"  # 行方向切面字段：
#              )
# fig.show()
#%%