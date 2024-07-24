import pandas as pd
import plotly.graph_objects as go

#load the dataset
file_path = "core_landmarks_MP_image_data_for_scatter_plot.csv"
data = pd.read_csv(file_path)

pose_landmarks_dict = {'cat pose' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                       'chair pose' : [66, 67, 68, 69, 70, 71, 72, 73, 74, 75],
                       'cow pose' : [220, 221, 222, 223, 224, 225, 226, 227, 228, 229],
                       'downward dog pose' : [278, 279, 280, 281, 282, 283, 284, 285, 286, 287],
                       'flat back pose' : [358, 359, 360, 361, 362, 363, 364, 365, 366, 367],
                       'high lunge pose' : [437, 438, 439, 440, 441, 442, 443, 444, 445, 446],
                       'knee to elbow plank pose' : [543, 544, 546, 547, 548, 549, 550, 551, 552, 553],
                       'knee to chest pose' : [569, 570, 571, 572, 573, 574, 575, 576, 577, 578],
                       'low lunge pose' : [638, 639, 640, 641, 642, 643, 644, 645, 646, 647],
                       'mountain pose' : [718, 719, 720, 721, 722, 723, 724, 725, 726, 727],
                       'runners lunge pose' : [877, 878, 879, 880, 881, 882, 883, 884, 885, 886],
                       'seated spinal twist pose' : [937, 938, 939, 940, 941, 942, 943, 944, 945, 946],
                       'side plank pose' : [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022],
                       'standing forward fold pose' : [1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114],
                       'table top pose' : [1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214],
                       'three legged dog pose' : [1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266],
                       'tip toe pose' : [1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337],
                       'tree pose' : [1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377],
                       'upper plank pose' : [1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457]}


                       

#select key/value pair from dict pose examples
example_indices = pose_landmarks_dict['cat pose']
examples = data.iloc[example_indices]

#create scatter plot for Plotly with custom hover info
def plotly_landmarks_with_hover(data, color, name):
    #extract coordinates
    x = data.values[::3]
    y = data.values[1::3]
    z = data.values[2::3]
    
    #prepare hover text: include column headers and values
    hover_text = [f"{data.index[i*3]}: {data.values[i*3]:.2f}, {data.index[i*3+1]}: {data.values[i*3+1]:.2f}, {data.index[i*3+2]}: {data.values[i*3+2]:.2f}" for i in range(len(x))]

    #create a trace
    trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=4, color=color, opacity=0.8),
        name=name,
        text=hover_text,  #set hover text
        hoverinfo='text'
    )
    return trace

#create traces with customized hover text
trace1 = plotly_landmarks_with_hover(examples.iloc[0], 'red', 'Example 1')
trace2 = plotly_landmarks_with_hover(examples.iloc[1], 'green', 'Example 2')
trace3 = plotly_landmarks_with_hover(examples.iloc[2], 'blue', 'Example 3')
trace4 = plotly_landmarks_with_hover(examples.iloc[3], 'orange', 'Example 4')
trace5 = plotly_landmarks_with_hover(examples.iloc[4], 'teal', 'Example 5')
trace6 = plotly_landmarks_with_hover(examples.iloc[5], 'purple', 'Example 6')
trace7 = plotly_landmarks_with_hover(examples.iloc[6], 'lime', 'Example 7')
trace8 = plotly_landmarks_with_hover(examples.iloc[7], 'magenta', 'Example 8')
trace9 = plotly_landmarks_with_hover(examples.iloc[8], 'brown', 'Example 9')
trace10 = plotly_landmarks_with_hover(examples.iloc[9], 'cyan', 'Example 10')

                                     
#create a layout
layout = go.Layout(
    title='3D Scatter Plot of Landmarks with Custom Hover Text',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    margin=dict(l=0, r=0, b=0, t=0)  #adjust margins to fit everything
)

#create the figure
fig = go.Figure(data=[trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10], layout=layout)

# Show the plot
fig.show()
