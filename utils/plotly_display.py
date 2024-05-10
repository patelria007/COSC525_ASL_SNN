'''
Credits to Dr. Charlie Rizzo from the University of Tennessee - Knoxville. 

This code was used as a reference for us to understand how to visualize 
the spatiotemporal data within the ASL-DVS dataset. Dr. Rizzo created 
a great script that generates HTML-style Plotly animations of the data.
Using Dr. Rizzo's code, we generated 2 animations and uploaded them
inside the `animations/` directory of this project.
'''

import csv
import sys
import os
import argparse
import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

class DVSDataset:
    dataset = []
    unique_frames = []
    ds_predictions = []
    num_unique_timestamps = 0

    def __init__(self,datafile):
        ''' Takes a file with event camera data and creates a numpy dataset of size (len(unique_frames),rows,cols). Replaces the '0' 
            polarity_decreasing value and the '1' polarity_increasing value with the specified argument values. Might be beneficial
            for machine learning. 
        Args:
            datafile (str): Name of h5 datafile with event camera events in repo's data/ directory
        Returns:
            Nothing
        '''
        df = pd.DataFrame()
        if str(datafile[-3:] == "csv"):
            with open(datafile,'r') as csv_in:
                csv_reader = csv.reader(csv_in)
                data = [[],[],[],[]] # Each entry in the array: timestamp, x, y, polarity.
                for row in csv_reader:   
                    if 'timestamp' in row[1]:
                        continue          

                    # Modification for ASL-DVS
                    data[0].append(int(row[1]))
                    data[1].append(int(row[2]))
                    data[2].append(int(row[3]))
                    data[3].append(int(row[4]))

            # Data formatted as an array of events.    
            df['timestamp'] = data[0] # timestamp in microsec
            df['x'] = data[1] # pixel position x
            df['y'] = data[2] # pixel position y
            df['polarity'] = data[3] # polarity
            
        else:
            '''
            with h5py.File(datafile, 'r') as f:
                data = np.array(f['events'])
            # Data formatted as an array of events.
            # Each entry in the array: timestamp, x, y, polarity.
            df['timestamp'] = data[:,0]
            df['x'] = data[:,1]
            df['y'] = data[:,2]
            df['polarity'] = data[:,3]
            '''
            print("Not made to work with h5 file atm")
        
        minClockTime = df['timestamp'][0]        
        df['relTime'] = df['timestamp'] - minClockTime # relative time 
        
        self.dataset = df.drop_duplicates()
        self.num_unique_timestamps = int(self.dataset['relTime'].iloc[-1]) #len(self.dataset['relTime'].unique().tolist())
        self.dataset = [self.dataset['x'].values.tolist(),self.dataset['y'].values.tolist(),self.dataset['relTime'].values.tolist(),self.dataset['polarity'].values.tolist()]

    def __str__(self) -> str:
        return "Dataset Shape: " + str(np.shape(self.dataset)) + "\n" 
    
    # For right now, we treat negative and positive events as the same
    def event_count_ds(self,tsl,chip_row,chip_col,stride,threshold,ds_type,crs,ccs,cre,cce):

        #Every downsampling pass for the entire dataset will be stored for easy access... might destroy memory, but oh well
        if len(self.ds_predictions) == 0:
            self.ds_predictions.append(np.zeros((math.ceil(self.num_unique_timestamps/tsl),math.ceil((cre - crs)/stride),math.ceil((cce - ccs)/stride))))
            downsampling_index = len(self.ds_predictions) - 1
            print("Downsampling %dx%d to %dx%d"%(cre-crs,cce-ccs,math.ceil((cre - crs)/stride),math.ceil((cce - ccs)/stride)))
        else:
            previous_downsampling_index = len(self.ds_predictions) - 1
            self.ds_predictions.append(np.zeros((math.ceil(self.num_unique_timestamps/tsl),math.ceil(len(self.ds_predictions[previous_downsampling_index][0])/stride),math.ceil(len(self.ds_predictions[previous_downsampling_index][0][0])/stride)))) 
            downsampling_index = len(self.ds_predictions) - 1
            print("Downsampling %dx%d to %dx%d"%(len(self.ds_predictions[previous_downsampling_index][0]),len(self.ds_predictions[previous_downsampling_index][0][0]),math.ceil(len(self.ds_predictions[previous_downsampling_index][0])/stride),math.ceil(len(self.ds_predictions[previous_downsampling_index][0][0])/stride)))

        start = 0
        stop = 0
        #Iterate through one "frame" or timeseries observation sample at a time, construct the frame, do the event counting
        for frame_index,frame in enumerate(range(0,self.num_unique_timestamps,tsl)):
            # If the downsampling_index is 0, we're using the dataframe of events as out source of events
            if downsampling_index == 0:
                #Determine the range from which to slice values from the numpy arrays
                for i in range(start,len(self.dataset[0])):
                    if self.dataset[2][i] >= frame + tsl:
                        stop = i
                        break

                # Stop is the index of the first element of the NEXT sample, but list slicing is exclusive
                # Grab events from timestamp frame to frame + tsl
                x_list = self.dataset[0][start:stop]
                y_list = self.dataset[1][start:stop]

                start = stop

                #Populate a temporary frame with the events (addition works fine for pixels that have multiple events)
                temp_frame = np.zeros((210,160))
                for i in range(len(x_list)):
                    temp_frame[int(y_list[i])][int(x_list[i])] += 1

                #Stride over the temp_frame and either do a sum of the entire chip to event_count or do a count_nonzero to do pixel_count
                for i,row in enumerate(range(crs,cre,stride)):
                    for j,col in enumerate(range(ccs,cce,stride)):
                        if ds_type == "ec":
                            if np.sum(temp_frame[row:row+chip_row,col:col+chip_col]) > threshold:
                                self.ds_predictions[downsampling_index][frame_index][i][j] = 1
                        elif ds_type == "pc":
                            if np.count_nonzero(temp_frame[row:row+chip_row,col:col+chip_col]) > threshold:
                                self.ds_predictions[downsampling_index][frame_index][i][j] = 1
                        else:
                            print("This is a bad case")
                            sys.exit()

            # If the downsampling_index is not 0, we use the previous self.ds_predictions[downsampling_index - 1] as the source of events
            else:
                #Stride over the temp_frame and either do a sum of the entire chip to event_count or do a count_nonzero to do pixel_count
                for i,row in enumerate(range(0,len(self.ds_predictions[downsampling_index - 1][0]),stride)):
                    for j,col in enumerate(range(0,len(self.ds_predictions[downsampling_index - 1][0][0]),stride)):
                        if ds_type == "ec":
                            if np.sum(self.ds_predictions[downsampling_index - 1][frame_index][row:row+chip_row,col:col+chip_col]) > threshold:
                                self.ds_predictions[downsampling_index][frame_index][i][j] = 1
                        elif ds_type == "pc":
                            if np.count_nonzero(self.ds_predictions[downsampling_index - 1][frame_index][row:row+chip_row,col:col+chip_col]) > threshold:
                                self.ds_predictions[downsampling_index][frame_index][i][j] = 1
                        else:
                            print("This is a bad case")
                            sys.exit() 
                #print(self.ds_predictions[downsampling_index][frame_index])




# Parameters for plotly slider
sliders_dict = {
    "active": 0,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
        "prefix": "Timestep:",
        "visible": True,
        "xanchor": "right"
    },
    "transition": {"duration": 5, "easing": "cubic-in-out"},
    "pad": {"b": 10, "t": 50},
    "len": 0.9,
    "x": 0.1,
    "y": 0,
    "steps": []
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="File Generator for Plotly with Network Outputs")
    parser.add_argument("--datafile","-d",required=True,type=str)
    parser.add_argument("--time_segmentation_length","-e",default=1,type=int)
    parser.add_argument("--chip_sizes","-c",nargs='*',type=int,help="Chip sizes specified as: row col")
    parser.add_argument("--strides","-s",nargs='*',type=int, help="Stride values for each downsampling layer")
    parser.add_argument("--thresholds","-t",nargs='*',type=int, help="Threshold values to use for each downsampling layer")
    parser.add_argument("--ds_type","-dt",nargs="*",type=str,help="Specifies what kind of downsampling layer each layer is (e.g. event_count, pixel_count, track, etc.)")
    parser.add_argument("--camera_row_start","-crs",default=0,type=int)
    parser.add_argument("--camera_col_start","-ccs",default=0,type=int)
    parser.add_argument("--camera_row_end","-cre",default=180,type=int)
    parser.add_argument("--camera_col_end","-cce",default=240,type=int)
#    parser.add_argument("--network_tracking_file","-ntf",type=str,default="")
#    parser.add_argument("--tracking_threshold","-tt",default=2,type=int)
    parser.add_argument("--save_animation","-sa",default="",type=str)
    args = parser.parse_args()

    print("Event Based Camera Rows: %d Cols: %d" %(args.camera_row_end,args.camera_col_end))
    print("Assuming time_segmentation_length value of %d."%(args.time_segmentation_length))

    if (args.strides is not None and args.thresholds is not None and args.chip_sizes is not None and args.ds_type is not None):
        if (len(args.strides) != len(args.thresholds) and len(args.chip_sizes) != (2 * len(args.strides)) and len(args.ds_type) != len(args.strides)):
            print("error: args.strides and args.thresholds must be the same length. args.chip_sizes must be twice the length of those values")
            sys.exit()

    # Read in events from original h5 (or csv) file
    ds = DVSDataset(datafile=args.datafile)
    #print(ds)

    if args.strides is not None:
        for index,stride in enumerate(args.strides):
            if args.ds_type[index] == "ec" or args.ds_type[index] == "pc":
                ds.event_count_ds(args.time_segmentation_length,args.chip_sizes[index * 2],args.chip_sizes[index * 2 + 1],args.strides[index],args.thresholds[index],args.ds_type[index],args.camera_row_start,args.camera_col_start,args.camera_row_end,args.camera_col_end)

        # Draw boxes for the predictions... 
        predictions_dict = {}
        box_map = {}
        for stride_index,stride in enumerate(args.strides):
            for frame_index,frame in enumerate(range(0,ds.num_unique_timestamps,args.time_segmentation_length)):
                tmp_row = []
                tmp_col = []
                #print("Here 2")
                if stride_index == 0:
                    for i,row in enumerate(range(args.camera_row_start,args.camera_row_end,stride)):
                        for j,col in enumerate(range(args.camera_col_start,args.camera_col_end,stride)):
                            if ds.ds_predictions[stride_index][frame_index][i][j] == 1:
                                tmp_row.append([row,row + args.chip_sizes[stride_index * 2],row + args.chip_sizes[stride_index * 2],row,row,None])
                                tmp_col.append([col,col,col + args.chip_sizes[stride_index * 2 + 1], col + args.chip_sizes[stride_index * 2 + 1],col,None])
                else:
                    # Chip length is prev_chip_stride(current_chip_len - 1) + prev_chip_len
                    # The next starting position for a chip is current_chip_stride * prev_chip_stride 
                    if stride_index == 1:
                        n_prev_chips_row_length = (args.strides[stride_index-1] * (args.chip_sizes[stride_index * 2] - 1)) + args.chip_sizes[(stride_index - 1) * 2]
                        n_prev_chips_col_length = (args.strides[stride_index-1] * (args.chip_sizes[stride_index * 2 + 1] - 1)) + args.chip_sizes[(stride_index - 1) * 2 + 1]
                        start_pos = args.strides[stride_index - 1] * args.strides[stride_index]
                    else:
                        stride_factor = 1
                        for s_index in range(0, stride_index):
                            stride_factor *= args.strides[s_index]
                        n_prev_chips_row_length = (stride_factor * (args.chip_sizes[stride_index * 2] - 1)) + box_map[stride_index-1][0]
                        n_prev_chips_col_length = (stride_factor * (args.chip_sizes[stride_index * 2 + 1] - 1)) + box_map[stride_index-1][1]
                        start_pos = stride_factor * args.strides[stride_index]
                    
                    box_map[stride_index] = (n_prev_chips_row_length,n_prev_chips_col_length)

                    for i,row in enumerate(range(0,len(ds.ds_predictions[stride_index - 1][0]),stride)):
                        for j,col in enumerate(range(0,len(ds.ds_predictions[stride_index - 1][0][0]),stride)):
                            if ds.ds_predictions[stride_index][frame_index][i][j] == 1:
                                tmp_r = i * start_pos + args.camera_row_start
                                tmp_c = j * start_pos + args.camera_col_start
                                tmp_row.append([tmp_r,tmp_r + n_prev_chips_row_length,tmp_r + n_prev_chips_row_length,tmp_r,tmp_r,None])
                                tmp_col.append([tmp_c,tmp_c,tmp_c + n_prev_chips_col_length, tmp_c + n_prev_chips_col_length,tmp_c,None])

                if len(tmp_row) > 0 or len(tmp_col) > 0:
                    tmp_row = [element for sublist in tmp_row for element in sublist]
                    tmp_col = [element for sublist in tmp_col for element in sublist]
                    if frame not in predictions_dict.keys():
                        predictions_dict[frame] = [(tmp_col,tmp_row)]
                    else:
                        predictions_dict[frame].append((tmp_col,tmp_row))

    frames = []
    start = 0
    stop = 0
    '''
        Okay, so we round microsecond down to a tenth of a millisecond (I think). Then we buffer time_segmentation_length amount of of timesteps
        worth of events into one observation that we will display here, or show to the classifier as a timeseries observation. For classification,
        the relative times of events matter. For display, it doesn't matter since one timeseries sample will contain and spike in the events.
        What will likely happen is that the first pass of downsampling networks will buffer those values for time_segmentation_length cycles before 
        emitting synchronized spikes.
    '''
    for i in range(0,ds.num_unique_timestamps,args.time_segmentation_length):
        #Need to have sentinel event data to force legend to show up each frame...
        tmpX_pos = [-100]
        tmpY_pos = [-100]
        tmpX_neg = [-100]
        tmpY_neg = [-100]

        #Determine the range from which to slice values from the numpy arrays
        for j in range(start,len(ds.dataset[0])):
            if ds.dataset[2][j] >= i + args.time_segmentation_length:
                stop = j
                break

        # Stop is the index of the first element of the NEXT sample, but list slicing is exclusive
        # Grab events from timestamp frame to frame + tsl
        for j in range(start,stop):
            if ds.dataset[3][j] == 1:
                tmpX_pos.append(ds.dataset[0][j])
                tmpY_pos.append(ds.dataset[1][j])
            elif ds.dataset[3][j] == 0:
                tmpX_neg.append(ds.dataset[0][j])
                tmpY_neg.append(ds.dataset[1][j]) 

        start = stop

        pos_points = go.Scatter(x=tmpX_pos, 
                            y=tmpY_pos,
                            mode='markers',
                            name="Pos. Polarity Events",
                            showlegend=True,
                            marker=dict(
                                color='rgb(0,255,0)'
                            ))
   
        neg_points = go.Scatter(x=tmpX_neg, 
                            y=tmpY_neg,
                            mode='markers',
                            name="Neg. Polarity Events",
                            showlegend=True,
                            marker=dict(
                                color='rgb(255,0,0)'
                            ))

        frame_data = [pos_points,neg_points]
        if args.strides is not None:
            if len(args.strides) > 0: 
                # create box for frame if need be with sentinel box, like with events
                colors = ['green','blue','purple']
                sentinelX_box = [-100,-100,-150,-150,-100]
                sentinelY_box = [-100,-150,-150,-100,-100]
                if i in predictions_dict.keys():
                    for ds_levels in range(len(args.strides)):
                        if ds_levels < len(predictions_dict[i]):
                            tmpX_box = predictions_dict[i][ds_levels][0]
                            tmpY_box = predictions_dict[i][ds_levels][1]

                            pred_rects = go.Scatter(x=tmpX_box,y=tmpY_box,fill='toself',mode='lines',name="Ds %d - %dx%d,%d,%d"%(ds_levels+1,args.chip_sizes[ds_levels * 2],args.chip_sizes[ds_levels * 2 + 1],args.strides[ds_levels],args.thresholds[ds_levels]),opacity=0.5,showlegend=True,marker=dict(color=colors[ds_levels]))
                            frame_data.append(pred_rects)
                        else: #This is for cases where we need to redraw any successive downsampling boxes
                            pred_rects = go.Scatter(x=sentinelX_box,y=sentinelY_box,fill='toself',mode='lines',name="Ds %d - %dx%d,%d,%d"%(ds_levels+1,args.chip_sizes[ds_levels * 2],args.chip_sizes[ds_levels * 2 + 1],args.strides[ds_levels],args.thresholds[ds_levels]),opacity=0.5,showlegend=True,marker=dict(color=colors[ds_levels]))
                            frame_data.append(pred_rects)
                else: #This is for when there are no boxes that are supposed to be drawn on the frame. we need to draw them somewhere else to update the frame from the previously drawn frame's rectangles
                    for ds_levels in range(len(args.strides)): 
                        pred_rects = go.Scatter(x=sentinelX_box,y=sentinelY_box,fill='toself',mode='lines',name="Ds %d - %dx%d,%d,%d"%(ds_levels+1,args.chip_sizes[ds_levels * 2],args.chip_sizes[ds_levels * 2 + 1],args.strides[ds_levels],args.thresholds[ds_levels]),opacity=0.5,showlegend=True,marker=dict(color=colors[ds_levels]))
                        frame_data.append(pred_rects)   

        # create a frame object
        frame = go.Frame(
            data=frame_data, 
            name=f'frame{i/args.time_segmentation_length}'
        )

        # Create corresponding slider step in order to match each slider step with each frame
        slider_step = {"args": [
            [f'frame{i/args.time_segmentation_length}'],
            {"frame": {"duration": 0, "redraw": False},
            "mode": "immediate",
            "transition": {"duration": 5}}
        ],
            "label": i/args.time_segmentation_length,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)
        # add the frame object to the frames list
        frames.append(frame)



    # In sliders_dict, slider_step, and layout
    # 1.) Either frame:duration:0 and transition:duration:5
    # 2.) Or frame:duration:30 and transition:duration:5

    # Graph layout parameters
    layout = go.Layout(        
        xaxis=dict(range=[args.camera_col_start, args.camera_col_end], autorange=False),
        yaxis=dict(range=[args.camera_row_end, args.camera_row_start], autorange=False),
        title="Event Based Camera Plot of %s" %(args.datafile),
        #title="Event Based Camera Plot: frame_sampling = %d, chip_size = %s, stride = %s, threshold = %s, tracking_threshold = %s" 
        #    %(args.frame_sampling,"N/A" if args.first_pass_chipping_file == "" else "%dx%d"%(args.chip_size[0],args.chip_size[1]),"N/A" if args.first_pass_chipping_file == "" else str(args.stride),"N/A" if args.first_pass_chipping_file == "" else str(args.threshold),"N/A" if args.network_tracking_file == "" else str(args.tracking_threshold)),
        updatemenus=[dict(
            type="buttons",
            # create the button; make duration:0 for no animation between frames; Make duration:5 and redraw=True for smoother transitioning
            buttons=[
                {
                    "args": [None, {"frame": {"duration": 0, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 5,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],           
            direction= "left",
            pad={"r": 10, "t": 87},
            showactive= False,
            x= 0.1,
            xanchor= "right",
            y=0,
            yanchor="top"
        )],
        sliders=[sliders_dict],
    )


    fig = go.Figure(data=frame_data,frames=frames,layout=layout)

    # fig.show()
    if args.save_animation != "":
        fig.write_html(args.save_animation,auto_play=False)
        
