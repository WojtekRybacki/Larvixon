import torch
import cv2
import numpy as np
import base64
import matplotlib as plt
import time
import plotly.graph_objs as go
import os
from PIL import Image
import io

def get_ai_response(vid_path="videos/GP032995_1.MP4", confidence_lvl=0.3, filename="test.png", break_time=20, color_changes_time=1800):
    path = os.path.join(os.getcwd(), 'yolov5', 'runs', 'train', 'exp2', 'weights', 'best.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)

    cap = cv2.VideoCapture(vid_path)
    
    results_frames = [] 
    trajectory = []
    trajectory_graph = []
    color_list = [(0, 0, 155),(0, 155, 155),(155, 155, 0),(155, 0, 255),(155, 155, 255),(100, 155, 100),(0, 0, 255),(255, 0, 0),(0, 255, 0)]
    trajectory_colors = []
    start_time = time.time()
    last_frame = None

    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('video.mp4', fourcc, 30.0, (1920, 540))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if time.time() - start_time > break_time:
            break
        
        results = model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        for i in range(len(labels)):
            if int(labels[i]) == 0 or int(labels[i]) == 1:  
                x1, y1, x2, y2, conf = cord[i]
                ##print(conf)
                if conf > confidence_lvl:  #detekcja
                    #center to picture
                    x1_graph, y1_graph, x2_graph, y2_graph = float(x1), float(y1), float(x2), float(y2)
                    center_graph = (float((x1_graph + x2_graph) / 2), float((y1_graph + y2_graph) / 2))
                    
                    h, w, _ = frame.shape
                    x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    
                    trajectory_graph.append(center_graph)
                    trajectory.append(center)

                    #kwadrat
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
        # Dynamic color change based on time
        elapsed_time = time.time() - start_time
        color_index = int(elapsed_time // color_changes_time) % len(color_list)
        current_color = color_list[color_index]
        
        #trajektoria
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], current_color, 2)
        
        if trajectory:
            trajectory_colors.append(current_color)
            
        
        out.write(frame)
        
        #last frame
        last_frame = frame
        

    cap.release() 
    out.release()
    cv2.destroyAllWindows()
    
    x,y = zip(*trajectory_graph)
    plot_colors = [f'rgb({r},{g},{b})' for r, g, b in trajectory_colors]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers+lines', 
                             marker=dict(color=plot_colors), 
                             line=dict(color=plot_colors[-1]),
                             name='Trajectory'))
    fig.update_layout(
    title='Trajectory of Detected Objects',
    xaxis_title='X Position',
    yaxis_title='Y Position',
    margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.write_image(f"plots/{filename}.png")
        
    pil_image = Image.fromarray(last_frame)
    buff = io.BytesIO()
    pil_image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        
        
    return fig, img_str