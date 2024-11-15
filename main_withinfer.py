import streamlit as st
import os
import torch
import torch.nn as nn 
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import cv2
import time
import argparse
from tqdm import tqdm
from rtdetr.tools.infer import InitArgs, draw, initModel

def main():
    st.title('無人機人員偵測系統')
    
    # upload the file
    uploaded_file = st.file_uploader("請上傳影片", type=['mp4'])
    
    if uploaded_file is not None:
        upload_success = st.success("影片已成功上傳！")
        st.video(uploaded_file)

        video_correct = st.checkbox("請確認影片是否正確")
        
        if video_correct:
            # create dir of to save the input file and inference outcome
            video_name = uploaded_file.name.split('.')[0]
            os.makedirs(f"inputFile/{video_name}", exist_ok=True)
            output_path = f"outputFile/{video_name}"
            os.makedirs(output_path, exist_ok=True)

            # copy the video to inputFile
            save_path = f"inputFile/{video_name}/{uploaded_file.name}"
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Create the parser for inference and init the model
            # The reason I put the initialize and model here \
            # is the time for inferecne will be shorter
            args = InitArgs(save_path, True, output_path)
            model = initModel(args)
            # close the success message    
            upload_success.empty()
            
            save_success = st.success(f"影片已儲存至 {save_path}")
            time.sleep(3)
            save_success.empty()
            
            # Start to inference
            detect_annotation = infer(args, model)
            
# '''
# Infer function : 
#     parameters: 
#         args:  paramerters for model initialize, including the path for 
#                input and output file and type of data
#         model: use for inference
#     function:
#         1. Inference the data from user input 
#         2. The interrupt button to stop the inference
#     To do:
#         If the video is too long maybe we can crop the video and than start to inference
# '''
def infer(args, model):
    infer_correct = st.checkbox("是否執行Inference")     
    detect_annotation = []
    if infer_correct:
        cap = cv2.VideoCapture(args.imfile)
        
        # get the fps, w, h, if the input video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # set output video type .mp4
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        output_video = cv2.VideoWriter(os.path.join(args.outputdir,"output.mp4"), fourcc, fps, (width, height))
        
        # Initialize the progress bar with the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)  # Progress bar initialized at 0%
        
        # Create a button to interrupt the inference
        interrupt_button = st.button('中斷推理')  
        is_interrupted = False
        
        if not cap.isOpened():
            print("cap can not open")
            exit()

        while cap.isOpened():
            
                ret, frame = cap.read()
                if not ret:
                    print("Frame end or can not read frame")
                    exit()
                if interrupt_button:
                    is_interrupted = True
                    st.warning("推理已中斷！")
                    break  # Break the loop to stop the inference
                    
                # change the graph type from bgr to rgb 
                im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                w, h = im_pil.size
                orig_size = torch.tensor([w, h])[None].to(args.device)
            
                # Resize the graph and change to tensor type to inference
                transforms = T.Compose([
                    T.Resize((640, 640)),  
                    T.ToTensor(),
                ])
                im_data = transforms(im_pil)[None].to(args.device)
                    
                output = model(im_data, orig_size)
                labels, boxes, scores = output
                    
                detect_frame, box_count = draw([im_pil], labels, boxes, scores, 0.5)
                
                frame_out = cv2.cvtColor(np.array(detect_frame), cv2.COLOR_RGB2BGR)
                output_video.write(frame_out)
                
                # Update the progress bar
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                progress = current_frame / total_frames
                progress_bar.progress(progress)  # Update progress bar

                # Collect the frame that is detected
                if  box_count > 0:
                    detect_annotation.append(current_frame)
                
        # close all the windows
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()
        if is_interrupted:
            st.info("推理過程已停止。")
        else:
            st.success("推理完成")
        return detect_annotation
# '''
# Evaluation function:
#     To do: 
#         I am still thinking is there any more functionalities that can help
#         firfighters to have a better working flow
# '''           
def Evaluation():
    st.title("登入")  
    
    
if __name__ == '__main__':
    st.sidebar.title("功能分頁")
    page = st.sidebar.selectbox("選擇頁面", ("推理", "結果分析"))
    if page == "推理":
        main()
    elif page == "結果分析":
        Evaluation()
