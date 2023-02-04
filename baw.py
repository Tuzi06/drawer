import cv2
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont

def process(img):
    c = 10
    hs = 1.23
    if img is None: return None
    img = cv2.resize(img,(len(img[0])//c,len(img)//c))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    med_val = np.median(gray) 
    (thresh, baw) = cv2.threshold(gray, med_val, 255, cv2.THRESH_BINARY)
    med_val = np.median(baw) 
    lower = int(max(0 ,0.7*thresh))
    upper = int(min(255,1.3*thresh))
    edge = cv2.Canny(baw,lower,upper)
    # edge =(255-edge)
    # output = Image.fromarray(edge)
    # return output

    df = pd.DataFrame(edge)
    output = Image.new('RGB',(c*(len(df.columns)),int(c*hs)*(len(df))),color = (0,0,0))
    d = ImageDraw.Draw(output)
    fnt = ImageFont.truetype('./data/ARIBL0.ttf',c)
    o = ''
    ch = ['- ','@']
    for i in range(len(df)):
        for j in range(len(df.columns)):
            o+=(ch[edge[i][j]//255]+' ')
        o+='\n'
    d.multiline_text((0,0),o,font = fnt,fill=(255,255,255),spacing = 0,)
    return output

def out(frames,i):
    fs = (1920,1080)
    out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'),30, fs)
    for j in range(i):
        source = cv2.resize(frames[j],fs)
        # cv2.imshow('asf',source)
        cv2.waitKey(1)
        out.write(source)
        print(j)
    out.release()

import time

def main():
    start =  time.perf_counter()
    cap = cv2.VideoCapture('./data/vergill4.mp4')
    i=0
    frames = []
    print('start')
    while(cap.isOpened()):
        ret,frame =cap.read()
        newFrame = process(frame)
        if newFrame is None or i>=0:
            break
        newFrame = np.asarray(newFrame)
        # newFrame = cv2.cvtColor(newFrame,cv2.COLOR_GRAY2BGR)
        frames.append(newFrame)
        print('finish ',i)
        i+=1
    out(frames,i)
    end =  time.perf_counter()
    print(end-start)

if __name__ =='__main__':
    main()