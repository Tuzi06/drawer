import cv2
import numpy as np
import pandas as pd
import threading
from PIL import Image,ImageDraw,ImageFont
import time

def process(img,i):
    # img = cv2.resize(img,(160,120))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    med_val = np.median(gray) 
    (thresh, baw) = cv2.threshold(gray, med_val, 255, cv2.THRESH_BINARY)
    med_val = np.median(baw) 
    lower = int(max(0 ,0.7*med_val))
    upper = int(min(255,1.3*med_val))
    edge = cv2.Canny(baw,lower,upper)
    df = pd.DataFrame(edge)
    width = 15
    height = 15
    output = Image.new('L',(width*len(df.columns),height*len(df)),color = (0))
    d = ImageDraw.Draw(output)
    fnt = ImageFont.truetype('./data/ARIBL0.ttf',15)
    for i in range(len(df)+1):
        for j in range(len(df.columns)+1):
            if i<len(df) and j<len(df.columns) and edge[i][j] == 255:
                s = '@'
            else:
                s = ' '
            d.text((j*width,i*height),s,font = fnt,fill = (255))
    # output = output.resize((1280,720))
    return output
    output.save('./data/res/res-%i.jpg'%i)
    print('finish ',i)
    

def out(i):
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'DIVX'),30, (1280,720))
    for j in range(i):
        source = cv2.imread('./data/res/res-%i.png'%j)
        source = cv2.resize(source,(1280,720))
        cv2.imshow('asf',source)
        cv2.waitKey(1)
        out.write(source)
        print(j)
    out.release()

def main():
    cap = cv2.VideoCapture('./data/vergill.mp4')
    i=0
    threads = []
    while(cap.isOpened()):
        ret,frame =cap.read()
        # threads.append(threading.Thread(target=process, args= (frame,i)))
        newFrame = process(frame,i)
        newFrame.save('./data/res/res-%i.png'%i)
        print('finish ',i)
        # threads[-1].start()
        i+=1
    # input('fffff')
    # for thread in threads:
    #     thread.join()
    out(i)

if __name__ =='__main__':
    main()
    # out(1750)