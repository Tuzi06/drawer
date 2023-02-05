import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from numba import jit
import time

c = 11
fnt = ImageFont.truetype('./data/ARIBL0.ttf',c)
chl = list('$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\^`.54')
fs = (1920,1080)
def preRender():
    trans = []
    for ch in chl:
        out = Image.new('L',(c,c),color = (0))
        d = ImageDraw.Draw(out)
        d.text((0,-2),ch,font = fnt, fill = (255))
        o = np.asarray(out)
        trans.append(o)
    trans = tuple(trans)
    return trans

@jit(target_backend='cuda')
def process (img,chs):
    imgShape = img.shape
    imgr =  cv2.GaussianBlur(img, (c,c), 0)
    Mask = np.zeros((imgShape[0],imgShape[1]),dtype = np.uint8) #1920x1080
    width = imgShape[0]//c
    height = imgShape[1]//c
    for i in range(width):
        for j in range(height):
            color = imgr[i][j] # 3x1
            index = int(np.mean(color)%len(chs))
            Mask[i*c:(i+1)*c,j*c:(j+1)*c] = chs[index]

    output = cv2.bitwise_and(imgr,imgr,mask = Mask)
    return output

def out(frames,i):
    out = cv2.VideoWriter('output720.mp4',cv2.VideoWriter_fourcc(*'mp4v'),30, fs)
    for j in range(i):
        source = cv2.resize(frames[j],fs)
        cv2.waitKey(1)
        out.write(source)
        print(j)
    out.release()

def main():
    cap = cv2.VideoCapture('./data/vergill4.mp4')
    chs = preRender()
    i=0
    frames = []

    print('start')
    start = time.perf_counter()
    while(cap.isOpened()):
        ret,frame =cap.read()
        if not ret:
            break
        newFrame = process(frame,chs)
        frames.append(newFrame)
        print('finish ',i)
        i+=1
    end = time.perf_counter()
    print('finish render')
    print(end-start)
    input('pause')
    out(newFrame,i)

if __name__ =='__main__':
    main()