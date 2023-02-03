import cv2
import numpy as np
import webp
from PIL import Image,ImageDraw,ImageFont,ImageEnhance

c = 10
fnt = ImageFont.truetype('./data/ARIBL0.ttf',c)
chl = list('$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\^`.54')
fs = (1920,1080)

def preRender(chl,c):
    trans = []
    for ch in chl:
        out = Image.new('L',(c,c),color = (0))
        d = ImageDraw.Draw(out)
        d.text((0,-2),ch,font = fnt, fill = (255))
        o = np.asarray(out)
        trans.append(o)
    return trans

def process (img,chs):
    imgShape = img.shape
    if(imgShape[1],imgShape[0])!= fs:
        img = cv2.resize(img,fs)
        imgShape = img.shape
    imgr = cv2.resize(img,(imgShape[1]//c,imgShape[0]//c))
    imgrShape = imgr.shape
    output = cv2.resize(imgr,fs)
    Mask = np.zeros((imgShape[0],imgShape[1])) #1920x1080
    for i in range(imgrShape[0]):
        for j in range(imgrShape[1]):
            color = imgr[i][j] # 3x1
            index = int(np.mean(color)%len(chs))
            Mask[i*c:(i+1)*c,j*c:(j+1)*c] = chs[index]
    Mask = Mask.astype(np.uint8)
    output = output.astype(np.uint8)
    print(Mask.shape)
    print(output.shape)
    output = cv2.bitwise_and(output,output,mask = Mask)
    return output

def out(frames,i):
    out = cv2.VideoWriter('output720.mp4',cv2.VideoWriter_fourcc(*'mp4v'),30, fs)
    for j in range(i):
        source = cv2.resize(frames[j],fs)
        # cv2.imshow('asf',source)
        cv2.waitKey(1)
        out.write(source)
        print(j)
    out.release()

def main():
    cap = cv2.VideoCapture('./data/vergill4.mp4')
    chs = preRender(chl,c)
    i=0
    frames = []
    print('start')
    while(cap.isOpened()):
        ret,frame =cap.read()
        if not ret:
            break
        newFrame = process(frame,chs)
        frames.append(newFrame)
        print('finish ',i)
        i+=1
    print('finish render')
    print(frames[0])
    out(frames,i)

if __name__ =='__main__':
    main()