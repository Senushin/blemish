import cv2
import numpy as np

img = cv2.imread('image.png')

def func1(action, x, y, flags, userdata):
    global top, bottom
    if action==cv2.EVENT_LBUTTONDOWN:
        cv2.destroyWindow('Blemish zone')
        top=[(x,y)]
        imgCopy = img.copy()
        #cv2.putText(imgCopy,'center is: ' + str(center),
            #(10,450), cv2.FONT_HERSHEY_SIMPLEX,
            #0.7,(255,255,255), 2 );
        cv2.imshow('Blemish',imgCopy)
    elif action==cv2.EVENT_LBUTTONUP:
        bottom=[(x,y)]
        zone = img [top[0][1]:bottom[0][1],top[0][0]:bottom[0][0]]
        #zone = zone.astype(cv2.CV_32F)
        #dft = cv2.dft(zone,flags = cv2.DFT_COMPLEX_OUTPUT)
        grayzone = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        f = cv2.dft(np.float32(grayzone), flags=cv2.DFT_COMPLEX_OUTPUT)
        f_shift = np.fft.fftshift(f)
        f_complex = f_shift[:,:,0] + 1j*f_shift[:,:,1]
        f_abs = np.abs(f_complex) + 1 # lie between 1 and 1e6
        f_bounded = 20 * np.log(f_abs)
        f_img = 255 * f_bounded / np.max(f_bounded)
        f_img = f_img.astype(np.uint8)
        print (type(f), f.shape)
        cv2.imshow('Blemish zone',f_img)
        zone = 0
cv2.namedWindow('Blemish')
cv2.setMouseCallback('Blemish', func1)
while True:
    cv2.imshow('Blemish',img)
    f = cv2.waitKey()
    if f ==27:
        break
cv2.destroyAllWindows()
