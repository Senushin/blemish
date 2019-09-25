import cv2
import numpy as np

img = cv2.imread('image.png')
mask = np.ones((30,30,3),np.uint8)+245
r = 15 # radius

def func1(action, x, y, flags, userdata):
    #global mask
    if action==cv2.EVENT_LBUTTONDOWN:
        magList = []

        for x_modl in [-(r*2), 0, r*2]:
            for y_modl in [-(r*2), 0, r*2]:
                x_mod = x+x_modl
                y_mod = y+y_modl
                center=[x_mod,y_mod]
                zone = img [y_mod-r:y_mod+r,x_mod-r:x_mod+r]
                grayzone = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
                f = np.fft.fft2(grayzone)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = 20*np.log(np.abs(fshift))
                f_ishift = np.fft.ifftshift(fshift)
                mag = [np.sum(magnitude_spectrum),center]
                #print(mag)
                magList.append(mag)
                zone = 0

        blemish = magList[4]
        replaser = min(magList)
        x_b, y_b = blemish [1][0], blemish [1][1]
        x_r, y_r = replaser [1][0], replaser [1][1]
        zone_b = img [y_b-r:y_b+r,x_b-r:x_b+r]
        zone_r = img [y_r-r:y_r+r,x_r-r:x_r+r]
        zone_sc = cv2.seamlessClone(zone_r, zone_b,mask, (r,r), cv2.NORMAL_CLONE)
        img[y-r:y+r,x-r:x+r] = zone_sc
        cv2.imshow('Blemish',img)

cv2.namedWindow('Blemish')
cv2.setMouseCallback('Blemish', func1)
while True:
    cv2.imshow('Blemish',img)
    f = cv2.waitKey()
    if f ==27:
        break
cv2.destroyAllWindows()

