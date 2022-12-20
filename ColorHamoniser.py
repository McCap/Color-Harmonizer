#from mpl_toolkits import mplot3d
#import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def nothing(x):
    pass

global spline, hue, row, col, percent
hue_center = []
maxima = []
### Math ####
hue_center= 0
maxima = 4
percent = .4

def labtolch(img_lab):
    img_lch = np.zeros(img_lab[:, :, :].shape, dtype=np.uint8)
    channel_a = img_lab[:, :, 1].astype(np.float16)
    channel_b = img_lab[:, :, 2].astype(np.float16)
    img_lch[:,:,0] = img_lab[:,:,0]
    channel_c = np.sqrt(np.square(channel_a-128) + np.square(channel_b-128) )           # 0 to 128
    img_lch[:, :, 1] = channel_c.astype(np.uint8)                                       # signed 8 bit int
    channel_h = (np.arctan2((channel_b-128), (channel_a-128)) ) / (2*np.pi) * 255       # 0 to 255
    for i in range(channel_h.shape[0]):
        for j in range(channel_h.shape[1]):
            if channel_h[i, j] < 0:
                channel_h[i, j] = 255 + channel_h[i, j]

    img_lch[:, :, 2] = channel_h.astype(np.uint8)
    #img_lch[:,:,2] = (np.arctan2((img_lab[:,:,2]-127), (img_lab[:,:,1]-127)) + np.pi) / (2*np.pi) * 255     # 0 to 2Pi
    return img_lch

def lchtolab(img_lch):
    img_lab = np.zeros(img_lch[:, :,:].shape, dtype=np.uint8)
    channel_l = img_lch[:, :, 0].astype(np.float16)
    channel_c = img_lch[:, :, 1].astype(np.float16)
    channel_h = img_lch[:, :, 2].astype(np.float16)

    channel_l
    channel_a = channel_c * np.cos( channel_h / 255 * 2 * np.pi ) + 127
    channel_b = channel_c * np.sin( channel_h / 255 * 2 * np.pi ) + 127

    row = 100
    img_lab[:, :, 0] = channel_l.astype(np.uint8)
    img_lab[:, :, 1] = channel_a.astype(np.uint8)
    img_lab[:, :, 2] = channel_b.astype(np.uint8)
    #print('lchtolab channel c: ', channel_c[127, 100:150])
    #print('lchtolab channel h: ', channel_h[127, 100:150])
    #print('lchtolab channel a: ', channel_a[row, 100:150])
    #print('lchtolab img_lab a: ', img_lab[row, 100:150, 1])
    #print('lchtolab channel b: ', channel_b[row, 100:150])
    #print('lchtolab img_lab b: ', img_lab[row, 100:150, 2])
    #print('lchtolab img_lab type: ', img_lab.dtype)
    return img_lab

def lchtobgr(img_lch):
    img_lab = lchtolab(img_lch)
    img_lab = img_lab.astype(np.uint8)
    img_bgr = cv.cvtColor(img_lab, cv.COLOR_Lab2BGR)
    return img_bgr

def bgrtolch(img_bgr):
    img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2Lab)
    img_lch =labtolch(img_lab)
    return img_lch

def CallBackFunc(event, x, y, flags, param):
    global hue_picked
    if event == cv.EVENT_LBUTTONDOWN:
        global img_lab
        hue_picked[0, 0, 2] = img_lab[y,x,2]
        print('b left: ', hue_picked[0, 0,2])
        hue_picked[0, 0, 1] = img_lab[y,x,1]
        print('a left: ', hue_picked[0, 0,1])
        hue_picked[0, 0, 0] = img_lab[y,x,0]
        print('hue_picked in Callback: ', hue_picked.dtype)
        if cv.getTrackbarPos('Type', 'Color Harmoniser') == 0:
            calc_shift_lab(cv.getTrackbarPos('Strength', 'Color Harmoniser') / 100)
        elif cv.getTrackbarPos('Type', 'Color Harmoniser') == 1:
            calc_shift_lch_h(cv.getTrackbarPos('Strength', 'Color Harmoniser') / 100)
        elif cv.getTrackbarPos('Type','Color Harmoniser') == 2:
            calc_shift_rgb(cv.getTrackbarPos('Strength', 'Color Harmoniser') / 100)
    elif event == cv.EVENT_RBUTTONDOWN:
        global  img_R_lch
        hue_picked[0, 1, 2] = img_lab[y,x,2]
        print('b right: ', hue_picked[0, 1,2])
        hue_picked[0, 1, 1] = img_lab[y,x,1]
        print('a right: ', hue_picked[0, 1,1])
        hue_picked[0, 1, 0] = img_lab[y,x,0]
        if cv.getTrackbarPos('Type', 'Color Harmoniser') == 0:
            calc_shift_lab(cv.getTrackbarPos('Strength', 'Color Harmoniser') / 100)
        elif cv.getTrackbarPos('Type','Color Harmoniser') == 1:
            calc_shift_lch_h(cv.getTrackbarPos('Strength', 'Color Harmoniser') / 100)
        elif cv.getTrackbarPos('Type','Color Harmoniser') == 2:
            calc_shift_rgb(cv.getTrackbarPos('Strength', 'Color Harmoniser') / 100)
    elif event == cv.EVENT_MBUTTONDOWN:
        print("Middle button of the mouse is clicked - position (", x, ", ", y, ")")
        hue_picked[0, 2, 2] = img_lab[y, x, 2]
        print('b middle: ', hue_picked[0, 1, 2])
        hue_picked[0, 2, 1] = img_lab[y, x, 1]
        print('a middle: ', hue_picked[0, 1, 1])
        hue_picked[0, 2, 0] = img_lab[y, x, 0]
        if cv.getTrackbarPos('Type', 'Color Harmoniser') == 0:
            calc_shift_lab(cv.getTrackbarPos('Strength', 'Color Harmoniser') / 100)
        elif cv.getTrackbarPos('Type', 'Color Harmoniser') == 1:
            calc_shift_lch_h(cv.getTrackbarPos('Strength', 'Color Harmoniser') / 100)
        elif cv.getTrackbarPos('Type','Color Harmoniser') == 2:
            calc_shift_rgb(cv.getTrackbarPos('Strength', 'Color Harmoniser') / 100)
    if flags == cv.EVENT_FLAG_CTRLKEY + cv.EVENT_FLAG_LBUTTON:
        print("Left mouse button is clicked while pressing CTRL key - position (", x, ", ",y, ")")
    elif flags == cv.EVENT_FLAG_RBUTTON + cv.EVENT_FLAG_SHIFTKEY:
        print("Right mouse button is clicked while pressing SHIFT key - position (", x, ", ", y, ")")


def calc_shift_lch_h(v):
    global img_lab_harm, hue_picked, img_lab
    img_lch = labtolch(img_lab)
    img_lch_harm = labtolch(img_lab)
    hue_picked_lch = labtolch(hue_picked)
    hue_picked_lch_trans = np.zeros(hue_picked_lch.shape)
    for i in range(hue_picked_lch.shape[1]):
        hue_picked_lch_trans[0, i, 2] = hue_picked_lch[0, i, 2] - hue_picked_lch[0, 0, 2]

    print('hue_picked: ', hue_picked_lch)
    print('hue_picked trans: ', hue_picked_lch_trans)

    mapping = np.zeros((1, 256, 1))
    print('mapping: ', mapping.shape)
    for i in range(128, 255):
        mapping[0, i, 0] = 255

    blrad = cv.getTrackbarPos('Blur', 'Color Harmoniser') * 10 + 1
    print('blrad: ', blrad)

    mapping[:, :, 0] = cv.blur(mapping[:, :, 0], (blrad, blrad))  # Blur the mapping function
    print('mapping: ', mapping.shape)

    if cv.getTrackbarPos('Number of Colors', 'Color Harmoniser') == 0 :
        print('One color LCH-hue')
        for i in range(img_lch.shape[0]):
            for j in range(img_lch.shape[1]):
                hue = img_lch[i, j, 2].astype(np.float16)  -  hue_picked_lch[0, 0, 2]      # transorm  hue
                if hue < 0:
                    hue = 255 + hue

                hue_harm = mapping[0, hue.astype(np.uint8), 0] * v + (1-v) * hue
                #print(hue_harm)
                if (hue_harm + hue_picked_lch[0, 0, 2]) > 255 :
                    img_lch_harm[i, j, 2] = hue_harm + hue_picked_lch[0, 0, 2] - 255
                else:
                    img_lch_harm[i, j, 2] = hue_harm + hue_picked_lch[0, 0, 2]

                #img_lch_harm[i,j,2] = hue_picked_lch[0, 0, 2] * v + (1 - v) * img_lch[i,j,2]
        print('Loop done')
    #print('img_lch_harm hue: ', img_lch_harm[100, 100:150, 2])
    #print('img_lch_harm c: ', img_lch_harm[100, 100:150, 1])
    img_lab_harm = lchtolab(img_lch_harm)
    #print('img_lab_harm b 2: ', img_lab_harm[100, 100:150, 2])

    show_image()

def calc_shift_rgb(v):
    global img_lab_harm, hue_picked, img_lab
    img_rgb = cv.cvtColor(img_lab, cv.COLOR_LAB2RGB)
    img_rgb_harm = cv.cvtColor(img_lab_harm, cv.COLOR_LAB2RGB)
    hue_picked_rgb = cv.cvtColor(hue_picked, cv.COLOR_LAB2RGB)

    grbgrad = createlabgrad()
    grbgradshift = createlabgrad()
    if cv.getTrackbarPos('Number of Colors', 'Color Harmoniser') == 0 :
        print('One color RGB')
        for i in range(img_rgb.shape[0]):
            for j in range(img_rgb.shape[1]):
                # G fixed, changing only R and B
                img_rgb_harm[i,j,0] = hue_picked[0, 0, 0] * v + (1 - v) * img_rgb[i,j,0]
                img_rgb_harm[i,j,2] = hue_picked[0, 0, 2] * v + (1 - v) * img_rgb[i,j,2]
        print('Loop done')

    else:
        print('Number of colors: ', cv.getTrackbarPos('Number of Colors', 'Color Harmoniser')+1)

        indexes = search_closest(grbgrad[:, :, 1], grbgrad[:, :, 2])
        grbgradshift[:, :, 1] = np.take(hue_picked_rgb[0, :, 0], indexes[:, :, 0])
        grbgradshift[:, :, 2] = np.take(hue_picked_rgb[0, :, 2], indexes[:, :, 0])

        blrad = cv.getTrackbarPos('Blur','Color Harmoniser') * 10 + 1
        print('blrad: ', blrad)

        blur_a = cv.blur(grbgradshift[:, :, 1], (blrad, blrad))       # Blur the mapping function
        blur_b = cv.blur(grbgradshift[:, :, 2], (blrad, blrad))       # Blur the mapping function
        grbgradshift[:, :, 1] = blur_a
        grbgradshift[:, :, 2] = blur_b

    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            # index = search_closest(img_lab[i,j, 1], img_lab[i, j, 2], 1 + cv.getTrackbarPos('Number of Colors', 'Color Harmoniser'))
            buffer_a = grbgradshift[img_rgb[i, j, 2], img_rgb[i, j, 0], 1] * v + (1 - v) * img_rgb[i, j, 0]
            buffer_b = grbgradshift[img_rgb[i, j, 2], img_lab[i, j, 1], 2] * v + (1 - v) * img_lab[i, j, 2]
            # print(buffer_a.astype(np.uint8))
            img_rgb_harm[i, j, 0] = buffer_a.astype(np.uint8)
            img_rgb_harm[i, j, 2] = buffer_b.astype(np.uint8)
            # img_lab_harm[i,j,1] = scale * (hue_picked[index[0], 1] - img_lab[i,j,1]) * v + img_lab[i,j,1]
            # img_lab_harm[i,j,2] = scale * (hue_picked[index[0], 2] - img_lab[i,j,2]) * v + img_lab[i,j,2]

    img_lab_harm = cv.cvtColor(img_rgb_harm, cv.COLOR_RGB2LAB)
    print('Loop done')
    show_image()


def calc_shift_lab(v):
    global img_lab_harm, hue_picked, img_lab

    labgrad = createlabgrad()
    labgradshift = createlabgrad()

    if cv.getTrackbarPos('Number of Colors', 'Color Harmoniser') == 0 :
        print('One color')
        for i in range(img_lab.shape[0]):
            for j in range(img_lab.shape[1]):
                if img_lab[i, j, 0] <127:
                    scale = img_lab[i, j, 0] / 127
                else:
                    scale = 1 - (img_lab[i, j, 0]-127) / 127
                img_lab_harm[i,j,1] =  hue_picked[0, 0, 1] * v + (1 - v) * img_lab[i,j,1]
                img_lab_harm[i,j,2] =  hue_picked[0, 0, 2] * v + (1 - v) * img_lab[i,j,2]
        print('Loop done')
    else:
        print('Number of colors: ', cv.getTrackbarPos('Number of Colors', 'Color Harmoniser')+1)

        indexes = search_closest(labgrad[:, :, 1], labgrad[:, :, 2])
        labgradshift[:, :, 1] = np.take(hue_picked[0, :, 1], indexes[:, :, 0])
        labgradshift[:, :, 2] = np.take(hue_picked[0, :, 2], indexes[:, :, 0])
        #for i in range(labgrad.shape[0]):
        #    for j in range(labgrad.shape[1]):
        #        index = search_closest(labgrad[i, j, 1], labgrad[i, j, 2])
        #        labgradshift[i, j, 1] = hue_picked[0, index[0], 1]
        #        labgradshift[i, j, 2] = hue_picked[0, index[0], 2]

        blrad = cv.getTrackbarPos('Blur','Color Harmoniser') * 10 + 1
        print('blrad: ', blrad)

        blur_a = cv.blur(labgradshift[:, :, 1],(blrad,blrad))       # Blur the mapping function
        blur_b = cv.blur(labgradshift[:, :, 2],(blrad,blrad))       # Blur the mapping function
        labgradshift[:, :, 1] = blur_a
        labgradshift[:, :, 2] = blur_b


        for i in range(img_lab.shape[0]):
            for j in range(img_lab.shape[1]):
                #index = search_closest(img_lab[i,j, 1], img_lab[i, j, 2], 1 + cv.getTrackbarPos('Number of Colors', 'Color Harmoniser'))
                if img_lab[i, j, 0] <127:
                    scale = img_lab[i, j, 0] / 127
                else:
                    scale = 1 - (img_lab[i, j, 0]-127) / 127
                buffer_a = labgradshift[img_lab[i,j,2], img_lab[i,j,1], 1] * v + (1-v) * img_lab[i,j,1]
                buffer_b = labgradshift[img_lab[i,j,2], img_lab[i,j,1], 2] * v + (1-v) * img_lab[i,j,2]
                #print(buffer_a.astype(np.uint8))
                img_lab_harm[i,j,1] = buffer_a.astype(np.uint8)
                img_lab_harm[i,j,2] = buffer_b.astype(np.uint8)
                #img_lab_harm[i,j,1] = scale * (hue_picked[index[0], 1] - img_lab[i,j,1]) * v + img_lab[i,j,1]
                #img_lab_harm[i,j,2] = scale * (hue_picked[index[0], 2] - img_lab[i,j,2]) * v + img_lab[i,j,2]

        print('Loop done')
    show_image()
    return None

def search_closest(a, b):
    af = np.atleast_3d(a.astype(float))
    bf = np.atleast_3d(b.astype(float))
    htype = cv.getTrackbarPos('Type','Color Harmoniser')

    if htype == 0:
        da = hue_picked[0, :, 1].reshape(1, 1, -1) - af
        db = hue_picked[0, :, 2].reshape(1, 1, -1) - bf
    elif htype == 2:
        hue_picked_rgb = cv.cvtColor(hue_picked, cv.COLOR_LAB2RGB)
        da = hue_picked_rgb[0, :, 1].reshape(1, 1, -1) - af
        db = hue_picked_rgb[0, :, 2].reshape(1, 1, -1) - bf
    dc2 = da ** 2 + db ** 2
    index = np.argsort(dc2)
    return index

def on_trackbar_strength(vs):
    print('vs: ', vs)
    #calc_shift_lab(vs / 100)
    return None

def show_image():
    out_harm = cv.cvtColor(img_lab_harm, cv.COLOR_Lab2BGR)
    out = np.concatenate((img_bgr, out_harm), axis=1)
    cv.imshow('Color Harmoniser', out)
    cv.setMouseCallback('Color Harmoniser', CallBackFunc)
    cv.imwrite('outimage.jpg', out_harm)
    return None

def createlabgrad():
    lab_image_fromL = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(0, 255):
        for j in range(0, 255):
            lab_image_fromL[i, j, 0] = 127  # L-chanel
            lab_image_fromL[i, j, 1] = j  # a-chanel
            lab_image_fromL[i, j, 2] = i  # b-chanel
    return lab_image_fromL



# Window
cv.namedWindow('Color Harmoniser')
cv.createTrackbar('Type','Color Harmoniser',0,2,nothing)         # Type 0: Lab-ab channels, Type 1: Lch-hue channel, Type 2: RGB-RB channels
cv.createTrackbar('Strength','Color Harmoniser',0,100,on_trackbar_strength)
cv.createTrackbar('Number of Colors','Color Harmoniser',0,2,nothing)
cv.createTrackbar('Blur','Color Harmoniser',0,10, nothing)

hue_picked = np.zeros((1, 4, 3), dtype=np.uint8)                   # pull points, 1 row, 4 columns for 4 possible points, 3 layers for L, a, b. Like image to make it convertible
img_bgr = cv.imread('image7.jpg')               # Load image
#img_lab = np.zeros(img_bgr.shape, dtype=float)
img_lab = cv.cvtColor(img_bgr,cv.COLOR_BGR2Lab)    # convert image to LAB, Opencv does 0-255 for L, a and b
img_lch = labtolch(img_lab)
print('img_lch: ', img_lch.dtype)
img_lab_harm = cv.cvtColor(img_bgr,cv.COLOR_BGR2Lab)    # Create image to harmonise in LAB
img_lch_harm = labtolch(img_lab_harm)


print('hue_picked: ', hue_picked.dtype)
print(img_bgr.dtype)
print(img_lab.dtype)
print(img_lch.dtype)
show_image()

cv.waitKey()
cv.destroyAllWindows()