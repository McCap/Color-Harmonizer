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
    img_lch = np.zeros_like(img_lab, dtype=np.uint8)
    channel_a = img_lab[:, :, 1].astype(np.float16)
    channel_b = img_lab[:, :, 2].astype(np.float16)
    img_lch[:,:,0] = img_lab[:,:,0]
    channel_c = np.sqrt(np.square(channel_a-128) + np.square(channel_b-128) )           # 0 to 128
    img_lch[:, :, 1] = channel_c.astype(np.uint8)                                       # signed 8 bit int
    channel_h = (np.arctan2((channel_b-128), (channel_a-128)) ) / (2*np.pi) * 255       # 0 to 255
    channel_h[channel_h < 0] += 255

    img_lch[:, :, 2] = channel_h.astype(np.uint8)
    return img_lch

def labtolch_2d(img_lab):
    img_lch = np.zeros_like(img_lab, dtype=np.uint8)
    channel_a = img_lab[:, 1].astype(np.float16)
    channel_b = img_lab[:, 2].astype(np.float16)
    img_lch[:,0] = img_lab[:,0]
    channel_c = np.sqrt(np.square(channel_a-128) + np.square(channel_b-128) )           # 0 to 128
    img_lch[:, 1] = channel_c.astype(np.uint8)                                       # signed 8 bit int
    channel_h = (np.arctan2((channel_b-128), (channel_a-128)) ) / (2*np.pi) * 255       # 0 to 255
    channel_h[channel_h < 0] += 255
    img_lch[:, 2] = channel_h.astype(np.uint8)
    return img_lch

def lchtolab(img_lch):
    img_lab = np.zeros_like(img_lch, dtype=np.uint8)
    channel_l = img_lch[:, :, 0].astype(np.float16)
    channel_c = img_lch[:, :, 1].astype(np.float16)
    channel_h = img_lch[:, :, 2].astype(np.float16)

    channel_a = channel_c * np.cos( channel_h / 255 * 2 * np.pi ) + 127
    channel_b = channel_c * np.sin( channel_h / 255 * 2 * np.pi ) + 127

    img_lab[:, :, 0] = channel_l.astype(np.uint8)
    img_lab[:, :, 1] = channel_a.astype(np.uint8)
    img_lab[:, :, 2] = channel_b.astype(np.uint8)
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
        hue_picked[0, 2] = img_lab[y,x,2]
        print('b left: ', hue_picked[ 0,2])
        hue_picked[ 0, 1] = img_lab[y,x,1]
        print('a left: ', hue_picked[ 0,1])
        hue_picked[ 0, 0] = img_lab[y,x,0]
        print('hue_picked in Callback: ', hue_picked.dtype)
        if cv.getTrackbarPos('Type', 'Color Harmoniser') == 0:
            calc_shift_lab(cv.getTrackbarPos('Strength', 'Color Harmoniser') / 100)
        elif cv.getTrackbarPos('Type', 'Color Harmoniser') == 1:
            calc_shift_lch_h(cv.getTrackbarPos('Strength', 'Color Harmoniser') / 100)
    elif event == cv.EVENT_RBUTTONDOWN:
        global  img_R_lch
        hue_picked[ 1, 2] = img_lab[y,x,2]
        print('b right: ', hue_picked[ 1,2])
        hue_picked[1, 1] = img_lab[y,x,1]
        print('a right: ', hue_picked[ 1,1])
        hue_picked[1, 0] = img_lab[y,x,0]
        if cv.getTrackbarPos('Type', 'Color Harmoniser') == 0:
            calc_shift_lab(cv.getTrackbarPos('Strength', 'Color Harmoniser') / 100)
        elif cv.getTrackbarPos('Type','Color Harmoniser') == 1:
            calc_shift_lch_h(cv.getTrackbarPos('Strength', 'Color Harmoniser') / 100)
    elif event == cv.EVENT_MBUTTONDOWN:
        print("Middle button of the mouse is clicked - position (", x, ", ", y, ")")
        hue_picked[2, 2] = img_lab[y, x, 2]
        print('b middle: ', hue_picked[1, 2])
        hue_picked[2, 1] = img_lab[y, x, 1]
        print('a middle: ', hue_picked[1, 1])
        hue_picked[2, 0] = img_lab[y, x, 0]
        if cv.getTrackbarPos('Type', 'Color Harmoniser') == 0:
            calc_shift_lab(cv.getTrackbarPos('Strength', 'Color Harmoniser') / 100)
        elif cv.getTrackbarPos('Type', 'Color Harmoniser') == 1:
            calc_shift_lch_h(cv.getTrackbarPos('Strength', 'Color Harmoniser') / 100)
    if flags == cv.EVENT_FLAG_CTRLKEY + cv.EVENT_FLAG_LBUTTON:
        print("Left mouse button is clicked while pressing CTRL key - position (", x, ", ",y, ")")
    elif flags == cv.EVENT_FLAG_RBUTTON + cv.EVENT_FLAG_SHIFTKEY:
        print("Right mouse button is clicked while pressing SHIFT key - position (", x, ", ", y, ")")


def search_closest_lch(strength, picked):
    print('search closest lch - picked: ', picked)
    hues = np.arange(256)
    dh = picked.reshape(len(picked), -1) - hues
    dh_abs = np.abs(dh)
    dh2 = np.where(dh < -127, 256+dh, dh)
    dh2_abs = np.abs(dh2)
    closest = np.argsort(dh2_abs, axis=0)
    result = closest[0]
    pull_colors = np.choose(result, picked)
    distance = np.where(result == 0, dh2[0],
                        np.where(result == 1, dh2[1], 0
                        ))
    return (hues + strength*distance).astype(np.uint8)
"""
    distance = np.where(result == 0, dh2[0],
                        np.where(result == 1, dh2[1],
                                 np.where(result == 2, dh2[2],
                                          np.where(result == 3, dh2[3], 0
                        ))))
"""


def calc_shift_lch_h(strength):
    global img_lab_harm, hue_picked, img_lab
    ncolors = cv.getTrackbarPos('Number of Colors', 'Color Harmoniser')+1
    hue_picked_lch = labtolch_2d((hue_picked))
    new_hues = search_closest_lch(strength, hue_picked[0:ncolors, 2])
    img_lch_harm[:,:,1] = new_hues[img_lch_harm[:, :, 2]]

    labgrad = createlabgrad()
    labgradshift = createlabgrad()
    labgradshift[:, :, 2] = new_hues
    out_labgrad = cv.cvtColor(createlabgrad(), cv.COLOR_Lab2BGR)
    cv.imshow('Gradient', out_labgrad)
    out_labgradshift = cv.cvtColor(labgradshift, cv.COLOR_Lab2BGR)
    cv.imshow('Shifted Gradient', out_labgradshift)

    print('Loop done')
    show_image()
    return None

def calc_shift_lab(strength):
    global img_lab_harm, hue_picked, img_lab

    labgrad = createlabgrad()
    labgradshift = createlabgrad()
    print('1. labgradshift.shape: ', labgradshift.shape)
    ncolors = cv.getTrackbarPos('Number of Colors', 'Color Harmoniser')


    if ncolors == 0 :
        print('One color')
        labgradshift[:, :, 1] = hue_picked[0, 1] * strength + (1 - strength) * labgrad[:, :, 1]
        labgradshift[:, :, 2] = hue_picked[0, 2] * strength + (1 - strength) * labgrad[:, :, 2]
        print('Loop done')
    else:
        ncolors += 1
        print('Number of colors: ', ncolors)

        indexes = search_closest_lab(labgrad[:, :, 1], labgrad[:, :, 2])
        labgradshift[:, :, 1] = np.take(hue_picked[:, 1], indexes[:, :, 0])
        labgradshift[:, :, 2] = np.take(hue_picked[:, 2], indexes[:, :, 0])
        print('hue_picked: ', hue_picked)
        print('hue_picked[1]: ', hue_picked[:, 1])
        print('2. labgradshift.shape: ', labgradshift.shape)

        blrad = cv.getTrackbarPos('Blur','Color Harmoniser') * 10 + 1
        print('blrad: ', blrad)

        blur_a = cv.blur(labgradshift[:, :, 1],(blrad,blrad))       # Blur the mapping function
        blur_b = cv.blur(labgradshift[:, :, 2],(blrad,blrad))       # Blur the mapping function
        labgradshift[:, :, 1] = blur_a
        labgradshift[:, :, 2] = blur_b

    img_lab_harm[:,:,1] = (labgradshift[img_lab[:,:,2], img_lab[:,:,1], 1] * strength + (1-strength) * img_lab[:,:,1]).astype('uint8')
    img_lab_harm[:,:,2] = (labgradshift[img_lab[:,:,2], img_lab[:,:,1], 2] * strength + (1-strength) * img_lab[:,:,2]).astype('uint8')


    out_labgrad = cv.cvtColor(labgrad, cv.COLOR_Lab2BGR)
    cv.imshow('Gradient', out_labgrad)
    out_labgradshift = cv.cvtColor(labgradshift, cv.COLOR_Lab2BGR)
    cv.imshow('Shifted Gradient', out_labgradshift)

    print('Loop done')
    show_image()
    return None

def search_closest_lab(a, b):
    af = np.atleast_3d(a.astype(float))
    bf = np.atleast_3d(b.astype(float))
    htype = cv.getTrackbarPos('Type','Color Harmoniser')
    ncolors = cv.getTrackbarPos('Number of Colors', 'Color Harmoniser') +1

    if htype == 0:
        da = hue_picked[0:ncolors, 1].reshape(1, 1, -1) - af
        db = hue_picked[0:ncolors, 2].reshape(1, 1, -1) - bf
    dc2 = da ** 2 + db ** 2
    index = np.argsort(dc2)
    print('index.shape: ',index.shape)
    print('index: ',index)
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
    lab_image_fromL[:, :, 0] = 127
    lab_image_fromL[:, :, 1] = np.arange(256)
    lab_image_fromL[:, :, 2] = lab_image_fromL[:, :, 1].T
    return lab_image_fromL



# Window
cv.namedWindow('Color Harmoniser')
cv.namedWindow('Gradient')
cv.namedWindow('Shifted Gradient')

cv.createTrackbar('Type','Color Harmoniser',0,1,nothing)         # Type 0: Lab-ab channels, Type 1: Lch-hue channel
cv.createTrackbar('Strength','Color Harmoniser',0,100,on_trackbar_strength)
cv.createTrackbar('Number of Colors','Color Harmoniser',0,2,nothing)
cv.createTrackbar('Blur','Color Harmoniser',0,10, nothing)

hue_picked = np.zeros((4, 3), dtype=np.uint8)                   # pull points, 4 rows for 4 possible points, 3 columns for L, a, b. Like image to make it convertible
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

while True:
    key = cv.waitKey()
    if key == 27 or key == 113:
        cv.destroyAllWindows()
        break
