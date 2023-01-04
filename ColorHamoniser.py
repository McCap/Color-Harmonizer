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
    hue_picked_lch_trans = np.zeros_like(hue_picked_lch)
    hue_picked_lch_trans[0, :, 2] = hue_picked_lch[0, :, 2] - hue_picked_lch[0, 0, 2]

    print('hue_picked: ', hue_picked_lch)
    print('hue_picked trans: ', hue_picked_lch_trans)

    mapping = np.zeros((1, 256, 1))
    print('mapping: ', mapping.shape)
    mapping[0, 128:255, 0] = 255

    blrad = cv.getTrackbarPos('Blur', 'Color Harmoniser') * 10 + 1
    print('blrad: ', blrad)

    mapping[:, :, 0] = cv.blur(mapping[:, :, 0], (blrad, blrad))  # Blur the mapping function
    print('mapping: ', mapping.shape)

    if cv.getTrackbarPos('Number of Colors', 'Color Harmoniser') == 0 :
        print('One color LCH-hue')
        hue = (img_lch[:, :, 2].astype('float16') - hue_picked_lch[0, 0, 2]).astype('uint8')
        # hue[hue < 0] += 255

        hue_harm = mapping[0, hue, 0] * v + (1 - v) * hue
        hue_mask = (hue_harm + hue_picked_lch[0, 0, 2]) > 255
        img_lch_harm[:, :, 2] = np.where(hue_mask, hue_harm + hue_picked_lch[0, 0, 2] - 255,
                                                   hue_harm + hue_picked_lch[0, 0, 2])

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
    ncolors = cv.getTrackbarPos('Number of Colors', 'Color Harmoniser')
    if ncolors == 0 :
        print('One color RGB')
        grbgradshift[:, :, 0] = hue_picked[0, 0, 0] * v + (1 - v) * grbgrad[:, :, 0]
        grbgradshift[:, :, 2] = hue_picked[0, 0, 2] * v + (1 - v) * grbgrad[:, :, 2]
        print('Loop done')

    else:
        ncolors += 1
        print('Number of colors: ', ncolors)

        indexes = search_closest(grbgrad[:, :, 1], grbgrad[:, :, 2])
        grbgradshift[:, :, 1] = np.take(hue_picked_rgb[0, :, 0], indexes[:, :, 0])
        grbgradshift[:, :, 2] = np.take(hue_picked_rgb[0, :, 2], indexes[:, :, 0])

        blrad = cv.getTrackbarPos('Blur','Color Harmoniser') * 10 + 1
        print('blrad: ', blrad)

        blur_a = cv.blur(grbgradshift[:, :, 1], (blrad, blrad))       # Blur the mapping function
        blur_b = cv.blur(grbgradshift[:, :, 2], (blrad, blrad))       # Blur the mapping function
        grbgradshift[:, :, 1] = blur_a
        grbgradshift[:, :, 2] = blur_b

    img_rgb_harm[:, :, 0] = (grbgradshift[img_rgb[:, :, 2], img_rgb[:, :, 0], 1] * v + (1 - v) * img_rgb[:, :, 0]).astype('uint8')
    img_rgb_harm[:, :, 2] = (grbgradshift[img_rgb[:, :, 2], img_lab[:, :, 1], 2] * v + (1 - v) * img_lab[:, :, 2]).astype('uint8')
    img_lab_harm = cv.cvtColor(img_rgb_harm, cv.COLOR_RGB2LAB)
    print('Loop done')
    show_image()


def calc_shift_lab(v):
    global img_lab_harm, hue_picked, img_lab

    labgrad = createlabgrad()
    labgradshift = createlabgrad()
    ncolors = cv.getTrackbarPos('Number of Colors', 'Color Harmoniser')

    if ncolors == 0 :
        print('One color')
        labgradshift[:, :, 1] = hue_picked[0, 0, 1] * v + (1 - v) * labgrad[:, :, 1]
        labgradshift[:, :, 2] = hue_picked[0, 0, 2] * v + (1 - v) * labgrad[:, :, 2]
        print('Loop done')
    else:
        ncolors += 1
        print('Number of colors: ', ncolors)

        indexes = search_closest(labgrad[:, :, 1], labgrad[:, :, 2])
        labgradshift[:, :, 1] = np.take(hue_picked[0, :, 1], indexes[:, :, 0])
        labgradshift[:, :, 2] = np.take(hue_picked[0, :, 2], indexes[:, :, 0])

        blrad = cv.getTrackbarPos('Blur','Color Harmoniser') * 10 + 1
        print('blrad: ', blrad)

        blur_a = cv.blur(labgradshift[:, :, 1],(blrad,blrad))       # Blur the mapping function
        blur_b = cv.blur(labgradshift[:, :, 2],(blrad,blrad))       # Blur the mapping function
        labgradshift[:, :, 1] = blur_a
        labgradshift[:, :, 2] = blur_b

    img_lab_harm[:,:,1] = (labgradshift[img_lab[:,:,2], img_lab[:,:,1], 1] * v + (1-v) * img_lab[:,:,1]).astype('uint8')
    img_lab_harm[:,:,2] = (labgradshift[img_lab[:,:,2], img_lab[:,:,1], 2] * v + (1-v) * img_lab[:,:,2]).astype('uint8')

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
    print(index[0, 0, :])
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

while True:
    key = cv.waitKey()
    if key == 27 or key == 113:
        cv.destroyAllWindows()
        break
