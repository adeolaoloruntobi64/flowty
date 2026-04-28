"""
@file morph_lines_detection.py
@brief Use morphology transformations for extracting horizontal and vertical lines sample code
"""
import numpy as np
import sys
import cv2

def show_wait_destroy(winname, img):
    cv2.imshow(winname, cv2.resize(img, (500, 500)))
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)

swd = lambda x: show_wait_destroy("w", x)

def a():
    img = cv2.imread("dichromate/pics/chainp3.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 35., 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel, iterations=3)
    thresh = (blackhat > 20).astype(np.uint8) * 255
    opn = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    dist = cv2.distanceTransform(thresh, cv2.DIST_L1, 5)
    ring_mask = cv2.inRange(dist, 7, 300)
    norm = cv2.normalize(dist, None, 0, 1., cv2.NORM_MINMAX)
    show_wait_destroy("w", opn)
    gray += blackhat
    show_wait_destroy("w", gray)
    cv2.imwrite("gray.png", gray)
    thresh2 = cv2.threshold(gray, 35., 255, cv2.THRESH_BINARY)[1]
    show_wait_destroy("w", thresh2)
    tophat = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel, iterations=7)
    #show_wait_destroy("w", dk)
    #show_wait_destroy("w", blackhat)
    show_wait_destroy("w", tophat)

def b():
    img = cv2.imread("dichromate/pics/moonten.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 35., 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 2))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 8))
    gray = cv2.bitwise_and(gray, gray, None, thresh)
    grid_h = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_h)
    grid_v = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_v)
    grid = cv2.max(grid_h, grid_v)
    grid = cv2.dilate(grid, kernel, iterations=1)
    tophat = cv2.morphologyEx(grid, cv2.MORPH_TOPHAT, kernel, iterations=5)
    opened = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, kernel, iterations=1)
    comb = cv2.subtract(opened, gray)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    thresh2 = cv2.threshold(comb, 25., 255, cv2.THRESH_BINARY)[1]
    dil = cv2.dilate(thresh2, kernel2, iterations=1)
    ero = cv2.morphologyEx(
        dil,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),
        iterations=5
    )
    show_wait_destroy("w", grid)
    show_wait_destroy("w", grid_h)
    show_wait_destroy("w", grid_v)
    show_wait_destroy("w", tophat)
    show_wait_destroy("w", opened)
    show_wait_destroy("w", dil)
    show_wait_destroy("w", ero)
    cv2.imwrite("gray.png", ero)

def c():
    img = cv2.imread("dichromate/pics/chainp3.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 35., 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    blackhat = cv2.morphologyEx(thresh, cv2.MORPH_BLACKHAT, kernel, iterations=4)
    tophat = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel, iterations=5)
    tophat2 = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel, iterations=5)
    tophat = (tophat > 5).astype(np.uint8) * 255
    dt = cv2.distanceTransform(tophat, cv2.DIST_L1, 3)
    nm = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX)
    dl = cv2.dilate(nm, kernel2, iterations=6)
    nt = thresh + (dl > 50).astype(np.uint8) * 255
    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    swd(cv2.bitwise_and(dl, cv2.bitwise_not(nm)))
    swd(nt)
    swd(tophat2)

# Can we use the fact that the rings have the exact same color as the cells
# maybe dilate and only capture exact pixel value. EXACT. Maybe +- some shi
def d():
    img = cv2.imread("dichromate/pics/chainp3.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p2, p98 = np.percentile(img, (2, 98))
    gray = np.clip((img - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    binary = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)[1]
    gray = cv2.bitwise_and(gray, gray, None, binary)
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    Jxx = Ix * Ix
    Jyy = Iy * Iy
    Jxy = Ix * Iy

    # Local accumulation (box filter, NOT Gaussian)
    Sxx = cv2.boxFilter(Jxx, -1, (3, 3))
    Syy = cv2.boxFilter(Jyy, -1, (3, 3))
    Sxy = cv2.boxFilter(Jxy, -1, (3, 3))
    eps = 1e-6
    tmp = np.sqrt((Sxx - Syy) ** 2 + 4 * Sxy ** 2)

    lambda1 = Sxx + Syy + tmp
    lambda2 = Sxx + Syy - tmp

    straightness = lambda1 / (lambda2 + eps)
    STRAIGHTNESS_T = 10000000   # 5–10 works well

    line_mask = (straightness > STRAIGHTNESS_T).astype(np.uint8) * 255
    swd(line_mask)

def e():
    def get_similarity_mask(img, target_bgr, threshold=30):
        # 1. Convert image and target to HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        target_mat = np.uint8([[target_bgr]])
        target_hsv = cv2.cvtColor(target_mat, cv2.COLOR_BGR2HSV)[0][0].astype(np.float32)

        # 2. Define Weights (Hue is vital, Value/Brightness matters "a bit")
        # Weights: [Hue, Saturation, Value]
        weights = np.array([1.0, 0.5, 0.3]) 

        # 3. Calculate Circular Hue Difference
        # In OpenCV, Hue is 0-179. We find the shortest distance around the circle.
        dh = np.abs(hsv_img[:,:,0] - target_hsv[0])
        dh = np.minimum(dh, 180 - dh)
        
        # 4. Calculate S and V differences
        ds = hsv_img[:,:,1] - target_hsv[1]
        dv = hsv_img[:,:,2] - target_hsv[2]
        # 5. Compute Weighted Euclidean Distance
        # distance = sqrt( w1*dh² + w2*ds² + w3*dv² )
        dist_sq = (weights[0] * dh)**2 + (weights[1] * ds)**2 + (weights[2] * dv)**2
        dist = np.sqrt(dist_sq)

        # 6. Create Binary Mask
        mask = (dist < threshold).astype(np.uint8) * 255
        return mask

    img = cv2.imread("dichromate/pics/moonten.png")
    cv2.COLOR_BGR2Lab
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 35., 255, cv2.THRESH_BINARY)[1]
    imgc = cv2.bitwise_and(img, img,  mask=thresh)
    swd(imgc)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    op = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
    swd(op)
    msk = cv2.bitwise_and(img, img, mask=op)
    swd(msk)
    dil = cv2.dilate(msk, kernel2, iterations=5)
    swd(dil)
    diff = cv2.absdiff(dil, imgc)
    swd(diff)
    gdiff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    swd(gdiff)
    _, thresh2 = cv2.threshold(gdiff, 30, 255, cv2.THRESH_BINARY)
    swd(thresh2)
    grayc = cv2.bitwise_and(gray, gray, mask=thresh2)
    swd(grayc)

def f():
    import time
    def fast_Lstar_precise(img_bgr):
        bgr = img_bgr.astype(np.float32)/255.0
        r = bgr[..., 2]
        g = bgr[..., 1]
        b = bgr[..., 0]

        # sRGB → linear
        def linearize(c):
            mask = c <= 0.04045
            return np.where(mask, c / 12.92, ((c + 0.055)/1.055) ** 2.4)

        r_lin = linearize(r)
        g_lin = linearize(g)
        b_lin = linearize(b)

        # luminance Y
        Y = 0.2126*r_lin + 0.7152*g_lin + 0.0722*b_lin

        # piecewise L*
        epsilon = 0.008856
        L = np.where(Y > epsilon, 116*np.cbrt(Y) - 16, 903.3*Y)

        # scale to OpenCV range 0–255
        L = L * 255 / 100
        return L.astype(np.uint8)
    img = cv2.imread("dichromate/pics/multi-3.png")
    #s = time.time()
    #lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #e = time.time()
    #print(f"Uhh {e - s}")
    #l, a, b = cv2.split(lab)
    s = time.time()
    l = fast_Lstar_precise(img)
    e = time.time()
    print(f"Uhh1.5 {e - s}")
    #print(type(l))
    #diff = cv2.absdiff(l2, l)
    #swd(diff)
    _, lthresh = cv2.threshold(l, 40., 255., cv2.THRESH_BINARY)
    swd(lthresh)
    img = cv2.bitwise_and(img, img, mask=lthresh)
    swd(img)
    cv2.imwrite("gra1.png", img)
    s = time.time()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    e = time.time()
    print(f"Uhh2 {e - s}")
    mthresh = cv2.bitwise_not(cv2.inRange(
        hsv,
        (0, 225, 0),
        (255, 255, 90)
    ))
    swd(mthresh)
    img = cv2.bitwise_and(img, img, mask=mthresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tophat = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=5)
    swd(tophat)
    #in_range(
    #    temp_mat,
    #    &Vec3b::from([0, 225, 0]),
    #    &Vec3b::from([255, 255, 90]),
    #    blob_mat
    #)?;
    #in_range(
    #    temp_mat,
    #    &Vec3b::from_array([0, 0, 100]),
    #    &Vec3b::from_array([255, 255, 255]),
    #    blob_mat
    #)?;
    # ---- Display ----
    #cv2.imshow("mask", mask)
    #cv2.imshow("result", result)
    #cv2.imwrite("gray.png", np.hstack(l))
    cv2.imwrite("gra2.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def g():
    import time
    img = cv2.imread("dichromate/pics/chainp3.png")
    s = time.time()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    e = time.time()
    print(f"Uhh {e - s}")
    l, a, b = cv2.split(lab)
    _, lthresh = cv2.threshold(l, 37., 255., cv2.THRESH_BINARY)
    swd(lthresh)
    img = cv2.bitwise_and(img, img, mask=lthresh)
    swd(img)
    cv2.imwrite("gra1.png", img)
    s = time.time()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    e = time.time()
    print(f"Uhh2 {e - s}")
    mthresh = cv2.bitwise_not(cv2.inRange(
        hsv,
        (0, 225, 0),
        (255, 255, 90)
    ))
    swd(mthresh)
    img = cv2.bitwise_and(img, img, mask=mthresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel, iterations=5)
    swd(tophat)
    #in_range(
    #    temp_mat,
    #    &Vec3b::from([0, 225, 0]),
    #    &Vec3b::from([255, 255, 90]),
    #    blob_mat
    #)?;
    #in_range(
    #    temp_mat,
    #    &Vec3b::from_array([0, 0, 100]),
    #    &Vec3b::from_array([255, 255, 255]),
    #    blob_mat
    #)?;
    # ---- Display ----
    #cv2.imshow("mask", mask)
    #cv2.imshow("result", result)
    cv2.imwrite("gray.png", np.hstack((l, a, b)))
    cv2.imwrite("gra2.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def h():
    img = cv2.imread("dichromate/pics/chainp3.png")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    thresh = cv2.inRange(hsv, (0, 0, 0), (255, 255, 10))
    inv = cv2.bitwise_not(thresh)
    hsv = cv2.bitwise_and(hsv, hsv, mask=inv)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tophat = cv2.morphologyEx(hsv, cv2.MORPH_TOPHAT, kernel, iterations=5)
    swd(tophat)
h()