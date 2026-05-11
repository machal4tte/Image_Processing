import cv2
import os
# os.environ["QT_QPA_PLATFORM"] = "wayland"
import matplotlib
# matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def Compare(ImgA, ImgB, cmap=None):
    plt.figure(figsize=(10,5)) 
    plt.subplot(1, 2, 1)
    plt.imshow(ImgA, cmap)
    plt.subplot(1, 2, 2)
    plt.imshow(ImgB, cmap)
    plt.show()


def Frequency_enhance(img, sharpness=30):  
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows//2 , cols//2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-sharpness:crow+sharpness, ccol-sharpness:ccol+sharpness] = 0
    fshift_filtered = fshift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    img_back = np.abs(img_back)

    img_back = np.uint8(img_back)
    return img_back


def gamma_correction(img, gamma):
    table = np.array([(i / 255.0) ** gamma * 255
                      
    for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def CLAHE(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    CLAHE = clahe.apply(img)

    return CLAHE

def Sobel(img, KernelSize=5):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=KernelSize, scale=2) # dx=1, dy=0
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=KernelSize, scale=2) # dx=0, dy=1

    # 4. Convert back to 8-bit unsigned integers
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    # 5. Combine the two gradients
    sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    return sobel_combined

def Open(img, KernelSize=5):
    kernel = np.ones((KernelSize,KernelSize), np.uint8)
    open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return open

def Close(img, KernelSize=5, dest=None):
    kernel = np.ones((KernelSize,KernelSize), np.uint8)
    close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, dst=dest)
    return close

def Erode(img, KernelSize=5):
    kernel = np.ones((KernelSize,KernelSize), np.uint8)
    erode = cv2.erode(img, kernel)
    return erode

def Dialate(img, KernelSize=5):
    kernel = np.ones((KernelSize,KernelSize), np.uint8)
    Dialate = cv2.dilate(img, kernel)
    return Dialate

def Seeding(img, seed):
    mask = np.zeros_like(img)
    threshold = 55
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if abs(int(img[i,j]) - int(img[seed])) < threshold:
                mask[i,j] = 255
    return mask

def retinex(img, sigma=30):
    img = np.float32(img) + 1
    blur = cv2.GaussianBlur(img, (0,0), sigma)
    result = np.log(img) - np.log(blur)
    return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Linux
# Path = "//mnt//C0E08C6AE08C690C//work//my_work//Second_year//Second Semester//Image Processing//FinalProject//Picture"
# Window
Path = "C://work//my_work//Second_year//Second Semester//Image Processing//FinalProject//Picture"
# C:\work\my_work\Second_year\Second Semester\Image Processing\FinalProject\Picture

# Road detection
def road_detection():
    Road_path = "03_Driving-the-Grossglockner-High-Alpine-Road.webp"
    Road_img = cv2.imread(os.path.join(Path, Road_path))
    Road_gray = cv2.cvtColor(Road_img, cv2.COLOR_BGR2GRAY)
    Road_rgb = cv2.cvtColor(Road_img, cv2.COLOR_BGR2RGB)
    Road_lab = cv2.cvtColor(Road_img, cv2.COLOR_BGR2LAB)

    L, A, B = cv2.split(Road_lab)

    clahe = CLAHE(L)

    clahe = cv2.merge((L, A, B))
    
    Road_hsv = cv2.cvtColor(clahe, cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(Road_hsv)

    Road_thresh = cv2.inRange(H, 80, 140)
    Road_thresh = cv2.medianBlur(Road_thresh, 5)
    InvertRoad = cv2.bitwise_not(Road_thresh)
    close = Close(InvertRoad, 9)

    contour, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Filled = np.zeros_like(Road_gray)
    Filled = cv2.fillPoly(Filled, contour, color=255)
    cv2.medianBlur(Filled, 5, Filled)

    Flood = Filled.copy()
    cv2.floodFill(Flood, None, (0,0), (255))
    Road = cv2.bitwise_not(Flood)
    Ground = cv2.bitwise_or(Road,  Filled)

    H_ground = cv2.bitwise_and(H, Ground)
    S_ground = cv2.bitwise_and(S, Ground)
    V_ground = cv2.bitwise_and(V, Ground)

    H_road = cv2.inRange(H_ground, 90, 140, Road_img)

    cv2.imwrite("Filled.jpg", Filled)
    # cv2.imwrite("H_ground.jpg", H_ground)
    # cv2.imwrite("Ground.jpg", Ground)
    # cv2.imwrite("Flood.jpg", Flood)
    # cv2.imwrite("Road_thresh.jpg", Road_thresh)
    # cv2.imwrite("H.jpg", H)
    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1)
    plt.imshow(Flood, cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(Ground, cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(H_road, cmap='gray')
    plt.savefig("Final.jpg")
    # Compare(Road_thresh, H_road)
    

def Coin():
    # coin counting
    Coin_path = "05_Coin_Counting.jpeg"
    Coin_img = cv2.imread(os.path.join(Path, Coin_path))
    Coin_gray = cv2.cvtColor(Coin_img, cv2.COLOR_BGR2GRAY)
    Coin_rgb = cv2.cvtColor(Coin_img, cv2.COLOR_BGR2RGB)

    # Adjust lightning
    Light = gamma_correction(Coin_gray, 0.2)
    Light = CLAHE(Light)

    #Reduce noise 
    CoinDenoise = cv2.fastNlMeansDenoising(Light, None, 10, 7, 21)

    # Edge Enhance
    EdgeEnhance = Sobel(CoinDenoise,3)

    #threshold
    _, thresh = cv2.threshold(EdgeEnhance, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.medianBlur(thresh, 7)

    # Find Area
    BlankPage = np.zeros_like(Coin_gray)
    contour, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Filled = cv2.fillPoly(BlankPage, contour, color=255)

    #Find Coin 
    circles = cv2.HoughCircles(
        EdgeEnhance, 
        cv2.HOUGH_GRADIENT, 
        dp=1,            # Resolution of the accumulator (1 = same as image)
        minDist=35,       # Minimum distance between circle centers
        param1=70,       # Canny edge detection higher threshold
        param2=45,       # Accumulator threshold (lower = more circles)
        minRadius=45,     # Minimum radius of detected circle
        maxRadius=69      # Maximum radius (0 = no limit)
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw outer circle (green)
            print(i)
            cv2.circle(Coin_rgb, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw center (red)
            cv2.circle(Coin_rgb, (i[0], i[1]), 2, (0, 0, 255), 3)


def Fundus():
    Fundus_path = "01_Fundus_photograph_of_normal_left_eye.jpg"
    Fundus_img = cv2.imread(os.path.join(Path, Fundus_path))
    Fundus_gray = cv2.cvtColor(Fundus_img, cv2.COLOR_BGR2GRAY)

    _, ROI = cv2.threshold(Fundus_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    Light = CLAHE(Fundus_gray)

    Addaptive_Mean = cv2.adaptiveThreshold(Light, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 15)


# Extract the nerves
    Xorgate = cv2.bitwise_xor(ROI, Addaptive_Mean)
    Nerves = cv2.bitwise_and(ROI, Xorgate)


    num_labels, component_labels, stats, centroids = cv2.connectedComponentsWithStats(Nerves, connectivity=8)

    new_img = np.zeros_like(Nerves)

    for i in range(1, num_labels):  
        if stats[i, cv2.CC_STAT_AREA] >= 5:
            new_img[component_labels == i] = 255

    new_img = np.uint8(new_img)
    M = cv2.moments(new_img)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(new_img, (cX, cY), 5, (255, 0, 0), -1)
    cv2.putText(new_img, (f'Centroid = X : {cX}, Y :{cY}.'), (cX - 50, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

def Deer():
    Deer_img = cv2.imread(os.path.join(Path, "02_Deer-camera.jpg"))
    Deer_rgb = cv2.cvtColor(Deer_img, cv2.COLOR_BGR2RGB)
    Deer_hsv = cv2.cvtColor(Deer_img, cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(Deer_hsv)

    NLM = cv2.fastNlMeansDenoising(S, None, 
                                    21, #Higher Lower noise smoother edge 
                                    7, # use with color
                                    21 # Higher Lower noise smoother edge
                                    )

    Deer = cv2.inRange(NLM, 20, 120)
    denoise = cv2.medianBlur(Deer, 5)

    ROI = cv2.bitwise_and(Deer, S)


    sure_deer = cv2.inRange(ROI, 20, 100)
    dist = cv2.distanceTransform(sure_deer, cv2.DIST_L2, 5)

    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sure_fg = cv2.threshold(dist_norm, 16, 255, cv2.THRESH_BINARY)

    print(dist.shape)
    print(sure_fg.shape)
    num_labels, markers, stats, centroid = cv2.connectedComponentsWithStats(sure_fg, connectivity=8)

    Blank = np.zeros_like(sure_fg)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if 2000  < area < 20000 :
            Blank[markers == label] = 255

            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]

            cv2.rectangle(Deer_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)


