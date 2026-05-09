import cv2
import os
os.environ["QT_QPA_PLATFORM"] = "wayland"
import matplotlib
matplotlib.use('Qt5Agg')
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
Path = "//mnt//C0E08C6AE08C690C//work//my_work//Second_year//Second Semester//Image Processing//FinalProject//Picture"
# Window
# Path = "C://ACER//work//my_work//Second_year//Second Semester//Image Processing//FinalProject//Picture"

# Deer_path = "02_Deer-camera.jpg"
# Deer_img = cv2.imread(os.path.join(Path, Deer_path))
# Deer_rgb = cv2.cvtColor(Deer_img, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(10,5))
# plt.subplot(1,1,1)
# plt.imshow(Deer_rgb )
# plt.show()


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


    Compare(Road_thresh, H_road)
    

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
    CoinDenoise = cv2.fastNlMeansDenoising(Coin_gray, None, 10, 7, 21)

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

    Compare(Filled, Coin_rgb)


def Fundus():
    Fundus_path = "01_Fundus_photograph_of_normal_left_eye.jpg"
    Fundus_img = cv2.imread(os.path.join(Path, Fundus_path))
    Fundus_gray = cv2.cvtColor(Fundus_img, cv2.COLOR_BGR2GRAY)

    _, ROI = cv2.threshold(Fundus_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    Light = CLAHE(Fundus_gray)

    #Denoise
    Denoise = cv2.fastNlMeansDenoising(Light, None, 
                                    15, #Higher Lower noise smoother edge 
                                    7, # use with color
                                    35 # Higher Lower noise smoother edge
                                    )

    canny =  cv2.Canny(Denoise, 100, 100)

    Addaptive_Mean = cv2.adaptiveThreshold(Light, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 15)


# Extract the nerves
    Xorgate = cv2.bitwise_xor(ROI, Addaptive_Mean)
    Nerves = cv2.bitwise_and(ROI, Xorgate)


    num_labels, component_labels, stats, centroids = cv2.connectedComponentsWithStats(Nerves, connectivity=8)

    new_img = np.zeros_like(Nerves)

    for i in range(1, num_labels):  
        if stats[i, cv2.CC_STAT_AREA] >= 5:
            new_img[component_labels == i] = 255

    dilate = Dialate(new_img, 2)

    new_img = np.uint8(new_img)
    M = cv2.moments(new_img)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(new_img, (cX, cY), 5, (255, 0, 0), -1)
    cv2.putText(new_img, (f'Centroid = X : {cX}, Y :{cY}.'), (cX - 50, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

    Compare(dilate, new_img)


def Deer():
    Deer_img = cv2.imread(os.path.join(Path, "02_Deer-camera.jpg"))
    Deer_rgb = cv2.cvtColor(Deer_img, cv2.COLOR_BGR2RGB)
    Deer_hsv = cv2.cvtColor(Deer_img, cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(Deer_hsv)

    denoise = cv2.fastNlMeansDenoising(S, None, 
                                    21, #Higher Lower noise smoother edge 
                                    7, # use with color
                                    21 # Higher Lower noise smoother edge
                                    )

    Deer = cv2.inRange(denoise, 20, 120)
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

    Compare(markers, Deer_rgb)


#def Banana():

output_path = "//mnt//C0E08C6AE08C690C//work//my_work//Second_year//Second Semester//Image Processing//FinalProject//Output//Banana"

Mas_path = os.path.join(Path, "Banana/กล้วยไข่")
Awak_path = os.path.join(Path, "Banana/กล้วยน้ำว้า")
Cavendish_path = os.path.join(Path, "Banana/กล้วยหอม")
Mas = []
Awak = []
Cavendish = []

for index in range(1, 59):
    name = f"Mas{index:02}.JPG"
    banana_img = cv2.imread(os.path.join(Mas_path, name))
    banana_hsv = cv2.cvtColor(banana_img, cv2.COLOR_BGR2HSV)
    banana_gray = cv2.cvtColor(banana_img, cv2.COLOR_BGR2GRAY)

    H, S, V = cv2.split(banana_hsv)

    _, thresh = cv2.threshold(S , 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contour, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    banana = np.zeros_like(banana_img)
    banana = cv2.fillPoly(banana, contour, color=255)

    # edge
    edge = cv2.Canny(banana_gray, 50, 50)
    edge_pixels = np.sum(edge>0)
    total_pixels = edge.shape[0]*edge.shape[1]
    edge_density = edge_pixels/total_pixels
    Edge_percent = f"{edge_density * 100:.2f}%"

    #mean color
    R, G, B = cv2.split(banana)
    banana_pixels = R == 255
    H_mean = H[banana_pixels].mean()
    H_mean = f"{H_mean:.2f}"

    # banana area
    banana_area = banana_pixels.sum()
    ratio = banana_area / total_pixels
    banana_percent = f"{ratio * 100:.2f}%"

    print(name,  H_mean, banana_percent, Edge_percent)

    Mas.append([name, H_mean, banana_percent, Edge_percent])


for index in range(3, 49):
    if index == 3:
        name = "Awak4.JPG"
    else: name = f"Awak{index:02}.JPG"
    banana_img = cv2.imread(os.path.join(Awak_path, name))
    banana_hsv = cv2.cvtColor(banana_img, cv2.COLOR_BGR2HSV)
    banana_gray = cv2.cvtColor(banana_img, cv2.COLOR_BGR2GRAY)

    H, S, V = cv2.split(banana_hsv)

    _, thresh = cv2.threshold(S , 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contour, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    banana = np.zeros_like(banana_img)
    banana = cv2.fillPoly(banana, contour, color=255)

    # edge
    edge = cv2.Canny(banana_gray, 50, 50)
    edge_pixels = np.sum(edge>0)
    total_pixels = edge.shape[0]*edge.shape[1]
    edge_density = edge_pixels/total_pixels
    Edge_percent = f"{edge_density * 100:.2f}%"

    #mean color
    R, G, B = cv2.split(banana)
    banana_pixels = R == 255
    H_mean = H[banana_pixels].mean()
    H_mean = f"{H_mean:.2f}"

    # banana area
    banana_area = banana_pixels.sum()
    ratio = banana_area / total_pixels
    banana_percent = f"{ratio * 100:.2f}%"

    print(name,  H_mean, banana_percent, Edge_percent)

    # cv2.imwrite(os.path.join(output_path, name), banana)
    Awak.append([name, H_mean, banana_percent, Edge_percent])


for index in range(4, 49):
    name = f"Cavendish{index:02}.JPG"
    banana_img = cv2.imread(os.path.join(Cavendish_path, name))
    banana_hsv = cv2.cvtColor(banana_img, cv2.COLOR_BGR2HSV)
    banana_gray = cv2.cvtColor(banana_img, cv2.COLOR_BGR2GRAY)

    H, S, V = cv2.split(banana_hsv)

    _, thresh = cv2.threshold(S , 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contour, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    banana = np.zeros_like(banana_img)
    banana = cv2.fillPoly(banana, contour, color=255)

    # edge
    edge = cv2.Canny(banana_gray, 50, 50)
    edge_pixels = np.sum(edge>0)
    total_pixels = edge.shape[0]*edge.shape[1]
    edge_density = edge_pixels/total_pixels
    Edge_percent = f"{edge_density * 100:.2f}%"

    #mean color
    R, G, B = cv2.split(banana)
    banana_pixels = R == 255
    H_mean = H[banana_pixels].mean()
    H_mean = f"{H_mean:.2f}"

    # banana area
    banana_area = banana_pixels.sum()
    ratio = banana_area / total_pixels
    banana_percent = f"{ratio * 100:.2f}%"

    print(name,  H_mean, banana_percent, Edge_percent)

    # cv2.imwrite(os.path.join(output_path, f"กล้วยหอม//{name}"), banana)
    Cavendish.append([name, H_mean, banana_percent, Edge_percent])

import csv
BanType = [Mas, Cavendish, Awak]
name = ["Mas", "Cavendish", "Awak"]
index = 0
for List in BanType:
    # One CSV file per banana type
    filename = f'banana_{name[index]}.csv'
    index += 1

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'H_mean', 'banana_percent', 'Edge_percent'])

        for i in List : 
            writer.writerow([i[0], i[1], i[2], i[3]])

print(Mas)
print(Awak)
print(Cavendish)

