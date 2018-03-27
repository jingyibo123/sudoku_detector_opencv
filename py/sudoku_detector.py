import cv2
import numpy as np
from timer import Timer
import svm


SZ = 9 # dimension of sudoku

# ----Parameters of finding grid
PARAM_GRID_BLUR = 3
PARAM_GRID_THRES_BLOCKSIZE = 11
PARAM_GRID_THRES_C = 2
# denoise
PARAM_GRID_OPEN_KERNEL = cv2.MORPH_ELLIPSE
PARAM_GRID_OPEN_KERNEL_SIZE = 2
# fill holes
PARAM_GRID_CLOSE_KERNEL = cv2.MORPH_ELLIPSE
PARAM_GRID_CLOSE_KERNEL_SIZE = 5
# Threshold of size proportion of grid of the whole image
PARAM_GRID_SIZE_THRES = 1/25

# ----Parameters of locating digit
PARAM_CELL_BLUR = 7
PARAM_CELL_THRES_BLOCKSIZE = 15
PARAM_CELL_THRES_C = 2
# denoise
PARAM_CELL_OPEN_KERNEL = cv2.MORPH_ELLIPSE
PARAM_CELL_OPEN_KERNEL_SIZE = 2


# debugFindGrid = True;
debugFindGrid = False;
timer = Timer(False)
mySVM = cv2.ml.SVM_create()
mySVM = cv2.ml.SVM_load(svm.SVM_FILE_NAME)
timer.timeit('load mySVM machine')

def parse_sudoku(img_path):
    # print(img_path)

    timer.timeit()
    
    img = cv2.imread(img_path)
    
    timer.timeit('read')
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    timer.timeit("convert to gray")
    
    grid = find_grid(gray)
    if grid is None:
        return None

#     cv2.imwrite('../../resource/output/grid/' + img_path[img_path.rfind('/'):], grid)
    
    timer.timeit('find grid')

    digit_images = extract_digits(grid)
    
    
    digits = [0] * SZ * SZ
    
    for pos, digit in digit_images.items():
        # digit = svm.deskew(digit)
        hogdata = svm.hog(digit)
        testData = np.float32(hogdata).reshape(-1,svm.NB_BIN * svm.NB_AREA)
        
        result = mySVM.predict(testData)[1][0][0]
        digits[pos] = (int(result))
        
    timer.total_time(' ')
    return digits


def find_grid(img):
    """ find the biggest rectangle in the given image. 

    Args:
        img: the input openCV gray image 

    Returns:
        None if no such rectangle is found.
        The image of biggest rectangle after perspective transformation
    """
    
    timer.timeit()
    
    size = img.shape[0]
    
    if debugFindGrid:
        cv2.imshow('diff',img)
        cv2.waitKey(0)
    
    # more blur to eliminate noise
    img = cv2.GaussianBlur(img,(PARAM_GRID_BLUR, PARAM_GRID_BLUR),0)
    # img = cv2.bilateralFilter(img, 15, 60, 60)
    
    timer.timeit('find grid blur')
    
    # Here use low C param to avoid missing line segments
    thresh_raw = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,PARAM_GRID_THRES_BLOCKSIZE,PARAM_GRID_THRES_C)
    
    
    if debugFindGrid:
        cv2.imshow('diff',thresh_raw)
        cv2.waitKey(0)
    
    # Use morphology to remove noise
    kernel = cv2.getStructuringElement(PARAM_GRID_OPEN_KERNEL, (PARAM_GRID_OPEN_KERNEL_SIZE, PARAM_GRID_OPEN_KERNEL_SIZE))
    thresh_raw = cv2.morphologyEx(thresh_raw, cv2.MORPH_OPEN, kernel)
    
    if debugFindGrid:
        cv2.imshow('diff',thresh_raw)
        cv2.waitKey(0)
    
    # Fill possible holes in lines
    kernel = cv2.getStructuringElement(PARAM_GRID_CLOSE_KERNEL, (PARAM_GRID_CLOSE_KERNEL_SIZE, PARAM_GRID_CLOSE_KERNEL_SIZE)) 
    thresh_raw = cv2.morphologyEx(thresh_raw, cv2.MORPH_CLOSE, kernel)
    
    if debugFindGrid:
        cv2.imshow('diff',thresh_raw)
        cv2.waitKey(0)
    
    # Use morphology to remove all noise (including digits) other than long lines
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size // 20))
    # v_thresh = cv2.morphologyEx(thresh_raw, cv2.MORPH_OPEN, kernel)
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size // 20, 1))
    # h_thresh = cv2.morphologyEx(thresh_raw, cv2.MORPH_OPEN, kernel)
    
    # thresh_raw = cv2.bitwise_or(v_thresh, h_thresh)
    
    # if debugFindGrid:
        # cv2.imshow('diff',thresh_raw)
        # cv2.waitKey(0)
    
    timer.timeit("find grid threshold and denoising")
    
    # find largest rectangle
    _, contours, hierarchy = cv2.findContours(thresh_raw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    timer.timeit('find grid contour')
    
    grids = dict()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > img.size * PARAM_GRID_SIZE_THRES:
            peri = cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,0.04*peri,True)
            # put all possible grids
            if len(approx)==4:
                # print(approx)
                grids[area] = approx
            
    if len(grids) == 0:
        return None
    
    timer.timeit('find grid filter contour')
    
    # sort the grids from largest to smallest
    areas = list(grids.keys())
    areas.sort(reverse=True)
    
    for area in areas:
        grid = grids[area]
        square = sort_grid_vertex(grid)
        # see if parallel sides are of similar length
        # TODO CONSTANT
        thres_diff_length = 0.5 # TODO need to test case by case
        l1 = (square[1][0] - square[0][0])*(square[1][0] - square[0][0]) \
            + (square[1][1] - square[0][1])*(square[1][1] - square[0][1])
        
        l2 = (square[2][0] - square[1][0])*(square[2][0] - square[1][0]) \
            + (square[2][1] - square[1][1])*(square[2][1] - square[1][1])
            
        l3 = (square[3][0] - square[2][0])*(square[3][0] - square[2][0]) \
            + (square[3][1] - square[2][1])*(square[3][1] - square[2][1])
            
        l4 = (square[0][0] - square[3][0])*(square[0][0] - square[3][0]) \
            + (square[0][1] - square[3][1])*(square[0][1] - square[3][1])
            
        if l1 < (1-thres_diff_length) * l3 or l1 > (1+thres_diff_length) * l3 \
        or l2 < (1-thres_diff_length) * l4 or l2 > (1+thres_diff_length) * l4:
            continue
            
        grid_rows = int(square[3][1] - square[0][1] + square[2][1] - square[1][1]) // 2
        grid_cols = int(square[1][0] - square[0][0] + square[2][0] - square[3][0]) // 2
        
        h = np.array([ [0,0],[grid_cols - 1,0],[grid_cols - 1,grid_rows - 1],[0,grid_rows - 1] ],np.float32)
        # perspective transformation
        retval = cv2.getPerspectiveTransform(square,h)
        grid_img = cv2.warpPerspective(img,retval,(grid_cols, grid_rows))
        
        
        if debugFindGrid:
            cv2.imshow('diff',grid_img)
            cv2.waitKey(0)
            
        return grid_img
        '''
        # see if there's at least the principle lines
        v_lines, h_lines = detect_lines(grid_img)
        # TODO CONSTANT
        threshold_line_shift = 1/(50*50)
        
        nb_v = 0
        for v_line in v_lines:
            if (v_line - grid_cols//3)/grid_cols * (v_line - grid_cols//3)/grid_cols < threshold_line_shift:
                nb_v = nb_v + 1
            if (v_line - 2*grid_cols//3)/grid_cols * (v_line - 2*grid_cols//3)/grid_cols < threshold_line_shift:
                nb_v = nb_v + 1
                
        nb_h = 0
        for h_line in h_lines:
            if (h_line - grid_rows//3)/grid_rows * (h_line - grid_rows//3)/grid_rows < threshold_line_shift:
                nb_h = nb_h + 1
            if (h_line - 2*grid_rows//3)/grid_rows * (h_line - 2*grid_rows//3)/grid_rows < threshold_line_shift:
                nb_h = nb_h + 1
        
        timer.timeit('find grid validate square')
    
        if nb_v == 2 and nb_h == 2:
            if debugFindGrid:
                cv2.imshow('diff',grid_img)
                cv2.waitKey(0)
            return grid_img
        '''
    
    return None

    
def sort_grid_vertex(pts):

    pts = pts.reshape((4,2))
    square = np.zeros((4,2), dtype = np.float32)
    
    sum = pts.sum(1)
    square[0] = pts[np.argmin(sum)]
    square[2] = pts[np.argmax(sum)]
    diff = np.diff(pts,axis = 1)
    square[1] = pts[np.argmin(diff)]
    square[3] = pts[np.argmax(diff)]
    return square


def extract_digits(grid):
    """ extract all the digits in the gri. 

    Args:
        img: the input gray image of the found grid

    Returns:
        dict of digit 
            key : pos of cell
            value : images with proper border ready for classification, 
    """
    
    cell_size_raw = grid.shape[0] // SZ
    nb_lines_correct = False
    # use hough to detect lines 
    # v_lines, h_lines = detect_lines(grid)
    # divide the cell manually if lines found were wrong
    # TODO if 10 lines found but wrong position
    # nb_lines_correct = len(v_lines) == SZ + 1 and len(h_lines) == SZ + 1
    
    
    timer.timeit('detect lines')
    
    if not nb_lines_correct:
        # print("wrong number of lines detected")
        v_lines = [int(i * grid.shape[1] / SZ) for i in range(0, SZ + 1)] 
        h_lines = [int(i * grid.shape[0] / SZ) for i in range(0, SZ + 1)] 

    # iterate all cells

    digit_images = {}
    # blur and thresholding
    # grid_blur = cv2.bilateralFilter(grid, 15, 80, 80)  # will add 70% !!!!! processing time
    grid_blur = cv2.GaussianBlur(grid,(PARAM_CELL_BLUR, PARAM_CELL_BLUR),0)
    
    # if n == 75:
        # cv2.imshow('diff',grid_blur)
        # cv2.waitKey(0)
        
    
    grid_thresh = cv2.adaptiveThreshold(grid_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, PARAM_CELL_THRES_BLOCKSIZE, PARAM_CELL_THRES_C)
    
    # Use morphology to remove noise
    kernel = cv2.getStructuringElement(PARAM_CELL_OPEN_KERNEL, (PARAM_CELL_OPEN_KERNEL_SIZE, PARAM_CELL_OPEN_KERNEL_SIZE))
    grid_thresh = cv2.morphologyEx(grid_thresh, cv2.MORPH_OPEN, kernel)
    for n in range(81): 
            
        cell_raw = grid_thresh[h_lines[n//SZ]:h_lines[n//SZ + 1], v_lines[n%SZ]:v_lines[n%SZ + 1]]

        digit = locate_digit_in_cell(n, cell_raw)
            
        if digit is None:
            continue

        # result is too large => locating failed
        if digit.shape[1] / cell_raw.shape[1] > 0.9:
            # print("failed to locate cell no. ", n)
            continue
        
        digit_image = add_digit_border(digit, 28, 28, 28)
        digit_images[n] = digit_image
    return digit_images


def detect_lines(grid):
    """accept a square image(without blur) returns two sets(x/y) of coordinates of detected lines"""
    
    # timer.timeit()
    
    vertical = []
    horizontal = []
    
    height = grid.shape[0]
    width = grid.shape[1]
    
    # more blur to eliminate noise
    blur2 = cv2.GaussianBlur(grid,(5, 5),0)
    # Here use low C param to avoid missing line segments
    thresh_raw = cv2.adaptiveThreshold(blur2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
    

    # Fill possible missing boundaries in pattern
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh_raw = cv2.morphologyEx(thresh_raw, cv2.MORPH_CLOSE, kernel)
    
    # Use morphology to remove all noise (including digits) other than long lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 15))
    v_thresh = cv2.morphologyEx(thresh_raw, cv2.MORPH_OPEN, kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 15, 1))
    h_thresh = cv2.morphologyEx(thresh_raw, cv2.MORPH_OPEN, kernel)
    
    lines_thresh = cv2.bitwise_or(v_thresh, h_thresh)
    
    edges = cv2.Canny(lines_thresh,50,150,apertureSize = 3)
    
    lines = cv2.HoughLinesP(lines_thresh,rho = 1,theta = 10*np.pi/180,threshold = 70, minLineLength = height * 0.15,maxLineGap = 100)
    # timer.timeit('hough lines')
    
    # filter the lines to only keep vertical and horizontal lines
    v_raw = []
    h_raw = []
    # threshold of line shift between ends
    e = (height / SZ / 10) 
    if lines is None:
        return [vertical, horizontal]
    
    for line in lines:
        if line is None:
            continue
        x1,y1,x2,y2 = line[0]
        # vertical
        if (y1 - y2) < e and (y1 - y2) > -e:
            v_raw.append((y1 + y2) / 2)
        # horizontal
        if (x1 - x2) < e and (x1 - x2) > -e:
            h_raw.append((x1 + x2) / 2)
    if len(h_raw) == 0 or len(v_raw) == 0:
        return [[], []]
    # sort the lines by their position
    v_raw = sorted(v_raw)
    h_raw = sorted(h_raw)
    
    # merge adjacent lines into one line
    el = (height / SZ / 7) # threshold of line width
    current = v_raw[0]
    temp = []
    for v in v_raw:
        if (current - v) > el or (current - v) < -el:
            vertical.append(int(round(sum(temp)/len(temp))))
            del temp[:]
        temp.append(v)
        current = v
    if temp:
        vertical.append(int(round(sum(temp)/len(temp))))
    
    current = h_raw[0]
    temp = []
    for h in h_raw:
        if (current - h) > el or (current - h) < -el:
            horizontal.append(int(round(sum(temp)/len(temp))))
            del temp[:]
        temp.append(h)
        current = h
    if temp:
        horizontal.append(int(round(sum(temp)/len(temp))))
        
    # timer.timeit('filter lines')

    return [vertical, horizontal]


def locate_digit_in_cell(n, cell_thresh):

    """ locate the digit pattern given image of a rectangle cell 

    Args:
        cell_thresh: the input openCV thresh of a cell after blur.

    Returns:
        None if nothing is found in the center of the cell;
        The uncropped original image in binary color after thresholding if digit exists but can't be properly located;
        The most approximate rectangle containing the digit in binary color.(digit in black)
    """
    
    
    h = cell_thresh.shape[0]
    w = cell_thresh.shape[1]
    
    # count the active pixels in the center to see if it contains a digit
    cell_thresh_center = cell_thresh[int(h*0.25) : int(h*0.75), int(w*0.25) : int(w*0.75)]
    n_active_pixels = cv2.countNonZero(cell_thresh_center)
    if (n_active_pixels/cell_thresh_center.size > 0.85):
        return None
        
    
    # timer.timeit('detect if digit')
    
    # manually paint white all borders to avoid digit laying on border which fails to locate digit
    cell_thresh[h-1, :] = 255
    cell_thresh[0, :] = 255
    cell_thresh[:, w-1] = 255
    cell_thresh[:, 0] = 255

    # locate the digit in cell
    _, cell_coutours, cell_hierarchy = cv2.findContours(cell_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_coutour = None
    max_radius = 0
    for cell_coutour in cell_coutours:
        center, radius = cv2.minEnclosingCircle(cell_coutour)
        # print(center, radius)
        if radius < w  * 0.45 and radius > w  * 0.20 and radius > max_radius and(center[0] - h/2)*(center[0] - h/2) < h * h / 16 and (center[1] - w/2)*(center[1] - w/2) < w * w / 16:
            max_radius = radius 
            digit_coutour = cell_coutour
            
    if digit_coutour is None:
        return cell_thresh
        
    # crop the digit   ******** adds 3% processing time for eval_all
    # mask = np.full(cell_thresh.shape[:2], 0, dtype=np.uint8)
    # cv2.drawContours(mask, [digit_coutour], 0, 1, cv2.FILLED);
    # cell_thresh = cell_thresh * (mask)
    
    mask = np.full(cell_thresh.shape[:2], 255, dtype=np.uint8)
    cv2.drawContours(mask, [digit_coutour], 0, 0, cv2.FILLED);
    cell_thresh = cv2.bitwise_or(cell_thresh,mask)
    
    rect = cv2.boundingRect(digit_coutour)
    # --------original version
    cell_rect = cell_thresh[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    return cell_rect



def add_digit_border(digit, sample_height, sample_width, sample_digit_height):
    """ add white borders to an extracted digit pattern to match ML input format 
    Args:
        digit: a binary rectangle white-background image of the digit.
        sample_height: height of samplein pixels.
        sample_width: width of sample in pixels.
        sample_digit_height: the height of the digit in sample in pixels.

    Returns:
        None sample_digit_height exceeds sample_height;
        The generated image with borders
    """

    if sample_digit_height > sample_height:
        return None;
        
    h = digit.shape[0]
    w = digit.shape[1]
    sample_digit_width = int(round(w * sample_digit_height / h))
    # detected digit is too wide
    if(sample_digit_width > sample_width):
        sample_digit_width =  sample_width
        sample_digit_height = int(round(h * sample_digit_width / w))
        
    new_digit = cv2.resize(digit,(sample_digit_width, sample_digit_height),interpolation=cv2.INTER_AREA)
        
    if len(digit.shape) == 3 and digit.shape[2] == 3:
        borderColor = (255, 255, 255)
    if len(digit.shape) == 2 or digit.shape[2] == 1:
        borderColor = (255)

    sample = cv2.copyMakeBorder(new_digit, top=(sample_height-sample_digit_height)//2,
                                bottom=sample_height-sample_digit_height-(sample_height-sample_digit_height)//2,
                                left=(sample_width-sample_digit_width)//2,
                                right=sample_width - sample_digit_width-(sample_width-sample_digit_width)//2,
                                 borderType= cv2.BORDER_CONSTANT, value=borderColor)

    return sample





