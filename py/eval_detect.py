import cv2
import numpy as np
from timer import Timer
from sudoku_detector import parse_sudoku, find_grid, extract_digits, locate_digit_in_cell
from solve import solver
from svm import train_svm, SZ
import os
import json
import xlsxwriter
from datashape.coretypes import int32


SZ = 9

timer = Timer(True)
image_path = '../../resource/images/all/'
generate_digit_path = '../../resource/extractedDigits/'
generate_cell_path = '../../resource/extractedCells/'

def search_testfiles(path = image_path):
    filenames = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            filenames.append(filename.rstrip(".jpg"))
        if filename.endswith(".png"):
            filenames.append(filename.rstrip(".png"))
    return filenames


def extract_digit_samples_from_puzzles():
    filenames = search_testfiles()
    
    for name in filenames:
        expected = load_dat(name)
        
        img = cv2.imread(image_path + name + '.jpg')
    
    
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        grid = find_grid(gray)
        if grid is None:
            continue
    
        digit_images = extract_digits(grid)
        for pos, digit in digit_images.items():
            cv2.imwrite(generate_digit_path+'all/from'+name+'_digit'+str(expected[pos])+'_pos'+str(pos)+'.png', digit)
            cv2.imwrite(generate_digit_path+str(expected[pos])+'/'+'from'+name+'_digit'+str(expected[pos])+'_pos'+str(pos)+'.png', digit)


def extract_cells_from_puzzles():
    filenames = search_testfiles(image_path)
    
    # read from file
    with open('../../resource/images_grid_pts.txt', 'r') as f:
      img_grid_pts = json.load(f)
    
    for name in filenames:
      expected = load_dat(name)
      
      img = cv2.imread(image_path + name + '.jpg')
  
      square = np.asarray(img_grid_pts[name], np.float32)
      
      grid_rows = int(square[3][1] - square[0][1] + square[2][1] - square[1][1]) // 2
      grid_cols = int(square[1][0] - square[0][0] + square[2][0] - square[3][0]) // 2
      
      h = np.array([ [0,0],[grid_cols - 1,0],[grid_cols - 1,grid_rows - 1],[0,grid_rows - 1] ],np.float32)
      # perspective transformation
      retval = cv2.getPerspectiveTransform(square,h)
      grid = cv2.warpPerspective(img,retval,(grid_cols, grid_rows))
        
      v_lines = [int(i * grid.shape[1] / SZ) for i in range(0, SZ + 1)] 
      h_lines = [int(i * grid.shape[0] / SZ) for i in range(0, SZ + 1)] 

      for n in range(81):
            
        cell = grid[h_lines[n//SZ]:h_lines[n//SZ + 1], v_lines[n%SZ]:v_lines[n%SZ + 1]]
        cell = cv2.resize(cell, (128, 128));
        cv2.imwrite(generate_cell_path+'all/from'+name+'_digit'+str(expected[n])+'_pos'+str(n)+'.png', cell)
        cv2.imwrite(generate_cell_path+str(expected[n])+'/'+'from'+name+'_digit'+str(expected[n])+'_pos'+str(n)+'.png', cell)
        pass


def load_dat(filename):
    digits = []
    with open(image_path + filename +".dat") as f:
        txt_lines = f.readlines()[2:]
        for l in txt_lines:
            digits.extend([int(s) for s in l.strip().split(' ')])
    return(digits)


def eval_digits(result, expected):
    re = {}
    result = np.int8(result).ravel()
    expected = np.int8(expected).ravel()
    total = 0
    for ex in expected: 
        if ex != 0:
            total = total + 1
    re['total'] = total
    # re['total'] = int(np.count_nonzero(expected))
    re['missed'] = 0
    if np.count_nonzero(result) < np.count_nonzero(expected):
        re['missed'] = int(np.count_nonzero(expected) - np.count_nonzero(result))
    
    re['wrong'] = int(np.sum(result != expected) - re['missed'])
    
    return re


def eval_all():

    imgs = search_testfiles(image_path)
    
    incorrect = {}
    eval_result = {}
    timer.timeit()
    notGrid = 0
    for name in imgs:
        expected = load_dat(name)
        digits = parse_sudoku(image_path + name + '.jpg')
        if digits is None:
            print("grid not found", name)
            notGrid = notGrid + 1
            continue
        re = eval_digits(digits, expected)
        eval_result[name] = re
        if sum(digits) != 0:
            for i in range(81):
                if expected[i] != digits[i]:
                    inc = {'pos': i, 'expected': expected[i], 'result': digits[i]}
                    incorrect[name] = inc
                    
    print("grid not found : " + str(notGrid))
    
    timer.timeit('all images processed')
    print(incorrect)
    
    total = [r['total'] for k, r in eval_result.items()]
    miss = [r['missed'] for k, r in eval_result.items()]
    wrong = [r['wrong'] for k, r in eval_result.items()]
    print('total :', sum(total), 'correct :', sum(total)-sum(miss)-sum(wrong), 'missed :', sum(miss), 'wrong :', sum(wrong), 'correct ratio :', 1-sum(wrong)/(sum(total)-sum(miss)))
    return eval_result


def generate_digit_box():
  """
    use opencv to locate digit in grid images, output the rounding box coordinates and draw the box on the grid image.
  """
  imgs = search_testfiles('../../resource/output/grid/all/')
  
  # read from file 
  img_grid_pts = dict()
  
  for name in imgs:
    
    expected = load_dat(name)
    boxes = dict()
    
    grid = cv2.imread('../../resource/output/grid/all/' + name + '.png')
    grid_gray = cv2.cvtColor(grid,cv2.COLOR_BGR2GRAY)
    grid_blur = cv2.GaussianBlur(grid_gray,(7, 7),0)
    grid_thresh = cv2.adaptiveThreshold(grid_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    # Use morphology to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    grid_thresh = cv2.morphologyEx(grid_thresh, cv2.MORPH_OPEN, kernel)
  
    v_lines = [int(i * grid.shape[1] / SZ) for i in range(0, SZ + 1)] 
    h_lines = [int(i * grid.shape[0] / SZ) for i in range(0, SZ + 1)] 
    for n in range(81):
      if expected[n] == 0:
        continue
      
      cell_raw = grid_thresh[h_lines[n//SZ]:h_lines[n//SZ + 1], v_lines[n%SZ]:v_lines[n%SZ + 1]]
      
      digit = locate_digit_in_cell(n, cell_raw)
      if digit is None or len(digit) != 4 :
        print("digit not located", name, n)
        continue
      
      for d in digit:
        d[0] = d[0] + v_lines[n%SZ]
        d[1] = d[1] + h_lines[n//SZ]
      grid = cv2.line(grid, tuple(digit[0]), tuple(digit[1]), (255,0,0),1)
      grid = cv2.line(grid, tuple(digit[1]), tuple(digit[2]), (255,0,0),1)
      grid = cv2.line(grid, tuple(digit[2]), tuple(digit[3]), (255,0,0),1)
      grid = cv2.line(grid, tuple(digit[3]), tuple(digit[0]), (255,0,0),1)
      boxes[n] = digit
    

    cv2.imwrite('../../resource/output/grid/all_boxes/' + name + '.png', grid)
    img_grid_pts[name] = boxes
  
  
  # write to file
  with open('../../resource/grid_digit_boxes.txt', 'w') as f:
      json.dump(img_grid_pts, f)


def draw_digit_box():

  path = '../../resource/output/grid/all/'
  imgs = search_testfiles(path)
  
  # read from file
  with open('../../resource/grid_digit_boxes.txt', 'r') as f:
    img_grid_pts = json.load(f)
  
  for name in imgs:
    
    expected = load_dat(name)
    
    grid = cv2.imread('../../resource/output/grid/all/' + name + '.png')
    
    for n in range(81):
      if expected[n] == 0:
        continue
      
      if str(n) not in img_grid_pts[name]:
        print("box not defined for ", name, " cell no.", n)
        continue
      digit = img_grid_pts[name][str(n)]
      
      grid = cv2.line(grid, tuple(digit[0]), tuple(digit[1]), (255,0,0),1)
      grid = cv2.line(grid, tuple(digit[1]), tuple(digit[2]), (255,0,0),1)
      grid = cv2.line(grid, tuple(digit[2]), tuple(digit[3]), (255,0,0),1)
      grid = cv2.line(grid, tuple(digit[3]), tuple(digit[0]), (255,0,0),1)
    
    cv2.imwrite('../../resource/output/grid/all_boxes/' + name + '.png', grid)
    
    
def class_int_to_text(row_label):
  if row_label == 1:
      return 'one'
  if row_label == 2:
      return 'two'
  if row_label == 3:
      return 'three'
  if row_label == 4:
      return 'four'
  if row_label == 5:
      return 'five'
  if row_label == 6:
      return 'six'
  if row_label == 7:
      return 'seven'
  if row_label == 8:
      return 'eight'
  if row_label == 9:
      return 'nine'
  else:
      return None
    
    
def generate_grid_digits_label_csv():

  path = '../../resource/output/grid/all/'
  imgs = search_testfiles(path)
  
  # read from file
  with open('../../resource/grid_digit_boxes.txt', 'r') as f:
    img_grid_pts = json.load(f)
  
  sep = ";"
  with open('../../resource/grid_digit_labels.csv', 'w') as f:
    f.write("filename"+sep+"width"+sep+"height"+sep+"class"+sep+"xmin"+sep+"ymin"+sep+"xmax"+sep+"ymax\n")
    
    for name in imgs:
      expected = load_dat(name)
      
      grid = cv2.imread('../../resource/output/grid/all/' + name + '.png')
      
      for n in range(81):
        if expected[n] == 0:
          continue
        
        if str(n) not in img_grid_pts[name]:
          print("box not defined for ", name, " cell no.", n)
          continue
        
        f.write(name + ".png" + sep + str(grid.shape[1]) + sep + str(grid.shape[0]) + sep + class_int_to_text(expected[n]) + sep 
                + str(img_grid_pts[name][str(n)][0][0]) + sep + str(img_grid_pts[name][str(n)][0][1]) + sep 
                + str(img_grid_pts[name][str(n)][2][0]) + sep + str(img_grid_pts[name][str(n)][2][1]) + "\n")


def generate_puzzle_grid_label_csv():

  path = '../../resource/images/all/'
  imgs = search_testfiles(path)
  
  # read from file
  with open('../../resource/images_grid_pts.txt', 'r') as f:
    img_grid_pts = json.load(f)
  
  sep = ";"
  with open('../../resource/images_grid_pts.csv', 'w') as f:
    f.write("filename"+sep+"width"+sep+"height"+sep+"class"+sep+"xmin"+sep+"ymin"+sep+"xmax"+sep+"ymax\n")
    
    for name in imgs:
      
      img = cv2.imread('../../resource/images/all/' + name + '.jpg')
      
      max = np.asarray(img_grid_pts[name], dtype=np.int32).max(axis=0)
      min = np.asarray(img_grid_pts[name], dtype=np.int32).min(axis=0)
      
      f.write(name + ".jpg" + sep + str(img.shape[1]) + sep + str(img.shape[0]) + sep + "sudoku" + sep 
              + str(min[0]) + sep + str(min[1]) + sep 
              + str(max[0]) + sep + str(max[1]) + "\n")


def eval_one(name):
    expected = load_dat(name)
    digits = parse_sudoku(image_path + name + '.jpg')
    if digits is None:
        print("grid not found")
        return
    # TODO loop result to find wrong ones
    incorrect = []
    for i in range(81):
        if expected[i] != digits[i]:
            inc = {'pos': i, 'expected': expected[i], 'result': digits[i]}
            incorrect.append(inc)
    
    print("incorrect : ", incorrect)
    
    re = eval_digits(digits, expected)
    
    print(re)


def xlsx(re):
        
    # write to xlsx
    with xlsxwriter.Workbook('eval_all.xlsx') as book:

        # Raw data
        sheet = book.add_worksheet('raw')
        # fetch data
        # Fill worksheet
        
        # write column names
            
        sheet.write(0, 1, "Total")
        sheet.write(0, 2, "missed")
        sheet.write(0, 3, "wrong")
        
        j = 1
        
        for k, r in re.items():
        
            sheet.write(j, 0, k)
            sheet.write(j, 1, r['total'])
            sheet.write(j, 2, r['missed'])
            sheet.write(j, 3, r['wrong'])
            j = j + 1

    
if __name__ == '__main__':

    # train_svm()
    
    # eval_one("image1024")# high resolution, clear, mid curl
    # eval_one("image211") # middle resolution, clear, mid curl
    # eval_one("image25") # low resolution, blurry, mid curl
    # eval_one("image17") # low resolution, blurry
    # parse_sudoku("../../resource/cascade.png")
    
    # eval_one("image153") # easy 
    # eval_one("image34") # blurry
    # eval_one("image1087")
    # re = eval_all()


    # extract_cells_from_puzzles()

    # extract_digit_samples_from_puzzles()

    # xlsx(re)
    











