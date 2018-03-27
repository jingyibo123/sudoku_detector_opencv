import cv2
import numpy as np
import os


SZ=28 # size of sample
NB_BIN = 16 # number of bins
NB_R_AREA = 5 # number of rows of the sample to divide
NB_C_AREA = 5 # number of columns of the sample to divide
NB_AREA = NB_R_AREA * NB_C_AREA # number of areas of the sample to divide
SVM_C = 2.67
SVM_GAMMA = 5.383

SVM_FILE_NAME = '../../resource/mysvm.yml.gz'
digit_samples_path = '../../resource/generatedDigits/28x28_28x28/'

affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags, borderMode=cv2.BORDER_CONSTANT, borderValue=(255))
    return img

def hog(img):

    r = img.shape[0] # number of rows
    c = img.shape[1] # number of columns
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy) # 28*28
    bins = np.int8(np.rint((NB_BIN-1)*ang/(2*np.pi)))    # quantizing binvalues in (0...16)
    
    # 4 * 4
    rs = [r//NB_R_AREA, 2 * r//NB_R_AREA, 3 * r//NB_R_AREA, 4 * r//NB_R_AREA]
    cs = [c//NB_C_AREA, 2 * c//NB_C_AREA, 3 * c//NB_C_AREA, 4 * c//NB_C_AREA]
    bin_cells = bins[:rs[0],:cs[0]],      bins[rs[0]:rs[1],:cs[0]],        bins[rs[1]:rs[2],:cs[0]],       bins[rs[2]:rs[3],:cs[0]],       bins[rs[3]:,:cs[0]],\
                bins[:rs[0],cs[0]:cs[1]], bins[rs[0]:rs[1],cs[0]:cs[1]],   bins[rs[1]:rs[2],cs[0]:cs[1]],  bins[rs[2]:rs[3],cs[0]:cs[1]],  bins[rs[3]:,cs[0]:cs[1]],\
                bins[:rs[0],cs[1]:cs[2]], bins[rs[0]:rs[1],cs[1]:cs[2]],   bins[rs[1]:rs[2],cs[1]:cs[2]],  bins[rs[2]:rs[3],cs[1]:cs[2]],  bins[rs[3]:,cs[1]:cs[2]],\
                bins[:rs[0],cs[2]:cs[3]], bins[rs[0]:rs[1],cs[2]:cs[3]],   bins[rs[1]:rs[2],cs[2]:cs[3]],  bins[rs[2]:rs[3],cs[2]:cs[3]],  bins[rs[3]:,cs[2]:cs[3]],\
                bins[:rs[0],cs[3]:],      bins[rs[0]:rs[1],cs[3]:],        bins[rs[1]:rs[2],cs[3]:],       bins[rs[2]:rs[3],cs[3]:],       bins[rs[3]:,cs[3]:]
    mag_cells = mag[:rs[0],:cs[0]],      mag[rs[0]:rs[1],:cs[0]],        mag[rs[1]:rs[2],:cs[0]],       mag[rs[2]:rs[3],:cs[0]],       mag[rs[3]:,:cs[0]],\
                mag[:rs[0],cs[0]:cs[1]], mag[rs[0]:rs[1],cs[0]:cs[1]],   mag[rs[1]:rs[2],cs[0]:cs[1]],  mag[rs[2]:rs[3],cs[0]:cs[1]],  mag[rs[3]:,cs[0]:cs[1]],\
                mag[:rs[0],cs[1]:cs[2]], mag[rs[0]:rs[1],cs[1]:cs[2]],   mag[rs[1]:rs[2],cs[1]:cs[2]],  mag[rs[2]:rs[3],cs[1]:cs[2]],  mag[rs[3]:,cs[1]:cs[2]],\
                mag[:rs[0],cs[2]:cs[3]], mag[rs[0]:rs[1],cs[2]:cs[3]],   mag[rs[1]:rs[2],cs[2]:cs[3]],  mag[rs[2]:rs[3],cs[2]:cs[3]],  mag[rs[3]:,cs[2]:cs[3]],\
                mag[:rs[0],cs[3]:],      mag[rs[0]:rs[1],cs[3]:],        mag[rs[1]:rs[2],cs[3]:],       mag[rs[2]:rs[3],cs[3]:],       mag[rs[3]:,cs[3]:]
                
    # 3 * 3    
    # bin_cells = bins[:9,:10], bins[10:19,:10], bins[19:,:10], bins[:9, 10:19], bins[10:19, 10:19], bins[19:, 10:19], bins[:9, 19:], bins[10:19, 19:], bins[19:, 19:]
    # mag_cells = mag[:9,:10], mag[10:19,:10], mag[19:,:10], mag[:9, 10:19], mag[10:19, 10:19], mag[19:, 10:19], mag[:9, 19:], mag[10:19, 19:], mag[19:, 19:]
    
    # 2 * 3 
    # bin_cells = bins[:9,:14], bins[10:19,:14], bins[19:,:14],bins[:9, 14:], bins[10:19, 14:], bins[19:, 14:]
    # mag_cells = mag[:9,:14], mag[10:19,:14], mag[19:,:14],mag[:9, 14:], mag[10:19, 14:], mag[19:, 14:]
    
    # 2 * 2
    # bin_cells = bins[:14,:14], bins[14:,:14], bins[:14,14:], bins[14:,14:]
    # mag_cells = mag[:14,:14], mag[14:,:14], mag[:14,14:], mag[14:,14:]
    
    hists = [np.bincount(b.ravel(), m.ravel(), NB_BIN) for b, m in zip(bin_cells, mag_cells)] # NB_AREA * array(4*4)
    
    hist = np.hstack(hists)     # hist is a NB_BIN * NB_AREA bit vector
    
    return hist


def generate_blur_imgs():

    directory = os.path.realpath(digit_samples_path)

    for digit in range(0, 10):
        for filename in os.listdir(directory+"\\"+str(digit)):
            if 'blur' in filename:
                os.remove(directory + "\\" + str(digit) + "\\" + filename)
        for filename in os.listdir(directory+"\\"+str(digit)):
            
            if 'blur' not in filename and 'erosion' not in filename and 'fromimage' not in filename:
                path = directory + "\\" + str(digit) + "\\" + filename
                
                img = cv2.imread(path)
                
                # blur = cv2.GaussianBlur(img,(5, 5),0)
                # cv2.imwrite(path.rstrip(".png") + "_blur5.png", blur)
                
                blur = cv2.GaussianBlur(img,(9, 9),0)
                cv2.imwrite(path.rstrip(".png") + "_blur9.png", blur)


def generate_erosion_imgs():
    import sudoku_detector
    directory = os.path.realpath(digit_samples_path)
    
    for digit in range(0, 10):
        for filename in os.listdir(directory+"\\"+str(digit)):
            if 'erosion' in filename:
                os.remove(directory + "\\" + str(digit) + "\\" + filename)
        for filename in os.listdir(directory+"\\"+str(digit)):
            
            if 'blur' not in filename and 'erosion' not in filename and 'fromimage' not in filename:
                path = directory + "\\" + str(digit) + "\\" + filename
                
                img = cv2.imread(path)
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
                erosion = cv2.dilate(img,kernel,iterations = 1)
                cv2.imwrite(path.rstrip(".png") + "_erosion1.png", erosion)
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                erosion = cv2.dilate(img,kernel,iterations = 1)
                cv2.imwrite(path.rstrip(".png") + "_erosion2.png", erosion)


def generate_28x28_20x20_imgs():
    from sudoku_detector import add_digit_border
    
    directory = os.path.realpath(digit_samples_path)
    
    for digit in range(0, 10):
        for filename in os.listdir(directory+"\\"+str(digit)):
            if 'blur' in filename or 'erosion' in filename or 'fromimage' in filename:
                os.remove(directory + "\\" + str(digit) + "\\" + filename)
                
        for filename in os.listdir(directory+"\\"+str(digit)):
            
            if 'blur' not in filename and 'erosion' not in filename and 'fromimage' not in filename:
                path = directory + "\\" + str(digit) + "\\" + filename
                # read image
                img = cv2.imread(path)
                # delete image
                os.remove(directory + "\\" + str(digit) + "\\" + filename)
                
                img = add_digit_border(img, 28, 28, 20)

                # write 20x20 image
                cv2.imwrite(path, img)


def train_svm():
    
    
    directory = os.path.realpath(digit_samples_path)
    nb_file = len(os.listdir(directory+"\\1"))

    # read all images
    cells = [] # 9*nb_img*28*28
    for digit in range(1, 10):
        lines = []
        for filename in os.listdir(directory+"\\"+str(digit)):
            img = cv2.imread(directory + "\\" + str(digit) + "\\" + filename)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                
            lines.append(gray)
        cells.append(lines)
        
    # generate responses
    responses = np.repeat(np.arange(1, 10),nb_file)[:,np.newaxis]
    
    # preprocessing
    deskewed = [[deskew(i) for i in row] for row in cells]
    hogdata = [[hog(i) for i in row] for row in deskewed]
    trainData = np.float32(hogdata).reshape(-1,NB_BIN * NB_AREA) # 140*400
    
    # create svm
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(SVM_C)
    svm.setGamma(SVM_GAMMA)
    
    # train and save
    svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    svm.save(SVM_FILE_NAME)

    
def getDigitSamplesInText():
    
    directory = os.path.realpath(digit_samples_path)
    nb_file = len(os.listdir(directory+"\\1"))

    # read all images
    for digit in range(1, 10):
        
        with open(digit_samples_path + str(digit)+'.txt', 'w') as f:
            for filename in os.listdir(directory+"\\"+str(digit)):
                f.write("%s\n" % filename)
    

    
if __name__ == '__main__':

    # generate_erosion_imgs()
    
    # generate_blur_imgs()
    
    # train_svm()
    
    # getDigitSamplesInText()
    
    # generate_28x28_20x20_imgs()
    
    
    pass