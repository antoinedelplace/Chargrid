import numpy as np
import matplotlib.pyplot as plt
import os

## Hyperparameters
dir_np_chargrid = "./data/np_chargrids/"
dir_np_gt = "./data/np_gt/"
outdir_np_chargrid_reduced = "./data/np_chargrids_reduced/"
outdir_png_chargrid_reduced = "./data/img_chargrids_reduced/"
outdir_np_gt_reduced = "./data/np_gt_reduced/"
outdir_png_gt_reduced = "./data/img_gt_reduced/"
list_filenames = [f for f in os.listdir(dir_np_chargrid) if os.path.isfile(os.path.join(dir_np_chargrid, f))]
equal_threshold = 0.95
max_padding = 3

def get_reduce(img, axis):
    reduce_f = 1
    
    trust = 1.0
    reduce = 1
    #print(img.shape[axis])
    while reduce <= img.shape[axis]/2:
        reduce += 1
        if img.shape[axis]%reduce == 0:
            if axis == 0:
                img_reshaped = img.reshape(img.shape[0]//reduce, -1, img.shape[1])
                img2 = np.repeat(np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=img_reshaped), reduce, axis=axis)
            else:
                img_reshaped = img.reshape(img.shape[0], img.shape[1]//reduce, -1)
                img2 = np.repeat(np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=2, arr=img_reshaped), reduce, axis=axis)
            trust = np.sum(img == img2)/(np.shape(img)[0]*np.shape(img)[1])
            #print(reduce, trust)
            if trust > equal_threshold:
                reduce_f = reduce
    return reduce_f

def get_max_reduce(img, axis):
    reduce_f = get_reduce(img, axis)
    padding_left = 0
    padding_right = 0

    for i in range(0, max_padding):
        img = np.insert(img, 0, 0, axis=axis)
        reduce_f_ = get_reduce(img, axis)
        if reduce_f_ > reduce_f:
            reduce_f = reduce_f_
            padding_left = i+1
            padding_right = i
            
        img = np.insert(img, 0, img.shape[axis], axis=axis)
        reduce_f_ = get_reduce(img, axis)
        if reduce_f_ > reduce_f:
            reduce_f = reduce_f_
            padding_left = i+1
            padding_right = i+1
    
    return reduce_f, padding_left, padding_right

def get_img_reduced(img, reduce_x, reduce_y, padding_left, padding_right, padding_top, padding_bot):
    img2 = img
    for i in range(0, padding_left):
        img2 = np.insert(img2, 0, 0, axis=0)
    for i in range(0, padding_right):
        img2 = np.insert(img2, 0, img.shape[0], axis=0)
    for i in range(0, padding_top):
        img2 = np.insert(img2, 0, 0, axis=1)
    for i in range(0, padding_bot):
        img2 = np.insert(img2, 0, img.shape[1], axis=1)
    
    img2_reshaped = img2.reshape(img2.shape[0]//reduce_x, -1, img2.shape[1])
    img2 = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=img2_reshaped)
    
    img2_reshaped = img2.reshape(img2.shape[0], img2.shape[1]//reduce_y, -1)
    img2 = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=2, arr=img2_reshaped)
    
    return img2

for filename in list_filenames:
    img = np.load(os.path.join(dir_np_chargrid, filename))
    gt = np.load(os.path.join(dir_np_gt, filename))
    
    #print(np.shape(img))
    #plt.imshow(img)
    #plt.show()
    #plt.clf()
    
    if np.shape(img) != (0, 0):
        reduce_x, padding_left, padding_right = get_max_reduce(img, 0)
        print("final reduce_x = ", reduce_x, "padding_l = ", padding_left, "padding_r = ", padding_right)
        
        reduce_y, padding_top, padding_bot = get_max_reduce(img, 1)
        print("final reduce_y = ", reduce_y, "padding_t = ", padding_top, "padding_b = ", padding_bot)
        
        img2 = get_img_reduced(img, reduce_x, reduce_y, padding_left, padding_right, padding_top, padding_bot)
        gt2 = get_img_reduced(gt, reduce_x, reduce_y, padding_left, padding_right, padding_top, padding_bot)
        
        #gt2 = np.repeat(gt2, reduce_x, axis=0)
        #gt2 = np.repeat(gt2, reduce_y, axis=1)
        #fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        #ax1.imshow(img)
        #ax2.imshow(gt2)
        #plt.show()
        #plt.clf()
        
        ## Save        
        np.save(os.path.join(outdir_np_chargrid_reduced, filename), img2)
        np.save(os.path.join(outdir_np_gt_reduced, filename), gt2)
    
        plt.imshow(img2)
        plt.savefig(os.path.join(outdir_png_chargrid_reduced, filename).replace("npy", "png"))
        plt.close()
        
        plt.imshow(gt2)
        plt.savefig(os.path.join(outdir_png_gt_reduced, filename).replace("npy", "png"))
        plt.close()