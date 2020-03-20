import numpy as np
import matplotlib.pyplot as plt
import os

## Hyperparameters
dir_img = "./data/np_chargrids/"
list_filenames = [os.path.join(dir_img, f) for f in os.listdir(dir_img) if os.path.isfile(os.path.join(dir_img, f))]
equal_threshold = 0.95
max_padding = 3

def get_reduce(img, axis):
    reduce_f = 1
    img2_f = np.array([])
    
    trust = 1.0
    reduce = 1
    #print(img.shape[axis])
    while reduce < img.shape[axis]/2:
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
                img2_f = img2
    return reduce_f, img2_f

def get_max_reduce(img, axis):
    reduce_f, img2_f = get_reduce(img, axis)

    for i in range(0, max_padding):
        img = np.insert(img, 0, 0, axis=axis)
        reduce_f_, img2_f_ = get_reduce(img, axis)
        if reduce_f_ > reduce_f:
            reduce_f, img2_f = reduce_f_, img2_f_
            
        img = np.insert(img, 0, img.shape[axis], axis=axis)
        reduce_f_, img2_f_ = get_reduce(img, axis)
        if reduce_f_ > reduce_f:
            reduce_f, img2_f = reduce_f_, img2_f_
    
    return reduce_f, img2_f

for filename in list_filenames:
    img = np.load(filename)
    
    #print(np.shape(img))
    #plt.imshow(img)
    #plt.show()
    #plt.clf()
    
    if np.shape(img) != (0, 0):
        reduce_x, img2_x = get_max_reduce(img, 1)
        print("final reduce_x = ", reduce_x)
        
        reduce_y, img2_y = get_max_reduce(img2_x, 0)
        print("final reduce_y = ", reduce_y)
        
        #fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        #ax1.imshow(img)
        #ax2.imshow(img2_y)
        #plt.show()
        #plt.clf()
        
        ## Save
        img2_y = img2_y[::reduce_x, ::reduce_y]
        
        np.save(filename.replace("np_chargrids", "np_chargrids_reduced"), img2_y)
    
        plt.imshow(img2_y)
        plt.savefig(filename.replace("np_chargrids", "img_chargrids_reduced").replace("npy", "png"))
        plt.close()