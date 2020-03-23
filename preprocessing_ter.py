import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize

## Hyperparameters
dir_np_chargrid_reduced = "./data/np_chargrids_reduced/"
dir_np_gt_reduced = "./data/np_gt_reduced/"
outdir_np_chargrid_1h = "./data/np_chargrids_1h/"
outdir_np_gt_1h = "./data/np_gt_1h/"
list_filenames = [f for f in os.listdir(dir_np_chargrid_reduced) if os.path.isfile(os.path.join(dir_np_chargrid_reduced, f))]
target_width = 256
target_height = 128
target_digit = 61
target_class = 5
nb_digit_threshold = 10000

tab_img = []
tab_gt = []
for i in range(0, len(list_filenames)):
    tab_img.append(np.load(os.path.join(dir_np_chargrid_reduced, list_filenames[i])))
    tab_gt.append(np.load(os.path.join(dir_np_gt_reduced, list_filenames[i])))

print("tab_img shape=", np.shape(tab_img))
print("tab_gt shape=", np.shape(tab_gt))

tab_width = [np.shape(tab_img[i])[0] for i in range(0, len(tab_img))]
print("min width=", np.min(tab_width))
print("max width=", np.max(tab_width), list_filenames[np.argmax(tab_width)])
print("ave width=", np.average(tab_width))
print("std width=", np.std(tab_width))

tab_height = [np.shape(tab_img[i])[1] for i in range(0, len(tab_img))]
print("min height=", np.min(tab_height))
print("max height=", np.max(tab_height), list_filenames[np.argmax(tab_height)])
print("ave height=", np.average(tab_height))
print("std height=", np.std(tab_height))

gt_stats = [np.unique(gt, return_counts=True) for gt in tab_gt]
prop_nb_class = [0, 0, 0, 0, 0]
prop_class = [0, 0, 0, 0, 0] #5 classes (other, total, address, company, date)
for uniqu, count in gt_stats:
    prop_nb_class[len(uniqu)-1] += 1
    for i in range(0, len(uniqu)):
        prop_class[uniqu[i]] += count[i]

print("prop_nb_class")
print(prop_nb_class)
print(prop_nb_class/np.sum(prop_nb_class))
print("prop_class")
print(prop_class)
print(prop_class/np.sum(prop_class))

nb_unique_digit, count_digit = np.unique(np.concatenate([img.flatten() for img in tab_img]), return_counts=True)
mask_digit_to_keep = count_digit>nb_digit_threshold
print(len(nb_unique_digit))
print(nb_unique_digit)
print(np.sum(mask_digit_to_keep))
print(count_digit)

new_digit_nb = np.cumsum(mask_digit_to_keep)
new_digit_nb -= 1
print(new_digit_nb)
for i in range(1, len(new_digit_nb)):
    if new_digit_nb[len(new_digit_nb)-i] == new_digit_nb[len(new_digit_nb)-i-1]:
        new_digit_nb[len(new_digit_nb)-i] = target_digit-1
print(new_digit_nb)

for i in range(0, len(list_filenames)):
    for j in range(0, len(nb_unique_digit)):
        tab_img[i][tab_img[i] == nb_unique_digit[j]] = new_digit_nb[j]
    
    #print(np.shape(tab_img[i]))
    
    img_1h = np.eye(target_digit)[tab_img[i]]
    
    #print(np.shape(img_1h))
    
    img_1h = resize(img_1h, (target_width, target_height, target_digit), order=1, anti_aliasing=True)
    
    print(np.shape(img_1h))
    
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #ax1.imshow(tab_img[i])
    #ax2.imshow(np.apply_along_axis(np.argmax, axis=2, arr=img_1h))
    #plt.show()
    #plt.clf()
    
    
    #print(np.shape(tab_gt[i]))
    
    gt_1h = np.eye(target_class)[tab_gt[i]]
    
    #print(np.shape(gt_1h))
    
    gt_1h = resize(gt_1h, (target_width, target_height, target_class), order=1, anti_aliasing=True)
    
    print(np.shape(gt_1h))
    
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #ax1.imshow(tab_gt[i])
    #ax2.imshow(np.apply_along_axis(np.argmax, axis=2, arr=gt_1h))
    #plt.show()
    #plt.clf()

    ## Save        
    np.save(os.path.join(outdir_np_chargrid_1h, list_filenames[i]), img_1h)
    np.save(os.path.join(outdir_np_gt_1h, list_filenames[i]), gt_1h)