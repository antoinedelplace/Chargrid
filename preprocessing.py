import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract as te
import os

te.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR/tesseract'

## Hyperparameters
dir_img = "./data/img_inputs/"
list_filenames = [os.path.join(dir_img, f) for f in os.listdir(dir_img) if os.path.isfile(os.path.join(dir_img, f))]
tesseract_conf_threshold = 10

for filename in list_filenames:
    ## Tesseract OCR
    img = plt.imread(filename, format='jpeg')
    dt = te.image_to_data(img, config="", output_type=te.Output.DATAFRAME, pandas_config=None)

    dt = dt[dt['conf']>tesseract_conf_threshold]
    dt["text"] = dt["text"].astype('str')

    ## Split character by character
    chargrid_pd = pd.DataFrame(columns = ['left', 'top', 'width', 'height', 'ord'])
    expand_x = 1
    for index, row in dt.iterrows():
        for i in range(0, len(row["text"])):
            row['width'] = (row['width']+len(row["text"])-1)//len(row["text"])*len(row["text"])
        
            chargrid_pd = chargrid_pd.append({
            'left':row['left']+row['width']*i//len(row["text"]),
            'top':row['top'],
            'width':row['width']//len(row["text"]),
            'height':row['height'],
            'ord':ord(row["text"][i])
            }, ignore_index = True)

    chargrid_pd = chargrid_pd[chargrid_pd['ord']>=33]
    chargrid_pd = chargrid_pd[chargrid_pd['ord']<=126]
    chargrid_pd['ord'] -= 32
    
    #print(chargrid_pd)
    #plot_chargrid(img, chargrid_pd)

    ## Save chargrid numpy
    chargrid_np = np.array([0]*img.shape[0]*img.shape[1]).reshape((img.shape[0], img.shape[1]))
    print(chargrid_np.shape)
    for index, row in chargrid_pd.iterrows():
        chargrid_np[row['top']:row['top']+row['height'], row['left']:row['left']+row['width']] = row['ord']
    
    chargrid_np = chargrid_np[:,~np.all(chargrid_np == 0, axis=0)]
    chargrid_np = chargrid_np[~np.all(chargrid_np == 0, axis=1),:]
    
    #plt.imshow(chargrid_np)
    #plt.show()
    #plt.clf()
    
    np.save(filename.replace("img_inputs", "np_chargrids").replace("jpg", "npy"), chargrid_np)
    
    ## Save chargrid png
    plt.imshow(chargrid_np)
    plt.savefig(filename.replace("img_inputs", "img_chargrids").replace("jpg", "png"))
    plt.close()