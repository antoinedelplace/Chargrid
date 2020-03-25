import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from skimage.transform import resize, rescale

# Hardware configuration
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print("Tensorflow version: ", tf.__version__)
print("Test GPU: ", tf.test.gpu_device_name())

## Hyperparameters
dir_np_chargrid_1h = "./data/np_chargrids_1h/"
dir_np_gt_1h = "./data/np_gt_1h/"
dir_np_bbox_anchor_mask = "./data/np_bbox_anchor_mask/"
dir_np_bbox_anchor_coord = "./data/np_bbox_anchor_coord/"
list_filenames = [f for f in os.listdir(dir_np_chargrid_1h) if os.path.isfile(os.path.join(dir_np_chargrid_1h, f))]
width = 128
height = 256
input_channels = 61
base_channels = 64
learning_rate = 0.05
momentum = 0.9
weight_decay = 0.1
spatial_dropout = 0.1
nb_classes = 5
proba_classes = np.array([0.89096397, 0.01125766, 0.0504345, 0.03237164, 0.01497223]) #other, total, address, company, date
constant_weight = 1.04
nb_anchors = 4 # one per foreground class
epochs = 10
batch_size = 6
prop_test = 0.2
seed = 123456
filename_backup = "./output/model.ckpt"
pad_left_range = 0.2
pad_top_range = 0.2
pad_right_range = 0.2
pad_bot_range = 0.2

class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        ## Block z
        self.z1 = tf.keras.layers.Conv2D(input_shape=(None, height, width, input_channels), filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.z1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.z1_bn = tf.keras.layers.BatchNormalization()

        self.z2 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.z2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.z2_bn = tf.keras.layers.BatchNormalization()
        
        self.z3 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.z3_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.z3_bn = tf.keras.layers.BatchNormalization()
        
        self.z4 = tf.keras.layers.Dropout(rate=spatial_dropout)
        
        ## Block a
        self.a1 = tf.keras.layers.Conv2D(filters=2*base_channels, kernel_size=3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.a1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.a1_bn = tf.keras.layers.BatchNormalization()

        self.a2 = tf.keras.layers.Conv2D(filters=2*base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.a2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.a2_bn = tf.keras.layers.BatchNormalization()
        
        self.a3 = tf.keras.layers.Conv2D(filters=2*base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.a3_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.a3_bn = tf.keras.layers.BatchNormalization()
        
        self.a4 = tf.keras.layers.Dropout(rate=spatial_dropout)


        ## Block a_bis
        self.a_bis_filters = [4*base_channels, 8*base_channels, 8*base_channels]
        self.a_bis_stride = [2, 2, 1]
        self.a_bis_dilatation = [2, 4, 8]
        
        self.a_bis1 = []
        self.a_bis1_lrelu = []
        self.a_bis1_bn = []
        
        self.a_bis2 = []
        self.a_bis2_lrelu = []
        self.a_bis2_bn = []
        
        self.a_bis3 = []
        self.a_bis3_lrelu = []
        self.a_bis3_bn = []
        
        self.a_bis4 = []
        
        for i in range(0, len(self.a_bis_filters)):
            self.a_bis1.append(tf.keras.layers.Conv2D(filters=self.a_bis_filters[i], kernel_size=3, strides=self.a_bis_stride[i], padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.a_bis1_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.a_bis1_bn.append(tf.keras.layers.BatchNormalization())

            self.a_bis2.append(tf.keras.layers.Conv2D(filters=self.a_bis_filters[i], kernel_size=3, strides=1, dilation_rate=self.a_bis_dilatation[i], padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.a_bis2_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.a_bis2_bn.append(tf.keras.layers.BatchNormalization())
        
            self.a_bis3.append(tf.keras.layers.Conv2D(filters=self.a_bis_filters[i], kernel_size=3, strides=1, dilation_rate=self.a_bis_dilatation[i], padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.a_bis3_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.a_bis3_bn.append(tf.keras.layers.BatchNormalization())
        
            self.a_bis4.append(tf.keras.layers.Dropout(rate=spatial_dropout))
        
        
        ## Block b_ss (semantic segmentation)
        self.b_ss_filters = [4*base_channels, 2*base_channels]
        
        self.b_ss1 = []
        self.b_ss1_lrelu = []
        self.b_ss1_bn = []
        
        self.b_ss2 = []
        self.b_ss2_lrelu = []
        self.b_ss2_bn = []
        
        self.b_ss3 = []
        self.b_ss3_lrelu = []
        self.b_ss3_bn = []
        
        self.b_ss4 = []
        self.b_ss4_lrelu = []
        self.b_ss4_bn = []
        
        self.b_ss5 = []
        
        for i in range(0, len(self.b_ss_filters)):
            self.b_ss1.append(tf.keras.layers.Conv2D(filters=2*self.b_ss_filters[i], kernel_size=1, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_ss1_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_ss1_bn.append(tf.keras.layers.BatchNormalization())

            self.b_ss2.append(tf.keras.layers.Conv2DTranspose(filters=self.b_ss_filters[i], kernel_size=3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_ss2_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_ss2_bn.append(tf.keras.layers.BatchNormalization())
        
            self.b_ss3.append(tf.keras.layers.Conv2D(filters=self.b_ss_filters[i], kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_ss3_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_ss3_bn.append(tf.keras.layers.BatchNormalization())
            
            self.b_ss4.append(tf.keras.layers.Conv2D(filters=self.b_ss_filters[i], kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_ss4_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_ss4_bn.append(tf.keras.layers.BatchNormalization())
        
            self.b_ss5.append(tf.keras.layers.Dropout(rate=spatial_dropout))
        
        
        ## Block b_bbr (bounding box regression)
        self.b_bbr_filters = [4*base_channels, 2*base_channels]
        
        self.b_bbr1 = []
        self.b_bbr1_lrelu = []
        self.b_bbr1_bn = []
        
        self.b_bbr2 = []
        self.b_bbr2_lrelu = []
        self.b_bbr2_bn = []
        
        self.b_bbr3 = []
        self.b_bbr3_lrelu = []
        self.b_bbr3_bn = []
        
        self.b_bbr4 = []
        self.b_bbr4_lrelu = []
        self.b_bbr4_bn = []
        
        self.b_bbr5 = []
        
        for i in range(0, len(self.b_bbr_filters)):
            self.b_bbr1.append(tf.keras.layers.Conv2D(filters=2*self.b_bbr_filters[i], kernel_size=1, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_bbr1_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_bbr1_bn.append(tf.keras.layers.BatchNormalization())

            self.b_bbr2.append(tf.keras.layers.Conv2DTranspose(filters=self.b_bbr_filters[i], kernel_size=3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_bbr2_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_bbr2_bn.append(tf.keras.layers.BatchNormalization())
        
            self.b_bbr3.append(tf.keras.layers.Conv2D(filters=self.b_bbr_filters[i], kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_bbr3_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_bbr3_bn.append(tf.keras.layers.BatchNormalization())
            
            self.b_bbr4.append(tf.keras.layers.Conv2D(filters=self.b_bbr_filters[i], kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_bbr4_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_bbr4_bn.append(tf.keras.layers.BatchNormalization())
        
            self.b_bbr5.append(tf.keras.layers.Dropout(rate=spatial_dropout))
        
        
        ## Block c_ss
        self.c_ss1 = tf.keras.layers.Conv2D(filters=2*base_channels, kernel_size=1, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.c_ss1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.c_ss1_bn = tf.keras.layers.BatchNormalization()

        self.c_ss2 = tf.keras.layers.Conv2DTranspose(filters=base_channels, kernel_size=3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.c_ss2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.c_ss2_bn = tf.keras.layers.BatchNormalization()
        
        
        ## Block c_bbr
        self.c_bbr1 = tf.keras.layers.Conv2D(filters=2*base_channels, kernel_size=1, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.c_bbr1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.c_bbr1_bn = tf.keras.layers.BatchNormalization()

        self.c_bbr2 = tf.keras.layers.Conv2DTranspose(filters=base_channels, kernel_size=3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.c_bbr2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.c_bbr2_bn = tf.keras.layers.BatchNormalization()
        
        
        ## Block d
        self.d1 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.d1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.d1_bn = tf.keras.layers.BatchNormalization()

        self.d2 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.d2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.d2_bn = tf.keras.layers.BatchNormalization()
        
        self.d3 = tf.keras.layers.Conv2D(filters=nb_classes, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.constant_initializer(value=1e-3), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.d3_softmax = tf.keras.layers.Softmax()
        
        
        ## Block e
        self.e1 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.e1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.e1_bn = tf.keras.layers.BatchNormalization()

        self.e2 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.e2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.e2_bn = tf.keras.layers.BatchNormalization()
        
        self.e3 = tf.keras.layers.Conv2D(filters=2*nb_anchors, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.constant_initializer(value=1e-3), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.e3_softmax = tf.keras.layers.Softmax()
        
        
        ## Block f
        self.f1 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.f1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.f1_bn = tf.keras.layers.BatchNormalization()

        self.f2 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.f2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.f2_bn = tf.keras.layers.BatchNormalization()
        
        self.f3 = tf.keras.layers.Conv2D(filters=4*nb_anchors, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.constant_initializer(value=1e-3), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))


    def call(self, input):
        #print("input shape = ", input.shape)
        ## Encoder
        x = self.z1(input)
        x = self.z1_lrelu(x)
        x = self.z1_bn(x)
        x = self.z2(x)
        x = self.z2_lrelu(x)
        x = self.z2_bn(x)
        x = self.z3(x)
        x = self.z3_lrelu(x)
        x = self.z3_bn(x)
        out_z = self.z4(x)
        
        x = self.a1(out_z)
        x = self.a1_lrelu(x)
        x = self.a1_bn(x)
        x = self.a2(x)
        x = self.a2_lrelu(x)
        x = self.a2_bn(x)
        x = self.a3(x)
        x = self.a3_lrelu(x)
        x = self.a3_bn(x)
        out_a = self.a4(x)
        
        out_a_bis = []
        x = out_a
        for i in range(0, len(self.a_bis_filters)):
            x = self.a_bis1[i](x)
            x = self.a_bis1_lrelu[i](x)
            x = self.a_bis1_bn[i](x)
            x = self.a_bis2[i](x)
            x = self.a_bis2_lrelu[i](x)
            x = self.a_bis2_bn[i](x)
            x = self.a_bis3[i](x)
            x = self.a_bis3_lrelu[i](x)
            x = self.a_bis3_bn[i](x)
            x = self.a_bis4[i](x)
            out_a_bis.append(x)
        
        ## Decoder Semantic Segmentation
        concat_tab = [out_a_bis[1], out_a_bis[0]]
        for i in range(0, len(self.b_ss_filters)):
            x = tf.concat([x, concat_tab[i]], 3)
            x = self.b_ss1[i](x)
            x = self.b_ss1_lrelu[i](x)
            x = self.b_ss1_bn[i](x)
            x = self.b_ss2[i](x)
            x = self.b_ss2_lrelu[i](x)
            x = self.b_ss2_bn[i](x)
            x = self.b_ss3[i](x)
            x = self.b_ss3_lrelu[i](x)
            x = self.b_ss3_bn[i](x)
            x = self.b_ss4[i](x)
            x = self.b_ss4_lrelu[i](x)
            x = self.b_ss4_bn[i](x)
            x = self.b_ss5[i](x)
        
        x = tf.concat([x, out_a], 3)
        x = self.c_ss1(x)
        x = self.c_ss1_lrelu(x)
        x = self.c_ss1_bn(x)
        x = self.c_ss2(x)
        x = self.c_ss2_lrelu(x)
        x = self.c_ss2_bn(x)
        
        x = self.d1(x)
        x = self.d1_lrelu(x)
        x = self.d1_bn(x)
        x = self.d2(x)
        x = self.d2_lrelu(x)
        x = self.d2_bn(x)
        x = self.d3(x)
        out_d = self.d3_softmax(x)
        
        ## Decoder Bounding Box Regression
        concat_tab = [out_a_bis[1], out_a_bis[0]]
        x = out_a_bis[-1]
        for i in range(0, len(self.b_bbr_filters)):
            x = tf.concat([x, concat_tab[i]], 3)
            x = self.b_bbr1[i](x)
            x = self.b_bbr1_lrelu[i](x)
            x = self.b_bbr1_bn[i](x)
            x = self.b_bbr2[i](x)
            x = self.b_bbr2_lrelu[i](x)
            x = self.b_bbr2_bn[i](x)
            x = self.b_bbr3[i](x)
            x = self.b_bbr3_lrelu[i](x)
            x = self.b_bbr3_bn[i](x)
            x = self.b_bbr4[i](x)
            x = self.b_bbr4_lrelu[i](x)
            x = self.b_bbr4_bn[i](x)
            x = self.b_bbr5[i](x)
        
        x = tf.concat([x, out_a], 3)
        x = self.c_bbr1(x)
        x = self.c_bbr1_lrelu(x)
        x = self.c_bbr1_bn(x)
        x = self.c_bbr2(x)
        x = self.c_bbr2_lrelu(x)
        out_c_bbr = self.c_bbr2_bn(x)
        
        x = self.e1(out_c_bbr)
        x = self.e1_lrelu(x)
        x = self.e1_bn(x)
        x = self.e2(x)
        x = self.e2_lrelu(x)
        x = self.e2_bn(x)
        x = self.e3(x)
        out_e = self.e3_softmax(x)
        
        x = self.f1(out_c_bbr)
        x = self.f1_lrelu(x)
        x = self.f1_bn(x)
        x = self.f2(x)
        x = self.f2_lrelu(x)
        x = self.f2_bn(x)
        out_f = self.f3(x)
        
        #print("seg shape = ", out_d.shape)
        #print("boxmask shape = ", out_e.shape)
        #print("boxcoord shape = ", out_f.shape)
        
        return out_d, out_e, out_f

def augment_data(data, tab_rand, order, shape, coord=False):
    data_temp = resize(np.pad(data, ((tab_rand[1], tab_rand[3]), (tab_rand[0], tab_rand[2]), (0, 0)), 'constant'), shape, order=order, anti_aliasing=True)
    
    if coord:
        for i in range(0, nb_anchors):
            mask = (data_temp > 1e-6)[:, :, 4*i]
            data_temp[mask, 4*i] *= shape[1]
            data_temp[mask, 4*i] += tab_rand[0]
            data_temp[mask, 4*i] /= (tab_rand[0]+shape[1]+tab_rand[2])
            
            data_temp[mask, 4*i+2] *= shape[1]
            data_temp[mask, 4*i+2] += tab_rand[0]
            data_temp[mask, 4*i+2] /= (tab_rand[0]+shape[1]+tab_rand[2])
            
            data_temp[mask, 4*i+1] *= shape[0]
            data_temp[mask, 4*i+1] += tab_rand[1]
            data_temp[mask, 4*i+1] /= (tab_rand[1]+shape[0]+tab_rand[3])
            
            data_temp[mask, 4*i+3] *= shape[0]
            data_temp[mask, 4*i+3] += tab_rand[1]
            data_temp[mask, 4*i+3] /= (tab_rand[1]+shape[0]+tab_rand[3])

    return data_temp

def extract_batch(dataset, batch_size, pad_left_range, pad_top_range, pad_right_range, pad_bot_range):
    if batch_size > len(dataset):
        raise NameError('batch_size > len(dataset)')
    np.random.shuffle(dataset)
    
    tab_rand = np.random.rand(batch_size, 4)*[pad_left_range*width, pad_top_range*height, pad_right_range*width, pad_bot_range*height]
    tab_rand = tab_rand.astype(int)
    
    chargrid_input = []
    seg_gt = []
    anchor_mask_gt = []
    anchor_coord_gt = []
    
    for i in range(0, batch_size):
        data = np.load(os.path.join(dir_np_chargrid_1h, dataset[i]))
        chargrid_input.append(augment_data(data, tab_rand[i], order=1, shape=(height, width, input_channels)))
        
        data = np.load(os.path.join(dir_np_gt_1h, dataset[i]))
        seg_gt.append(augment_data(data, tab_rand[i], order=1, shape=(height, width, nb_classes)))
        
        data = np.load(os.path.join(dir_np_bbox_anchor_mask, dataset[i]))
        anchor_mask_gt.append(augment_data(data, tab_rand[i], order=1, shape=(height, width, 2*nb_anchors)))
        
        data = np.load(os.path.join(dir_np_bbox_anchor_coord, dataset[i]))
        anchor_coord_gt.append(augment_data(data, tab_rand[i], order=0, shape=(height, width, 4*nb_anchors), coord=True))
        
    return np.array(chargrid_input), np.array(seg_gt), np.array(anchor_mask_gt), np.array(anchor_coord_gt)

# Class weights
sample_weight_seg = np.ones((height, width, nb_classes))*1.0/np.log(constant_weight+proba_classes)
#print(sample_weight_seg)

proba_classes_boxmask = np.repeat(proba_classes[1:], 2)
proba_classes_boxmask[np.arange(1, 2*nb_anchors, 2)] = 1-proba_classes[1:]
#print(proba_classes_boxmask)

sample_weight_boxmask = np.ones((height, width, 2*nb_anchors))*1.0/np.log(constant_weight+proba_classes_boxmask)
#print(sample_weight_boxmask)

## Initialize Network
net = Network()
losses = {'output_1': tf.keras.losses.BinaryCrossentropy(), 'output_2': tf.keras.losses.BinaryCrossentropy(), 'output_3': tf.keras.losses.Huber()}
sample_weights = {'output_1': sample_weight_seg, 'output_2': proba_classes_boxmask}
net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=False), loss=losses, sample_weight=sample_weights)
net.build((None, height, width, input_channels))
#net.summary()

## Separate train and test sets
np.random.seed(seed=seed)
np.random.shuffle(list_filenames)
testset = list_filenames[:int(len(list_filenames)*prop_test)]
trainset = list_filenames[int(len(list_filenames)*prop_test):]
print("trainset: ", len(trainset), " - testset: ", len(testset))

"""
## Data augmentation
batch_chargrid, batch_seg, batch_mask, batch_coord = extract_batch(trainset, batch_size, pad_left_range, pad_top_range, pad_right_range, pad_bot_range)

index_to_test = 2
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(np.apply_along_axis(np.argmax, axis=2, arr=np.load(os.path.join(dir_np_chargrid_1h, trainset[index_to_test]))))
ax2.imshow(np.apply_along_axis(np.argmax, axis=2, arr=batch_chargrid[index_to_test]))
plt.show()
plt.clf()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(np.apply_along_axis(np.argmax, axis=2, arr=np.load(os.path.join(dir_np_gt_1h, trainset[index_to_test]))))
ax2.imshow(np.apply_along_axis(np.argmax, axis=2, arr=batch_seg[index_to_test]))
plt.show()
plt.clf()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(np.load(os.path.join(dir_np_bbox_anchor_mask, trainset[index_to_test]))[:, :, 0])
ax2.imshow(batch_mask[index_to_test][:, :, 0])
plt.show()
plt.clf()

print(batch_coord[index_to_test][(batch_coord[index_to_test] > 1e-6)[:, :, 0], 0]*width)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(np.load(os.path.join(dir_np_bbox_anchor_coord, trainset[index_to_test]))[:, :, 0])
ax2.imshow(batch_coord[index_to_test][:, :, 0])
plt.show()
plt.clf()
"""

history_time_train = []
history_loss = []
history_loss_output1 = []
history_loss_output2 = []
history_loss_output3 = []

history_time_test = []
history_val_loss = []
history_val_loss_output1 = []
history_val_loss_output2 = []
history_val_loss_output3 = []

def plot_loss(history_train, history_test, title, filename):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.plot(history_train, label="loss")
    plt.plot(history_test, label="val_loss")
    plt.plot(np.argmin(history_test), np.min(history_test), marker="o", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, format="pdf")
    plt.close()

def plot_time(history_time, title, filename):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.plot(history_time, label="time")
    plt.xlabel("Epochs")
    plt.ylabel("Time")
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, format="pdf")
    plt.close()

def train():
    tps = time.time()
    for epoch in range(epochs):
        #Train
        tps_train = time.time()
        batch_chargrid, batch_seg, batch_mask, batch_coord = extract_batch(trainset, batch_size, pad_left_range, pad_top_range, pad_right_range, pad_bot_range)
        
        history = net.fit(x=batch_chargrid, y=[batch_seg, batch_mask, batch_coord])
        history_time_train.append(time.time()-tps_train)
        history_loss.append(history.history["loss"])
        history_loss_output1.append(history.history["output_1_loss"])
        history_loss_output2.append(history.history["output_2_loss"])
        history_loss_output3.append(history.history["output_3_loss"])
        
        #Test
        tps_test = time.time()
        batch_chargrid, batch_seg, batch_mask, batch_coord = extract_batch(testset, batch_size, pad_left_range, pad_top_range, pad_right_range, pad_bot_range)
        
        history_val = net.evaluate(x=batch_chargrid, y=[batch_seg, batch_mask, batch_coord])
        history_time_test.append(time.time()-tps_test)
        history_val_loss.append(history_val[0])
        history_val_loss_output1.append(history_val[1])
        history_val_loss_output2.append(history_val[2])
        history_val_loss_output3.append(history_val[3])
        
        
    print("Execution time = ", time.time()-tps)

    net.save_weights(filename_backup)
    
    ## Plot loss
    plot_loss(history_loss, history_val_loss, "Global Loss", "./output/global_loss.pdf")
    plot_loss(history_loss_output1, history_val_loss_output1, "Output 1 Loss", "./output/output_1_loss.pdf")
    plot_loss(history_loss_output2, history_val_loss_output2, "Output 2 Loss", "./output/output_2_loss.pdf")
    plot_loss(history_loss_output3, history_val_loss_output3, "Output 3 Loss", "./output/output_3_loss.pdf")
    
    ## Plot time
    plot_time(history_time_train, "Train time", "./output/train_time.pdf")
    plot_time(history_time_test, "Test time", "./output/test_time.pdf")

train()