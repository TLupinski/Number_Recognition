import os
import sys
import time
#import psutil
import numpy as np
import skimage
from skimage.util import view_as_windows
import pickle


# def elapsed_since(start):
#     return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))


# def get_process_memory():
#     process = psutil.Process(os.getpid())
#     return process.memory_info().rss

def gaussian(x, c):
    xd = float(x)
    cd = float(c)
    s1 = 2.0
    s2 = 2.0
    c1 = cd-3.0
    c2 = cd+3.0
    return (-(xd-c2)*(xd-c2)/(2.0*s2*s2))# - math.exp(-(x-c1)*(x-c1)/(2*s1*s1)))

def lolilol():
    #im_path = "./DATASET/ORAND/Normalized_CAR-A/a_train_images/a_car_000154.png"#./DATASET/MNISTMulti1/MNISTM_Training/MNISTMult_13465.png"
    img = np.ones(shape=(256,32)).T
    img = np.ascontiguousarray(img)
    img_patched = view_as_windows(img, (32,20), step = 10)[0]
    nbp = np.shape(img_patched)[0]
    cpt = 0
    for i in range(nbp):
        for x in range(32):
            for y in range(20):
                cpt = cpt + img_patched[i][x][y]

def lolilul():
    img = np.ones(shape=(256,32)).T
    img_patched = view_as_windows(img, (32,20), step = 10)[0]
    nbp = np.shape(img_patched)[0]
    cpt = 0
    for i in range(nbp):
        for x in range(32):
            for y in range(20):
                cpt = cpt + img_patched[i][x][y]
# def profile(func):
#     def wrapper(*args, **kwargs):
#         mem_before = get_process_memory()
#         start = time.time()
#         result = func(*args, **kwargs)
#         elapsed_time = elapsed_since(start)
#         mem_after = get_process_memory()
#         print("{}: memory before: {:,}, after: {:,}, consumed: {:,}; exec time: {}".format(
#             func.__name__,
#             mem_before, mem_after, mem_after - mem_before,
#             elapsed_time))
#         return result
#     return wrapper

if True:
    x = 0
    history = {}
    epoch = []
    f = open(sys.argv[1],'rb')
    history = pickle.loads(f.read())
 #   init_epoch = len(history['loss']) - len(history['val_loss'])
    max = 0.0
    imax = -1
    p = history['val_the_output_categorical_accuracy']
    for i in range(len(p)):
        if p[i] > max:
            imax = i
            max = p[i]
    print(imax, max)
    #p = history['val_categorical_accuracy']
    f.close()
#     print(len(epoch))
#     print(len(history['loss']))
# #    print(len(history['val_loss']))
#     df=pd.DataFrame({'abs': epoch, 'train_loss': history['loss']}) #, 'val_loss': history['val_loss']}) #, 'train_acc': history['categorical_accuracy'], 'val_acc': history['val_categorical_accuracy']})
#     # multiple line plot
#     #plt.subplot(2,1,1)
# #    plt.plot( 'abs', 'val_loss', data=df, color='red', linewidth=2)
#     plt.plot( 'abs', 'train_loss', data=df, color='blue', linewidth=2)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss Function Value')
#     plt.ylim([0,60])
#     plt.legend(loc=0)
#     #plt.subplot(2,1,2)
#     #plt.plot( 'abs', 'val_acc', data=df, color='red', linewidth=4)
#     #plt.plot( 'abs', 'train_acc', data=df, color='blue', linewidth=4)
#     #plt.xlabel('Epoch')
#     #plt.ylabel('Accuracy')
#     #plt.ylim([0,1.0])
#     #plt.legend(loc=0)
#     plt.show("Figure 3")
else :
    # test = profile(lolilol)
    # test()
    # profile(lolilul)
    # print(np.shape(img_patched))
    # img_patched = img_patched[:,0]
    # nbp = np.shape(img_patched)[0]
    # plt.subplot2grid((2, nbp), (0, 0), colspan=nbp)
    # plt.imshow(img, cmap=cm.gray)
    # for i in range(nbp):
    #     plt.subplot2grid((2,nbp), (1, i))
    #     plt.imshow(img_patched[i], cmap=cm.gray)
    # plt.show()
    x = 1