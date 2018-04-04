import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # 画像の表示
import os             # ディレクトリの操作など
from alex_net import Alex
from chainer import serializers
from make_orientation_image import make_orientation_image

def preinput(img):
    '''
    input image ... np.ndarray  227 * 227 * 3 [RGB]
    return img 1 * 3 * 227 * 227 [BGR] sutable for Alex net
    '''
    img = img.astype(np.float32)
    img = img[:,:,[2,1,0]]
    img = img.transpose(2,0,1).astype(np.float32)
    img = np.expand_dims(img,axis=0)
    return img

def show10results(prediction):
    #load label from outsize
    categories = np.loadtxt('labels.txt', str, delimiter="\n")
    # sort to higher score represented label
    result = zip(prediction.reshape((prediction.size,)), categories)
    result = sorted(result, reverse=True)
    # show top 10 results
    print('top  percentage category')
    for i, (score, label) in enumerate(result[:10]):
        print('{:>3d} {:>6.2f}% {}'.format(i + 1, score * 100, label))

model = Alex()
serializers.load_npz(os.path.join('./data','Alex.npz'),model)

wave = make_orientation_image(np.pi/4,1,0.2)
#plt.imshow(wave)
#plt.show()

ang = np.pi/4 # 方位
index = 134321 # 活動を記録するユニットのindex
act_list = [] # ある時刻における活動地を取得
for t in range(50):
    img = make_orientation_image(ang,t) # ある時刻の方位画像を作成
    img = preinput(img)
    conv1 = F.relu(model(img,'conv1')['conv1']).data # AlexNetに入寮して活動値を得る
    conv1_vec = conv1.ravel()
    unit_act = conv1_vec[index]
    act_list.append(unit_act)

act_mean = np.mean(act_list) # 活動地の平均をとる
plt.plot(act_list)
plt.show()
print(act_mean)
# これを各方位に繰り返して横軸にすればチューニングカーブになる
