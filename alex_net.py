import chainer
import chainer.functions as F
from chainer import initializers
from chainer import Chain, Variable
import chainer.links as L

class Alex(chainer.Chain):
    insize = 227
    def __init__(self,dropout_rate = 0.0) :
        super(Alex, self).__init__(
            conv1=L.Convolution2D(None,  96, 11, stride=4),
            conv2=L.Convolution2D(None, 256,  5, pad=2),
            conv3=L.Convolution2D(None, 384,  3, pad=1),
            conv4=L.Convolution2D(None, 384,  3, pad=1),
            conv5=L.Convolution2D(None, 256,  3, pad=1),
            fc6=L.Linear(None, 4096),
            fc7=L.Linear(None, 4096),
            fc8=L.Linear(None, 1000),
        )
        self.dropout_rate = dropout_rate


    def __call__(self, x, layers):
        ret = {}
        en = layers[-1]
        h = self.conv1(x)
        if 'conv1' in layers:
            ret.update({'conv1':h})
            if en == 'conv1':
                return ret
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(h)), 3, stride=2)
        h = F.dropout(self.conv2(h),ratio = self.dropout_rate)
        if 'conv2' in layers:
            ret.update({'conv2':h})
            if en == 'conv2':
                return ret
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(h)), 3, stride=2)
        h = F.dropout(self.conv3(h),ratio = self.dropout_rate)
        if 'conv3' in layers:
            ret.update({'conv3':h})
            if en == 'conv3':
                return ret
        h = F.relu(h)
        h = F.dropout(self.conv4(h),ratio = self.dropout_rate)
        if 'conv4' in layers:
            ret.update({'conv4':h})
            if en == 'conv4':
                return ret
        h = F.relu(h)
        h = F.dropout(self.conv5(h),ratio = self.dropout_rate)
        if 'conv5' in layers:
            ret.update({'conv5':h})
            if en == 'conv5':
                return ret
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.dropout(self.fc6(h),ratio = self.dropout_rate)
        if 'fc6' in layers:
            ret.update({'fc6':h})
            if en == 'fc6':
                return ret
        h = F.relu(h)
        h = F.dropout(self.fc7(h),ratio = self.dropout_rate)
        if 'fc7' in layers:
            ret.update({'fc7':h})
            if en == 'fc7':
                return ret
        h = F.relu(h)
        h = self.fc8(h)
        if 'fc8' in layers:
            ret.update({'fc8':h})

        return ret

    def predict(self,x):
        h = self.__call__(x,layers ='fc8')['fc8']
        h = F.softmax(h)
        return h.data

    def get_layer_names(self):
        return ['conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8']

    def get_whole_units(self,layers):
        ret = {}
        if 'conv1' in layers:
            ret.update({'conv1':[96,55,55]})
        if 'conv2' in layers:
            ret.update({'conv2':[256,27,27]})
        if 'conv3' in layers:
            ret.update({'conv3':[384,13,13]})
        if 'conv4' in layers:
            ret.update({'conv4':[384,13,13]})
        if 'conv5' in layers:
            ret.update({'conv5':[256,13,13]})
        if 'fc6' in layers:
            ret.update({'fc6':[4096]})
        if 'fc7' in layers:
            ret.update({'fc7':[4096]})
        if 'fc8' in layers:
            ret.update({'fc8':[1000]})
        return ret
