#!/bin/python
import sys
import os

def print_var(var,des=''):
    print('%s : %s ' % (des,var.name) )

def infer_net_shapes(net):
    '''
    infert net shapes
    '''
    for inter in net.get_internals():
        #print('%s is %s' % (inter.name,inter.list_arguments()))
        if 'data' in inter.list_arguments():
            _,out_shape,_ = inter.infer_shape(data=(1,3,224,224))
            print('%s shape is %s'  % (inter.name, out_shape) )


if __name__ == '__main__':
    sys.path.insert(0,os.path.expanduser('~/incubator-mxnet/example/ssd/'))
    from symbol import vgg16_reduced
    vgg_net = vgg16_reduced.get_symbol(10)
    print '---------infer shape for vgg net---------------'
    infer_net_shapes(vgg_net)
    print '=======finish vgg neg=========='


    print '---------infer shape for mobile net--------------'
    from symbol import mobilenet_v1
    mobile_net = mobilenet_v1.get_symbol(10)
    infer_net_shapes(mobile_net)
    print '======finish resnet ==========='

    
    print '---------infer shape for resnet -----------------' 
    from symbol import resnet
    res_net = resnet.get_symbol(num_layers = 50,
        image_shape = '3,224,224',  # resnet require it as shape check
        num_classes = 10)
    infer_net_shapes(res_net)
    
