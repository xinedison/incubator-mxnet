# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: skip-file
import mxnet as mx
from mxnet.test_utils import *
import numpy as np
import os, gzip
import pickle as pickle
import time
try:
    import h5py
except ImportError:
    h5py = None
import sys
from common import get_data

def test_MNISTIter():
    # prepare data
    get_data.GetMNIST_ubyte()

    batch_size = 100
    train_dataiter = mx.io.MNISTIter(
            image="data/train-images-idx3-ubyte",
            label="data/train-labels-idx1-ubyte",
            data_shape=(784,),
            batch_size=batch_size, shuffle=1, flat=1, silent=0, seed=10)
    # test_loop
    nbatch = 60000 / batch_size
    batch_count = 0
    for batch in train_dataiter:
        batch_count += 1
    assert(nbatch == batch_count)
    # test_reset
    train_dataiter.reset()
    train_dataiter.iter_next()
    label_0 = train_dataiter.getlabel().asnumpy().flatten()
    train_dataiter.iter_next()
    train_dataiter.iter_next()
    train_dataiter.iter_next()
    train_dataiter.iter_next()
    train_dataiter.reset()
    train_dataiter.iter_next()
    label_1 = train_dataiter.getlabel().asnumpy().flatten()
    assert(sum(label_0 - label_1) == 0)

def test_Cifar10Rec():
    # skip-this test for saving time
    return
    get_data.GetCifar10()
    dataiter = mx.io.ImageRecordIter(
            path_imgrec="data/cifar/train.rec",
            mean_img="data/cifar/cifar10_mean.bin",
            rand_crop=False,
            and_mirror=False,
            shuffle=False,
            data_shape=(3,28,28),
            batch_size=100,
            preprocess_threads=4,
            prefetch_buffer=1)
    labelcount = [0 for i in range(10)]
    batchcount = 0
    for batch in dataiter:
        npdata = batch.data[0].asnumpy().flatten().sum()
        sys.stdout.flush()
        batchcount += 1
        nplabel = batch.label[0].asnumpy()
        for i in range(nplabel.shape[0]):
            labelcount[int(nplabel[i])] += 1
    for i in range(10):
        assert(labelcount[i] == 5000)

def test_NDArrayIter():
    data = np.ones([1000, 2, 2])
    label = np.ones([1000, 1])
    for i in range(1000):
        data[i] = i / 100
        label[i] = i / 100
    dataiter = mx.io.NDArrayIter(data, label, 128, True, last_batch_handle='pad')
    batchidx = 0
    for batch in dataiter:
        batchidx += 1
    assert(batchidx == 8)
    dataiter = mx.io.NDArrayIter(data, label, 128, False, last_batch_handle='pad')
    batchidx = 0
    labelcount = [0 for i in range(10)]
    for batch in dataiter:
        label = batch.label[0].asnumpy().flatten()
        assert((batch.data[0].asnumpy()[:,0,0] == label).all())
        for i in range(label.shape[0]):
            labelcount[int(label[i])] += 1

    for i in range(10):
        if i == 0:
            assert(labelcount[i] == 124)
        else:
            assert(labelcount[i] == 100)

def test_NDArrayIter_h5py():
    if not h5py:
        return

    data = np.ones([1000, 2, 2])
    label = np.ones([1000, 1])
    for i in range(1000):
        data[i] = i / 100
        label[i] = i / 100

    try:
        os.remove("ndarraytest.h5")
    except OSError:
        pass
    with h5py.File("ndarraytest.h5") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("label", data=label)

        dataiter = mx.io.NDArrayIter(f["data"], f["label"], 128, True, last_batch_handle='pad')
        batchidx = 0
        for batch in dataiter:
            batchidx += 1
        assert(batchidx == 8)

        dataiter = mx.io.NDArrayIter(f["data"], f["label"], 128, False, last_batch_handle='pad')
        labelcount = [0 for i in range(10)]
        for batch in dataiter:
            label = batch.label[0].asnumpy().flatten()
            assert((batch.data[0].asnumpy()[:,0,0] == label).all())
            for i in range(label.shape[0]):
                labelcount[int(label[i])] += 1

    try:
        os.remove("ndarraytest.h5")
    except OSError:
        pass

    for i in range(10):
        if i == 0:
            assert(labelcount[i] == 124)
        else:
            assert(labelcount[i] == 100)

def test_NDArrayIter_csr():
    import scipy.sparse as sp
    # creating toy data
    num_rows = rnd.randint(5, 15)
    num_cols = rnd.randint(1, 20)
    batch_size = rnd.randint(1, num_rows)
    shape = (num_rows, num_cols)
    csr, _ = rand_sparse_ndarray(shape, 'csr')
    dns = csr.asnumpy()

    # make iterators
    csr_iter = iter(mx.io.NDArrayIter(csr, csr, batch_size))
    begin = 0
    for batch in csr_iter:
        expected = np.zeros((batch_size, num_cols))
        end = begin + batch_size
        expected[:num_rows - begin] = dns[begin:end]
        if end > num_rows:
            expected[num_rows - begin:] = dns[0:end - num_rows]
        assert_almost_equal(batch.data[0].asnumpy(), expected)
        begin += batch_size

def test_LibSVMIter():
    def get_data(data_dir, data_name, url, data_origin_name):
        if not os.path.isdir(data_dir):
            os.system("mkdir " + data_dir)
        os.chdir(data_dir)
        if (not os.path.exists(data_name)):
            if sys.version_info[0] >= 3:
                from urllib.request import urlretrieve
            else:
                from urllib import urlretrieve
            zippath = os.path.join(data_dir, data_origin_name)
            urlretrieve(url, zippath)
            import bz2
            bz_file = bz2.BZ2File(data_origin_name, 'rb')
            with open(data_name, 'wb') as fout:
                try:
                    content = bz_file.read()
                    fout.write(content)
                finally:
                    bz_file.close()
        os.chdir("..")

    def check_libSVMIter_synthetic():
        cwd = os.getcwd()
        data_path = os.path.join(cwd, 'data.t')
        label_path = os.path.join(cwd, 'label.t')
        with open(data_path, 'w') as fout:
            fout.write('1.0 0:0.5 2:1.2\n')
            fout.write('-2.0\n')
            fout.write('-3.0 0:0.6 1:2.4 2:1.2\n')
            fout.write('4 2:-1.2\n')

        with open(label_path, 'w') as fout:
            fout.write('1.0\n')
            fout.write('-2.0 0:0.125\n')
            fout.write('-3.0 2:1.2\n')
            fout.write('4 1:1.0 2:-1.2\n')

        data_dir = os.path.join(cwd, 'data')
        data_train = mx.io.LibSVMIter(data_libsvm=data_path, label_libsvm=label_path,
                                      data_shape=(3, ), label_shape=(3, ), batch_size=3)

        first = mx.nd.array([[ 0.5, 0., 1.2], [ 0., 0., 0.], [ 0.6, 2.4, 1.2]])
        second = mx.nd.array([[ 0., 0., -1.2], [ 0.5, 0., 1.2], [ 0., 0., 0.]])
        i = 0
        for batch in iter(data_train):
            expected = first.asnumpy() if i == 0 else second.asnumpy()
            assert_almost_equal(data_train.getdata().asnumpy(), expected)
            i += 1

    def check_libSVMIter_news_data():
        news_metadata = {
            'name': 'news20.t',
            'origin_name': 'news20.t.bz2',
            'url': "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.t.bz2",
            'feature_dim': 62060,
            'num_classes': 20,
            'num_examples': 3993,
        }
        num_parts = 3
        batch_size = 128
        num_examples = news_metadata['num_examples']
        data_dir = os.path.join(os.getcwd(), 'data')
        get_data(data_dir, news_metadata['name'], news_metadata['url'],
                 news_metadata['origin_name'])
        path = os.path.join(data_dir, news_metadata['name'])
        data_train = mx.io.LibSVMIter(data_libsvm=path, data_shape=(news_metadata['feature_dim'],),
                                      batch_size=batch_size, num_parts=num_parts, part_index=0)
        num_batches = 0
        iterator = iter(data_train)
        for batch in iterator:
            # check the range of labels
            assert(np.sum(batch.label[0].asnumpy() > 20) == 0)
            assert(np.sum(batch.label[0].asnumpy() <= 0) == 0)
            num_batches += 1
        import math
        expected_num_batches = math.ceil(num_examples * 1.0 / batch_size / num_parts)
        assert(num_batches == int(expected_num_batches)), (num_batches, expected_num_batches)

    check_libSVMIter_synthetic()
    check_libSVMIter_news_data()

if __name__ == "__main__":
    test_NDArrayIter()
    if h5py:
        test_NDArrayIter_h5py()
    test_MNISTIter()
    test_Cifar10Rec()
    test_LibSVMIter()
    test_NDArrayIter_csr()
