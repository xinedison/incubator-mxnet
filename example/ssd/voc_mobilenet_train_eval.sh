gpus=3
train_epochs=1
prefix='voc_mobilenet_debug'
focal=''
net='mobilenet_v1'
if [ ! -d "./model/${prefix}" ]
then
    mkdir -p ./model/${prefix} 
fi

if [ -z $focal ]
then
    python train.py --gpus ${gpus} --batch-size 24 --log ./model/${prefix}/${prefix}.log --end-epoch ${train_epochs} --prefix ./model/${prefix}/${prefix} --net ${net} --pretrained ./model/mobilenet
    python evaluate.py --gpus ${gpus} --batch-size 128 --epoch ${train_epochs} --prefix ./model/${prefix}/${prefix}_ --net ${net}
else
    python train.py --loss_version ${focal} --gpus ${gpus} --batch-size 24 --log ./model/${prefix}/${prefix}.log --end-epoch ${train_epochs} --prefix ./model/${prefix}/${prefix} --net ${net} --pretrained ./model/mobilenet
    python evaluate.py --loss_version ${focal} --gpus ${gpus} --batch-size 128 --epoch ${train_epochs} --prefix ./model/${prefix}/${prefix}_ --net ${net}
fi
