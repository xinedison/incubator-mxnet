gpus=2
train_epochs=240
prefix='voc_mobilenet_v1_14_7_batch_128'
focal=''
net='mobilenet_v1'
batch_size=128
if [ ! -d "./model/${prefix}" ]
then
    mkdir -p ./model/${prefix} 
fi

if [ -z $focal ]
then
    python train.py --gpus ${gpus} --batch-size ${batch_size} --log ./model/${prefix}/${prefix}.log --end-epoch ${train_epochs} --prefix ./model/${prefix}/${prefix} --net ${net} --pretrained ./model/mobilenet
    python evaluate.py --gpus ${gpus} --batch-size 128 --epoch ${train_epochs} --prefix ./model/${prefix}/${prefix}_ --net ${net}
else
    python train.py --loss_version ${focal} --gpus ${gpus} --batch-size ${batch_size} --log ./model/${prefix}/${prefix}.log --end-epoch ${train_epochs} --prefix ./model/${prefix}/${prefix} --net ${net} --pretrained ./model/mobilenet
    python evaluate.py --loss_version ${focal} --gpus ${gpus} --batch-size 128 --epoch ${train_epochs} --prefix ./model/${prefix}/${prefix}_ --net ${net}
fi
