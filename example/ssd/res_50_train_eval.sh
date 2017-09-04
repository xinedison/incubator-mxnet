gpus=4
train_epochs=1
prefix='focal_voc_res50'
focal='focal'
net='resnet50'
if [ ! -d "./model/${prefix}" ]
then
    mkdir -p ./model/${prefix} 
fi

if [ -z $focal ]
then
    python train.py --gpus ${gpus} --batch-size 24 --log ${prefix}.log --end-epoch ${train_epochs} --prefix ./model/${prefix}/${prefix} --net ${net} --pretrained ./model/ssd_resnet50_512
    python evaluate.py --gpus ${gpus} --batch-size 128 --epoch ${train_epochs} --prefix ./model/${prefix}/${prefix}_ --net ${net}
else
    python train.py --loss_version ${focal} --gpus ${gpus} --batch-size 24 --log ${prefix}.log --end-epoch ${train_epochs} --prefix ./model/${prefix}/${prefix} --net ${net} --pretrained ./model/ssd_resnet50_512
    python evaluate.py --loss_version ${focal} --gpus ${gpus} --batch-size 128 --epoch ${train_epochs} --prefix ./model/${prefix}/${prefix}_ --net ${net}
fi
