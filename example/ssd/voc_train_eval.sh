gpus=4
train_epochs=240
prefix='focal_voc_vgg'
focal='focal'
if [ -z $focal ]
then
    python train.py --gpus ${gpus} --batch-size 32 --log ${prefix}.log --end-epoch ${train_epochs} --prefix ./model/${prefix}
    python evaluate.py --gpus ${gpus} --batch-size 128 --epoch ${train_epochs} --prefix ./model/${prefix}_
else
    python train.py --loss_version ${focal} --gpus ${gpus} --batch-size 32 --log ${prefix}.log --end-epoch ${train_epochs} --prefix ./model/${prefix}
    python evaluate.py --loss_version ${focal} --gpus ${gpus} --batch-size 128 --epoch ${train_epochs} --prefix ./model/${prefix}_
fi
