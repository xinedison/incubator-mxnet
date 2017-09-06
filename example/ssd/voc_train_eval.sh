gpus=6
train_epochs=1
prefix='focal_vggnet_voc_test'
focal='focal'

if [ ! -d "./model/${prefix}" ]
then
    mkdir -p ./model/${prefix} 
fi

if [ -z $focal ]
then
    python train.py --gpus ${gpus} --batch-size 32 --log ${prefix}.log --end-epoch ${train_epochs} --prefix ./model/${prefix}/${prefix}
    python evaluate.py --gpus ${gpus} --batch-size 128 --epoch ${train_epochs} --prefix ./model/${prefix}/${prefix}_
else
    python train.py --loss_version ${focal} --gpus ${gpus} --batch-size 32 --log ${prefix}.log --end-epoch ${train_epochs} --prefix ./model/${prefix}/${prefix}
    python evaluate.py --loss_version ${focal} --gpus ${gpus} --batch-size 128 --epoch ${train_epochs} --prefix ./model/${prefix}/${prefix}_
fi
