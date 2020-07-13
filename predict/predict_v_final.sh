#!/bin/bash

starttime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s)



. /home/projects/11001764/wenjuan/gec_wj/beast19/transformer/common/base.sh

# . $(dirname "$0")/../common/base.sh

DEVICE=gpu_v.txt 
#$1
# models=$transformer/ensemble/sy
models=/home/projects/11001764/songyang/gec/beast19/transformer/ensemble/model_1_2_3_4/model4.pt
#/home/projects/11001764/wenjuan/gec_wj/temp/checkpoint10.pt
#/home/projects/11001764/songyang/gec/beast19/transformer/ensemble/model_1_2_3_4/model2.pt
#/home/projects/11001764/wenjuan/gec_wj/model7.pt
#/home/projects/11001764/wenjuan/gec_distillation/v_model/checkpoint/checkpoint13.pt
#/home/projects/11001764/wenjuan/gec_wj/model7.pt
#/home/projects/11001764/wenjuan/gec_distillation/v_model/checkpoint/checkpoint9.pt
#/home/projects/11001764//wenjuan/gec_wj/model7.pt

#$2

# output=${models}/outputs`echo $FLAGS | sed 's| ||g'`
#bea_dev=wilocABCN-dev
bea_test=wilocABCN-test
#wilocABCN-test
#sample
#sample
#wilocABCN-test
dataf=/home/projects/11001764/wenjuan/gec_distillation/v_f
# conll=conll14st-test-corrected
dict=$data_wq/st19-train-corrected/processed

msg=$(cat <<-END
    Script: $0
    Device: $DEVICE
END
)

########################################################################
parallel_scripts=${gec}/neural-naacl2018/training/scripts
bin=${software}/bin
bpe_model=${gec}/beast19/models/bpe_model/train.st19.bpe.model
for ext in src; do
    cat ${dataf}/${bea_test}.tok.${ext} \
        | ${bin}/parallel --no-notice --pipe -k -j 32 --block 25M bash ${parallel_scripts}/preprocess_st19.sh \
        | ${bin}/parallel --no-notice --pipe -k -j 32 --block 25M ${software}/subword-nmt/apply_bpe.py -c ${bpe_model} \
        > ${dataf}/${bea_test}.tok.processed.bpe.${ext}
done

# train need thr following, test do not
# ${beast19_scripts}/moses_scripts/clean-corpus-n.perl ${tmp}/st19-train.tok.lc.bpe src trg ${tmp}/st19-train.clean100.tok.lc.bpe 1 100
################################################################################
$transformer/common/predict_for_distillation.sh $DEVICE $bea_test $models $dict $dataf
wait


endtime=`date +'%Y-%m-%d %H:%M:%S'`
end_seconds=$(date --date="$endtime" +%s)
echo "time used:  "$((end_seconds-start_seconds))"s"
