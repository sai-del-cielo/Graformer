#!/bin/bash

#            1          2         3         4         5
# script <dataset> <hdfs-dir> <encoder> <decoder> <suffix>




echo "[logging] CMD: ${CMD}"
echo "[logging] Start Running..."

# ted
lang_pairs="bg-en,bn-en,bs-en,cs-en,de-en,el-en,es-en,et-en,fa-en,fi-en,fr-en,hi-en,hr-en,hu-en,it-en,ja-en,kk-en,lt-en,mk-en,mr-en,nl-en,pl-en,pt-en,ro-en,ru-en,sr-en,ta-en,tr-en,uk-en,zh-en,en-bg,en-bn,en-bs,en-cs,en-de,en-el,en-es,en-et,en-fa,en-fi,en-fr,en-hi,en-hr,en-hu,en-it,en-ja,en-kk,en-lt,en-mk,en-mr,en-nl,en-pl,en-pt,en-ro,en-ru,en-sr,en-ta,en-tr,en-uk,en-zh"
lang_list="am,bg,bn,bs,cs,de,el,en,es,et,fa,fi,fr,gu,hi,hr,hu,it,iu,ja,kk,km,kn,ky,lt,lv,mk,ml,mr,nl,or,pa,pl,ps,pt,ro,ro_kd,ru,so,sr,sw,ta,te,tr,uk,zh"

lang_pairs="ja-en"
lang_list="en,ja"

#/home/steve/workspace_psu/Graformer/data-bin/parapat.tokenized.ja-en/


#$CMD ../train.py --num-workers 8 \
python ../train.py --num-workers 8 \
    ../data-bin/parapat.tokenized.ja-en/ \
    --task translation_multi_simple_epoch \
    --langs ${lang_list} --lang-pairs ${lang_pairs} \
    --sampling-method "temperature" --sampling-temperature 5 \
    --decoder-langtok --lang-tok-replacing-bos-eos \
    --arch bridge_transformer \
    --encoder-layers 12 --decoder-layers 12 \
    --no-encoder-attn-layers 0,1,2,3,4,5 \
    --encoder-learned-pos --decoder-learned-pos \
    --no-scale-embedding \
    --encoder-normalize-before --decoder-normalize-before \
    --activation-fn gelu \
    --finetune-from-model ../data-bin/parapat.tokenized.ja-en/encoder_ja_bert_small_finance.pt,../data-bin/parapat.tokenized.ja-en/decoder_en_tiny_gpt2.pt \
    --freeze-params "(.embed.)|(.layers\.(0|1|2|3|4|5)\..)|(.layers\.6\.self_attn_layer_norm.)" \
    --transfer-params "encoder.layer_norm.weight:encoder.layers.6.self_attn_layer_norm.weight,decoder.layer_norm.weight:decoder.layers.6.self_attn_layer_norm.weight,encoder.layer_norm.bias:encoder.layers.6.self_attn_layer_norm.bias,decoder.layer_norm.bias:decoder.layers.6.self_attn_layer_norm.bias,decoder.embed_tokens.weight:decoder.lm_output_projection.weight,decoder.layer_norm.weight:decoder.lm_layer_norm.weight,decoder.layer_norm.bias:decoder.lm_layer_norm.bias" \
    --lm-fusion \
    --max-update 100000000 \
    --max-tokens 4000 \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.001 --warmup-init-lr '1e-07' --min-loss-scale 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --update-freq 5  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --dropout 0.1 \
    --no-epoch-checkpoints \
    --disable-validation \
    --save-interval-updates 200 \
    --keep-interval-updates 50 \
    --save-dir local_checkpoint_path \
    --fp16 \
    --ddp-backend=no_c10d # $(echo ${extra_args[@]})
    #    --tensorboard-logdir "${tensorboard_logdir}/${signatures}" \

