#!/bin/bash

gpu_id=4
targets="5"

for target_subject in $targets; do
    for case_id in $(seq 1 10); do
        
        source_subjects=""
        for s in $(seq 1 9); do
            if [ "$s" -ne "$target_subject" ]; then
                source_subjects="${source_subjects}${s},"
            fi
        done
        source_subjects=${source_subjects%,}

        if [ $case_id -eq 1 ]; then
            n_band=1; use_mutual_learning=0; use_kl_alignment=0; use_multi_source_align=0
        elif [ $case_id -eq 2 ]; then
            n_band=1; use_mutual_learning=1; use_kl_alignment=0; use_multi_source_align=0
        elif [ $case_id -eq 3 ]; then
            n_band=1; use_mutual_learning=1; use_kl_alignment=1; use_multi_source_align=0
        elif [ $case_id -eq 4 ]; then
            n_band=5; use_mutual_learning=0; use_kl_alignment=0; use_multi_source_align=0
        elif [ $case_id -eq 5 ]; then
            n_band=5; use_mutual_learning=1; use_kl_alignment=0; use_multi_source_align=0
        elif [ $case_id -eq 6 ]; then
            n_band=5; use_mutual_learning=1; use_kl_alignment=1; use_multi_source_align=0
        elif [ $case_id -eq 7 ]; then
            n_band=1; use_mutual_learning=0; use_kl_alignment=0; use_multi_source_align=1
        elif [ $case_id -eq 8 ]; then
            n_band=5; use_mutual_learning=0; use_kl_alignment=0; use_multi_source_align=1
        elif [ $case_id -eq 9 ]; then
            n_band=1; use_mutual_learning=1; use_kl_alignment=1; use_multi_source_align=1
        elif [ $case_id -eq 10 ]; then
            n_band=5; use_mutual_learning=1; use_kl_alignment=1; use_multi_source_align=1
        fi

        save_path="./result/${target_subject}/c${case_id}"

        mkdir -p $save_path

        # echo "[Target:${target_subject} Case:${case_id}] PRETRAINING..."
        # CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=1 python main.py \
        #     --mode pretrain \
        #     --net EEGNet \
        #     --source_subjects=$source_subjects \
        #     --target_subject=$target_subject \
        #     --n_bands=$n_band \
        #     --gpu=0 \
        #     --lr 1e-4 \
        #     --epochs=50 \
        #     --batch_size=72 \
        #     --sch exp \
        #     --gamma=0.999 \
        #     --seed=42 \
        #     --labels=0,1,2,3 \
        #     $( [ "$use_mutual_learning" -eq 1 ] && echo "--use_mutual_learning" ) \
        #     $( [ "$use_kl_alignment" -eq 1 ] && echo "--use_kl_alignment" ) \
        #     $( [ "$use_multi_source_align" -eq 1 ] && echo "--use_multi_source_align" ) \
        #     --stamp c${case_id} \
        #     --save_path=$save_path

        echo "[Target:${target_subject} Case:${case_id}] FINETUNING..."
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=1 python main.py \
            --mode finetune \
            --net EEGNet \
            --source_subjects=$source_subjects \
            --target_subject=$target_subject \
            --n_bands=$n_band \
            --gpu=0 \
            --lr 1e-5 \
            --epochs=50 \
            --batch_size=72 \
            --sch exp \
            --gamma=0.999 \
            --seed=42 \
            --labels=0,1,2,3 \
            $( [ "$use_mutual_learning" -eq 1 ] && echo "--use_mutual_learning" ) \
            $( [ "$use_kl_alignment" -eq 1 ] && echo "--use_kl_alignment" ) \
            $( [ "$use_multi_source_align" -eq 1 ] && echo "--use_multi_source_align" ) \
            --use_pretrained \
            --pretrained_path=./result/${target_subject}/c${case_id}/pretrain/checkpoint/best_model.tar \
            --stamp c${case_id} \
            --save_path=$save_path

    done
done
