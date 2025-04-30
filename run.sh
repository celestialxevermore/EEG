gpu_id=0
n_bands_list="1 3"
target_subjects="1 2 3 4 5 6 7 8 9"

for n_bands in $n_bands_list; do
    for target_subject in $target_subjects; do
        for use_mutual in 0 1; do
            for use_align in 0 1; do

                stamp="t:${target_subject}_band:${n_bands}_mutual:${use_mutual}_align:${use_align}"

                # source_subjects = 1~9 중 target_subject만 뺀 것
                source_subjects=""
                for s in $(seq 1 9); do
                    if [ "$s" -ne "$target_subject" ]; then
                        source_subjects="${source_subjects}${s},"
                    fi
                done
                source_subjects=${source_subjects%,}  # 마지막 쉼표 제거

                echo "Running PRETRAIN for target:$target_subject n_bands:$n_bands mutual:$use_mutual align:$use_align"
                cmd="CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
                    --mode=pretrain \
                    --stamp=$stamp \
                    --net=EEGNet \
                    --source_subjects=$source_subjects \
                    --target_subject=$target_subject \
                    --n_bands=$n_bands \
                    --gpu=$gpu_id \
                    --epochs=50 \
                    --batch_size=72 \
                    --sch=exp \
                    --gamma=0.999 \
                    --seed=42 \
                    --labels=0,1,2,3"

                if [ "$use_mutual" -eq 1 ]; then
                    cmd="$cmd --use_mutual_learning"
                fi
                if [ "$use_align" -eq 1 ]; then
                    cmd="$cmd --use_multi_source_align"
                fi

                eval $cmd

                echo "Running FINETUNE for target:$target_subject n_bands:$n_bands mutual:$use_mutual align:$use_align"
                cmd="CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
                    --mode=finetune \
                    --stamp=$stamp \
                    --net=EEGNet \
                    --source_subjects=$source_subjects \
                    --target_subject=$target_subject \
                    --n_bands=$n_bands \
                    --gpu=$gpu_id \
                    --epochs=50 \
                    --batch_size=72 \
                    --sch=exp \
                    --gamma=0.999 \
                    --seed=42 \
                    --labels=0,1,2,3 \
                    --use_pretrained"

                if [ "$use_mutual" -eq 1 ]; then
                    cmd="$cmd --use_mutual_learning"
                fi
                # finetune은 무조건 multi_source_align 빼야 돼

                eval $cmd

            done
        done
    done
done
