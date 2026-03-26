# Set the directory containing the scenes
SCENE_DIR="/work/pi_rsitaram_umass_edu/tungi/datasets/neural3d"
# Set the directory to store results
RESULT_DIR="results_ori"
# Set the rendering trajectory path
RENDER_TRAJ_PATH="ellipse"
# List of scenes to process
SCENE_LIST="flame_salmon_1"
# List of entropy lambda values (rate-distortion tradeoff parameter)
ENTROPY_LAMBDA_LIST=(0.0005 0.001 0.002 0.004)
# Data factor for training
DATA_FACTOR=2
# Number of frames per GOP (Group of Pictures)
GOP=60
# The index of the first frame to process
FIRST_FRAME=0
# Total number of frames to process
TOTAL_FRAME=120

# Loop over each scene in the scene list
for SCENE in $SCENE_LIST;
do
    # Set TYPE based on the scene name
    if [ "$SCENE" = "coffee_martini" ]; then
        TYPE=neur3d_2
    elif [ "$SCENE" = "flame_salmon_1" ]; then
        TYPE=neur3d_1
    else
        TYPE=neur3d_0
    fi

    # Loop over each entropy lambda (rate)
    for ((RATE=0; RATE<${#ENTROPY_LAMBDA_LIST[@]}; RATE++));
    do
        # Loop over each GOP segment
        for ((GOP_ID=0; GOP_ID < $(((TOTAL_FRAME + GOP - 1)/GOP)) ; GOP_ID++));
        do
            echo "Running $SCENE"
            # Set experiment name and output directory
            EXP_NAME=$RESULT_DIR/${SCENE}/GOP_$GOP_ID/r$RATE
            # Calculate the starting frame for this GOP
            GOP_START_FRAME=$((FIRST_FRAME + GOP_ID * GOP ))
            # Calculate the maximum number of frames for this GOP
            MAX_GOP=$((TOTAL_FRAME + FIRST_FRAME - GOP_START_FRAME))
            if ((GOP_ID == 0)); then
                # If this is the first GOP, train from scratch
                CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE --disable_viewer --data_factor $DATA_FACTOR \
                    --render_traj_path $RENDER_TRAJ_PATH --data_dir $SCENE_DIR/$SCENE/ --result_dir $EXP_NAME \
                    --eval_steps 7000 30000 --save_steps 7000 30000 \
                    --compression_sim --rd_lambda ${ENTROPY_LAMBDA_LIST[RATE]} --entropy_model_opt --rate $RATE \
                    --batch_size 1 --GOP_size $(( MAX_GOP < GOP ? MAX_GOP : GOP)) --knn --start_frame $GOP_START_FRAME

                # Run evaluation and rendering after training
                CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE --disable_viewer --data_factor $DATA_FACTOR \
                    --render_traj_path $RENDER_TRAJ_PATH --data_dir $SCENE_DIR/$SCENE/ --result_dir $EXP_NAME \
                    --ckpt $EXP_NAME/ckpts/ckpt_29999_rank0.pt \
                    --compression end2end  --rate $RATE \
                    --GOP_size $(( MAX_GOP < GOP ? MAX_GOP : GOP)) --knn --start_frame $GOP_START_FRAME 
            else
                # For subsequent GOPs, continue training from first checkpoint
                CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE --disable_viewer --data_factor $DATA_FACTOR \
                    --render_traj_path $RENDER_TRAJ_PATH --data_dir $SCENE_DIR/$SCENE/ --result_dir $EXP_NAME \
                    --eval_steps 7000 30000 --save_steps 7000 30000 \
                    --compression_sim --rd_lambda ${ENTROPY_LAMBDA_LIST[RATE]} --entropy_model_opt --rate $RATE \
                    --batch_size 1 --GOP_size $(( MAX_GOP < GOP ? MAX_GOP : GOP)) --knn --start_frame $GOP_START_FRAME \
                    --ckpt $RESULT_DIR/${SCENE}/GOP_0/r$RATE/ckpts/ckpt_6999_rank0.pt --continue_training 

                # Run evaluation and rendering after training
                CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE --disable_viewer --data_factor $DATA_FACTOR \
                    --render_traj_path $RENDER_TRAJ_PATH --data_dir $SCENE_DIR/$SCENE/ --result_dir $EXP_NAME \
                    --ckpt $EXP_NAME/ckpts/ckpt_29999_rank0.pt \
                    --compression end2end  --rate $RATE \
                    --GOP_size $(( MAX_GOP < GOP ? MAX_GOP : GOP)) --knn --start_frame $GOP_START_FRAME 
            fi
        done
    done
done

