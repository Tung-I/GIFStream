SCENE_DIR="/work/pi_rsitaram_umass_edu/tungi/datasets/neural3d"
RESULT_DIR="/work/pi_rsitaram_umass_edu/tungi/GIFStream/results_test_run"
SCENE="flame_steak"

# Script Constants
TYPE="neur3d_0"
DATA_FACTOR=2
GOP_SIZE=60
START_FRAME=0

CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE \
--disable_viewer --data_factor $DATA_FACTOR --render_traj_path ellipse \
--data_dir $SCENE_DIR/$SCENE/ --result_dir $RESULT_DIR/$SCENE/GOP_0/r0 \
--eval_steps 7000 30000 --save_steps 7000 30000 --compression_sim \
--rd_lambda 0.0005 --entropy_model_opt --rate 0 --batch_size 1 \
--GOP_size $GOP_SIZE --knn --start_frame $START_FRAME


python examples/simple_trainer_GIFStream.py neur3d_0 \
--disable_viewer --data_factor 2 --render_traj_path ellipse \
--data_dir /work/pi_rsitaram_umass_edu/tungi/datasets/neural3d/flame_steak --result_dir work/pi_rsitaram_umass_edu/tungi/GIFStream/results_test/flame_steak/GOP_0/r0 \
--eval_steps 7000 30000 --save_steps 7000 30000 --compression_sim \
--rd_lambda 0.0005 --entropy_model_opt --rate 0 --batch_size 1 \
--GOP_size 60 --knn --start_frame 0