py -m random_smooth.zero_col_certifier ^
--lr 0.01 ^
--layer_dims 1024 ^
--barrier 0.2 ^
--show_limit 50 ^
--type d_pool ^
--reg_noise 0.1 ^
--reg_w 0.7 ^
--perturb_all true ^
--threads_limit 1 ^
--data_part secret ^
--collision_type hard ^
--speed_up true ^
--sample_size 1232 ^
--enable_thread false ^
--output out/tmp/ ^
--models_path trajnetbaselines/lstm/Target-Model/d_pool.state