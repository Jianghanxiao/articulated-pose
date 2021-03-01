cd evaluation
python compute_gt_pose.py --item='drawer' --domain='unseen' --nocs='ANCSH' --save
python compute_gt_pose.py --item='drawer' --domain='unseen' --nocs='NAOCS' --save

# run our processing over test group
python pose_multi_process.py --item='drawer' --domain='unseen'

# pose & relative joint rotation
python eval_pose_err.py --item='drawer' --domain='unseen' --nocs='ANCSH'

# 3d miou estimation
python compute_miou.py --item='eyeglasses' --domain='unseen' --nocs='ANCSH'

# performance on joint estimations 
python eval_joint_params.py --item='drawer' --domain='unseen' --nocs='ANCSH'