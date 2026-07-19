The goal of this task is to adapt the training procedure of the method proposed in this repository to work with the training and test procedures available here "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework".

Specifically, what you have to do:
1. Read the paper at the following path "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda/2302.04856v1.pdf". 
2. Analyse the code repository "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
3. Understand how to train the proposed methods by using as demonstrator the human video demonstration available at the path "/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset/pick_place/human_rgb_pick_place" and the agent the ur5e pick-place tasks available here "/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset/pick_place/ur5e_pick_place". To understand the information flows and variable used for the training you can take reference starting from here "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/run_bash.py", with the dataset-class available here "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/datasets/multi_task_datasets.py". Ignore the [real] keyword and fine for train_mosaic_target_obj_detector_double_policy.sh
4. Understand how to test the trained method by using the same repository and tasks available here "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test". The starting point for this analysis is "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/test_mosaic_cond_target_obj.sh".

# Outputs
1. A Markdown file describing the paper and the repository
2. A Markdown describing the steps you performed to adapt this project to the reference training setting and dataset
3. A Markdown describing the steps you performed to adapt this project to the reference test setting
4. A Markdown describing the texts you performed to understand whether the model goas under training
5. Save debug images in the different steps of training pre-processing, inference rollouts.

# What you can do
1. Create conda environments, by following the requirements.txt and installation procedure available in the README.md
2. Create new files and folders to organize the training and test scenarios.
3. Run sbatch and srun commands
4. You can run bash commands.