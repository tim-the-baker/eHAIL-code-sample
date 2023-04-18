"""this file will contain the active projects used in the EXP05_BNN directory"""
import signac
base_dir = r'C:\Users\Tim\PycharmProjects\stochastic-computing\Experiments\Exp05_BNNs'
JOB_DIR_FMNIST1 = rf'{base_dir}\data\FMNIST'
JOB_DIR_EXP5 = rf'{base_dir}\data\exp05'
JOB_DIR_EXP7 = rf'{base_dir}\data\exp07'
JOB_DIR_EXP12 = rf'{base_dir}\data\exp12'
JOB_DIR_EXP13 = rf'{base_dir}\data\exp13'
JOB_DIR_EXP14 = rf'{base_dir}\data\exp14'
JOB_DIR_EXP15 = rf'{base_dir}\data\exp15'
JOB_DIR_EXP18 = rf'{base_dir}\data\exp18'
JOB_DIR_EXP19 = rf'{base_dir}\data\exp19'
JOB_DIR_EXP21 = rf'{base_dir}\data\exp21'
JOB_DIR_EXP25 = rf'{base_dir}\data\exp25'

pr_fmnist1 = signac.init_project(name="FashionMNIST", root=JOB_DIR_FMNIST1)
pr_exp5 = signac.init_project(name="exp05", root=JOB_DIR_EXP5)
pr_exp7 = signac.init_project(name="exp07", root=JOB_DIR_EXP7)
pr_exp12 = signac.init_project(name="exp12", root=JOB_DIR_EXP12)
pr_exp13 = signac.init_project(name="exp13", root=JOB_DIR_EXP13)
pr_exp14 = signac.init_project(name="exp14", root=JOB_DIR_EXP14)
pr_exp15 = signac.init_project(name="exp15", root=JOB_DIR_EXP15)
pr_exp18 = signac.init_project(name="exp18", root=JOB_DIR_EXP18)
pr_exp19 = signac.init_project(name="exp19", root=JOB_DIR_EXP19)
pr_exp21 = signac.init_project(name="exp21", root=JOB_DIR_EXP21)
pr_exp25 = signac.init_project(name="exp25", root=JOB_DIR_EXP25)
