"""
General helper utilizies for the BNN_EXP05 experiments.
"""

def count_params(model):
    count = 0
    for param_idx, (name, param) in enumerate(model.named_parameters()):
        print(f"Param index:{param_idx}\tParam_name:{name}\tParam shape:{param.shape}\tParam size:{param.nelement()}")
        count += param.nelement()
    print(f"This model has {count} parameters.")


def check_and_load_job(project, sp, verbose=False, allow_new=True):
    jobs = list(project.find_jobs(sp))
    if len(jobs) > 1:
        for job in jobs:
            print(job.sp,'\n', job.doc.saved)
        print("Error! For some reason we found two jobs that matched the same statepoint, this should not happen.")
        exit()
    elif len(jobs) == 1:
        job = jobs[0]
        if verbose:
            print(f"Found existing job!: {job.sp}")
    elif allow_new:
        print("Did not find job... initializing.")
        job = project.open_job(sp)
        job.init()
        job.doc.saved = False
    else:
        raise FileNotFoundError("Job not found!")
    return job
