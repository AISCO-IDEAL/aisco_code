from ssh_connector import get_output_when_ready, submit_slurm_job

job_id = submit_slurm_job()

output_data = get_output_when_ready(job_id)
print(output_data)
