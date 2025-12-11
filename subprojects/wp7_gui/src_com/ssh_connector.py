import time
from pathlib import Path

import paramiko

HOST = "euler.ethz.ch"
USERNAME = "druefenacht"
KEY_PATH = str(Path(__file__).parent / "id_euler")     # or remove and use password
REMOTE_SCRIPT = "/cluster/home/druefenacht/testing/testscript.py"
REMOTE_OUTFILE = "/cluster/home/druefenacht/testing/text.txt"
REMOTE_SLURM_SCRIPT = "/cluster/home/druefenacht/auto_job.sbatch"


def _ssh_connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USERNAME, key_filename=KEY_PATH)
    return client


def submit_slurm_job():
    """
    Creates a SLURM submission (non-blocking) and returns the SLURM job ID.
    """

    client = _ssh_connect()

    # Create a minimal SLURM script (no uploading!)
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=py_job
#SBATCH --output={REMOTE_OUTFILE}

python3 {REMOTE_SCRIPT}
"""

    # Write SLURM script on remote HPC
    sftp = client.open_sftp()
    with sftp.file(REMOTE_SLURM_SCRIPT, "w") as f:
        f.write(slurm_script)
    sftp.close()

    # Submit SLURM job (non-blocking)
    stdin, stdout, stderr = client.exec_command(f"sbatch {REMOTE_SLURM_SCRIPT}")  # noqa: RUF059
    output = stdout.read().decode()

    client.close()

    # Extract job ID
    job_id = output.strip().split()[-1]
    return job_id


def get_output_when_ready(job_id):
    """
    Polls SLURM until the job is DONE and then reads the output file.
    This function **blocks** until completion, but you can call it later.
    """

    client = _ssh_connect()

    while True:
        stdin, stdout, stderr = client.exec_command(f"squeue -j {job_id}")  # noqa: RUF059
        queue_output = stdout.read().decode()

        if job_id not in queue_output:
            break  # job is finished
        
        time.sleep(5)  # polling interval

    # Read the output file after job completion
    sftp = client.open_sftp()
    try:
        with sftp.file(REMOTE_OUTFILE, "r") as f:
            output_data = f.read().decode()
    except OSError:
        output_data = "(Output file not found.)"
    sftp.close()
    client.close()

    return output_data
