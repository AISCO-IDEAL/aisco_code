Run this in your Terminal to get running access on Euler for the slurm script:
```
chmod +x ~/AISCO-code/subprojects/wp4_aeroshell_optimizer/wp4.sh
```
2. Adjust your Output-, Error Logs, your username,  main.py to you personal username 
3. Run in your Termina
```
sbatch ~/AISCO-code/subprojects/wp4_aeroshell_optimizer/wp4.sh
```

4. With the following command, you can automatically track the status of your Euler HPC:
```
watch -n 5 squeue -u $USER
