Working directory should be set to just outside the urbanscales folder i.e. inside
`WCS`

- Create a new conda environment with python3.8 as shown below:
```bash
conda create --name urbanscales python=3.8
```
- Activate the environment with 
```bash
conda activate urbanscales
```
- Install requirements using `pip`
```bash
pip install -r requirements.txt
```
- Run the `Pipeline.py` after installing the `requirements.txt`




- **Caution:**
The `Pipeline.py` must not be run multiple times to get several results together for multiple scenarios using multiple configs since we have system calls inside the `Pipeline.py` file. This is because the temporary files are deleted at the start of each <`city, scale`> run and hence, one of the commands like `python Pipeline.py` will be terminated after a few minutes. Instead, it is better to use all available resources for one run, get the results in, and then run the other one. Note that multiple runs will simply stop runs but will not result in erroneous/ mixing of generated results since the results and intermediate pickle files are always redirected to the correct folders based on the config values present in memory.
