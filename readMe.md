environment name: wcs3 (local mac)
                    wcs (server SEC)

- three folders have been kept in G-drive `backups_to_reduce_repo_size`; these were used for some plots in the appendix
- to run recurrent and non-recurrent end to end, run one of them first, make sure runallcities script has congestion type marker in the log filename, Thereafter wait for 10* ncities to ensure that all runs are posted; change the config to other congestion type and post the remaining runs with new marker in the run_all_cities shell script.


- Urbanscales20240628 has runs without normalising before computing SHAP values
- Urbanscales20240702 has runs with normalising before computing SHAP values
- Urbanscales20240713 has runs with normalising before computing SHAP values; and correct train data columns, so no name correction is done in the plotting scripts

After the log files are generated, these should be put in a folder and three bash scripts should be run. 
Followed by plotting scripts for recurrent and non recurrent.


To reuse recurrent data
- copy the recurrent network folder
- delete all prep_speed, train_data, and speed data object files. 
- run the Pipeline in full. This will save time since we don't need to re-do the X; only the Y.