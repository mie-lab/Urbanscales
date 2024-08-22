commit id for plots in paper: ec9e079b53142c4ac3ce12d48e5509b8349fcac1
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
- The latex-generation code is present only in the non-recurrent.py (it generates for both recurrent and non recurrent)
- Copy these feature_importance_NRC_Shifting_1 files into the compare_shiifting_FI folder and run the script.
- R2 plots for fix after tile areas
- R3 with simplified graphs.
- First run the filter_files_to_retain_the_SHAP_lines.py
- Then run bash extrac_fig{1..3}.sh
- run run_for_all_cities.sh but comment out the HPT_new call inside SHAP_analysis.py and also run only the SHAP_analysis.py inside the Pipeline.py
- run grep "#tiles" RECURRENT*.csv and copy the contents in `tile_count.csv`
