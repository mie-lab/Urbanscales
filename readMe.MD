


# How to run: (Latest version)
## This must be updated with every major update of the code
- Run the network 
- Prep the speed data (Raw JF  data; )
- Prep the JF data (Combine the N/w and JF files )
- Prep the train data
- Train the model and generate the results 
- Use the visualise module to visualise the plots 

- speed data
- prep network
- prep speed
  - generate training 

    - Updated algo
      - Step 1: Manually empty the network folder 
      - Step 2: Run `road_network.py` with all config.delete as False
      ```python
      rn_delete_existing_pickled_objects = False
      scl_delete_existing_pickle_objects = False
      sd_delete_existing_pickle_objects = False
      ps_delete_existing_pickle_objects = False
      td_delete_existing_pickle_objects = False`

    - Step 3: Run `prep_network.py` will all config.delete same as above
    - Step 4: Run `prep_speed.py` will all config.delete same as above
    - Step 5: Run `train_data.py` will all config.delete same as above


