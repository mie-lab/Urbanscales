import re
"""
Recurrent_RF_params.csv contents are like this: 

Found at line 39:
dtype: float64
city, scale, tod, config.shift_tile_marker, X.shape, Y.shape : Mumbai 20 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] 3 (69, 15) (69,)
X.shape, Y.shape, "Inside the model-fitting function" : (69, 15) (69,) Inside the model-fitting function
Dict: model_trials.best_params_
Key: Value
{'randomforestregressor__max_depth': 20,
'randomforestregressor__max_features': 'sqrt',
'randomforestregressor__n_estimators': 1000}
Dict: model_trials.best_params_
Found at line 132:
dtype: float64
city, scale, tod, config.shift_tile_marker, X.shape, Y.shape : Mumbai 30 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] 3 (129, 15) (129,)
X.shape, Y.shape, "Inside the model-fitting function" : (129, 15) (129,) Inside the model-fitting function
Dict: model_trials.best_params_
Key: Value
{'randomforestregressor__max_depth': 10,
'randomforestregressor__max_features': 'sqrt',
'randomforestregressor__n_estimators': 100}
Dict: model_trials.best_params_
Found at line 225:
dtype: float64
city, scale, tod, config.shift_tile_marker, X.shape, Y.shape : Mumbai 40 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] 3 (209, 15) (209,)
X.shape, Y.shape, "Inside the model-fitting function" : (209, 15) (209,) Inside the model-fitting function
Dict: model_trials.best_params_
Key: Value
{'randomforestregressor__max_depth': 50,


"""
def extract_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read()  # Read the whole file into a single string
    eachline = content.replace("city, scale, tod, config.shift_tile_marker, X.shape, Y.shape : ","").replace("Dict: model_trials.best_params_","").replace("Key: Value","").replace("{","").replace("}","").replace("dtype: float64","").replace("X.shape, Y.shape, \"Inside the model-fitting function\" : ","").replace("[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]","").split("\n")
    eachline_copy = []
    for val in eachline:
        if "" != val and "Found at line " not in val and "--------" not in val and "Inside the model-fitting" not in val :
            eachline_copy.append(val.replace("\'","").replace("\"","").replace(",",""))

    numtilesdict = {}
    areadict = {}
    depthdict = {}
    nestimatorsdict = {}
    for i in range(len(eachline_copy)):
        if i % 4 == 0:

            # eachline_copy[4].split(" ")
            # Out[19]: ['Mumbai', '30', '', '3', '(129', '15)', '(129)']

            print (i, eachline_copy[i])
            city = eachline_copy[i].split(" ")[0]
            scale = int(eachline_copy[i].split(" ")[1])
            numtiles = int(eachline_copy[i].split(" ")[4].replace("(", ""))
            if city.lower() != "istanbul":
                area = (50/scale) ** 2
            else:
                area = (75 / scale) ** 2

            numtilesdict[city, scale] = numtiles
            areadict[city, scale] = area

        if i%4 == 1:
            # eachline_copy[1]
            # Out[23]: 'randomforestregressor__max_depth: 20'
            depth = int(eachline_copy[i].replace("randomforestregressor__max_depth: ", ""))
            depthdict[city, scale] = depth # We use the city name from the previous run, since it is sequential, so it can be done


        if i%4 == 3:
            # eachline_copy[3]
            # Out[25]: 'randomforestregressor__n_estimators: 1000'
            nestimators = int(eachline_copy[i].replace("randomforestregressor__n_estimators: ", ""))
            nestimatorsdict[city, scale] = nestimators  # We use the city name from the previous run, since it is sequential, so it can be done

    for key in numtilesdict:
        print (key, numtilesdict[key], round(areadict[key], 2), nestimatorsdict[key], depthdict[key],  nestimatorsdict[key] * depthdict[key])




# Example of using the function
data = extract_data("Recurrent_RF_params.csv")
print(data)
