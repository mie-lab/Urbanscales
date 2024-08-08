import os

def extract_context_lines(file_path, keyword, lines_before, lines_after):
    with open(file_path, 'r') as file:
        content = file.readlines()
    
    results = []
    for i, line in enumerate(content):
        if keyword in line:
            start = max(0, i - lines_before)
            end = min(len(content), i + lines_after + 1)
            context = content[start:end]
            results.append((i, context))
    return results

keyword = 'randomforestregressor__max_depth'
lines_before = 5
lines_after = 3

for city_num in range(2, 9):
    file_name = f'RECURRENTcity{city_num}.csv'
    if os.path.exists(file_name):
        contexts = extract_context_lines(file_name, keyword, lines_before, lines_after)
        for line_index, context in contexts:
            print(f"Found at line {line_index + 1}:")
            for line in context:
                print(line.strip())
        print("\n" + "-"*50 + "\n")
    else:
        print(f"File {file_name} does not exist.")



#  python determine_RF_params.py > Recurrent_RF_params.csv
#  python determine_RF_params.py > Non_Recurrent_RF_params.csv