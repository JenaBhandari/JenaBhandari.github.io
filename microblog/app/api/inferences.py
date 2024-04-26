
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#===============================================================================================================================================================================
line = "====================================================================="

#second index
age_buckets = [i for i in range(0, 110, 1)]

#third index
gender_buckets = ['Male', 'Female']

#fourth index
income_buckets = [float('-inf'), 0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000,
                  100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000, 180000, 190000,
                  200000, 210000, 220000, 230000, 240000, 250000, 260000, 270000, 280000, 290000,
                  300000, 310000, 320000, 330000, 340000, 350000, 360000, 370000, 380000, 390000, 400000, 410000,
                  420000, 430000, 440000, 450000, 460000, 470000, 480000, 490000, 500000, 510000,
                  520000, 530000, 540000, 550000, 560000, 570000, 580000, 590000, 600000, 610000,
                  620000, 630000, 640000, 650000, 660000, 670000, 680000, 690000, 700000, 710000,
                  720000, 730000, 740000, 750000, 760000, 770000, 780000, 790000, 800000, 810000,
                  820000, 830000, 840000, 850000, 860000, 870000, 880000, 890000, 900000, 910000,
                  920000, 930000, 940000, 950000, 960000, 970000, 980000, 990000, 1000000, float('inf')]

#fith index
education_buckets_labels = [
    "N/A",
    "No schooling completed",
    "Nursery school, preschool",
    "Kindergarten",
    "Grade 1",
    "Grade 2",
    "Grade 3",
    "Grade 4",
    "Grade 5",
    "Grade 6",
    "Grade 7",
    "Grade 8",
    "Grade 9",
    "Grade 10",
    "Grade 11",
    "12th grade, no diploma",
    "Regular high school diploma",
    "GED or alternative credential",
    "Some college, but less than 1 year",
    "1 or more years of college, no degree",
    "Associate's degree",
    "Bachelor's degree",
    "Master's degree",
    "Professional degree beyond a bachelor's degree",
    "Doctorate degree"
]

education_buckets_labels_short = [
    "N/A",
    "None",
    "pre",
    "Kinder",
    "g1",
    "g2",
    "g3",
    "g4",
    "g5",
    "g6",
    "g7",
    "g8",
    "Gg9",
    "g10",
    "g11",
    "g12",
    "HSD",
    "GED",
    "<1 c",
    ">1 c",
    "AA",
    "BA",
    "MS",
    "Prof",
    "PHD"
]


#===============================================================================================================================================================================
# This loads up the master array and puma mapping array
# 
# 
# 
#===============================================================================================================================================================================
def prep():
    #load up file
    print("in prep")
    global puma_mapping, income_data_array, edu_data_array, zip_to_puma, data_array
  
    puma_mapping_path = "datafiles/puma_mapping.csv"
    puma_mapping = pd.read_csv(puma_mapping_path)

    zip_to_puma_path = "datafiles/ZiptoGEOID_PUMA5_10.csv"
    zip_to_puma = pd.read_csv(zip_to_puma_path)

    npy_path = "datafiles/master_array.npy"
    data_array = np.load(npy_path)

    print("after loading files")
    # Create a list to store the results
    result_list = []

    # Start threads for processing income and education data concurrently
    income_thread = threading.Thread(target=process_data, args=(data_array, result_list, 4))
    education_thread = threading.Thread(target=process_data, args=(data_array, result_list, 3))

    income_thread.start()
    education_thread.start()

    # Wait for both threads to finish
    income_thread.join()
    education_thread.join()

    # Access the results
    income_data_array = result_list[0]
    edu_data_array = result_list[1]

    print("after threads")


#===============================================================================================================================================================================
# This returns a string of the inference made using the age 
# 
# 
# 
#===============================================================================================================================================================================
def perform_inference(zip_code, age, gender, age_range, age_bottom=None, age_top=None):
    global puma_mapping, income_data_array, edu_data_array, zip_to_puma, data_array
    # Dummy logic for inference
    print(age_range)
    print(zip_code)
    print(type(zip_code))


    puma_index = zip_to_puma_index(zip_code, puma_mapping, zip_to_puma)

    cur_income_data_array = income_data_array[puma_index]
    print("shape of income array after selecting only the puma needed", cur_income_data_array.shape)
    cur_income_data_array = np.expand_dims(cur_income_data_array, axis=0)
    print("shape of income array after adding the puma dim", cur_income_data_array.shape)

    cur_edu_data_array = edu_data_array[puma_index]
    print("shape of edu array after selecting only the puma needed", cur_edu_data_array.shape)
    cur_edu_data_array = np.expand_dims(cur_edu_data_array, axis=0)
    print("shape of edu array after adding the puma dim", cur_edu_data_array.shape)


    age_index_input = np.digitize(age, age_buckets) - 1
    gender_index_input = gender_buckets.index(gender.capitalize())

    try:
        pmf_data = cur_income_data_array[:, age_index_input, gender_index_input, :]
        pmf_data_normalized = pmf_data / np.sum(pmf_data, axis=1, keepdims=True)
        aggregated_pmf = np.mean(pmf_data_normalized, axis=0)
        target_probability = 0.85
        result_income_string = find_smallest_income_range_with_labels(aggregated_pmf, income_buckets, target_probability)
    except ZeroDivisionError:
        # Handle the division by zero error here
        print("Division by zero error occurred.")
        result_income_string = "Division by zero error occurred."
    except RuntimeWarning:
        # Handle the runtime warning here
        print("Runtime warning: invalid value encountered in divide.")
        result_income_string = "Runtime warning: invalid value encountered in divide."

    try:
        pmf_data = cur_edu_data_array[:, age_index_input, gender_index_input, :]
        pmf_data_normalized = pmf_data / np.sum(pmf_data, axis=1, keepdims=True)
        aggregated_pmf = np.mean(pmf_data_normalized, axis=0)
        result_edu_string = find_smallest_edu_range_with_labels(aggregated_pmf, education_buckets_labels,target_probability)
    except ZeroDivisionError:
        # Handle the division by zero error here
        print("Division by zero error occurred.")
        result_edu_string = "Division by zero error occurred."
    except RuntimeWarning:
        # Handle the runtime warning here
        print("Runtime warning: invalid value encountered in divide.")
        result_edu_string = "Runtime warning: invalid value encountered in divide."

    # if age_bottom is not None and age_top is not None:
    #     # Age range provided
    #     result = f"Inference for age range {age_bottom} to {age_top} and gender {gender} in zip code {zip_code}."
    # elif age is not None:
    #     # Single age provided
    #     result = f"Inference for age {age} and gender {gender} in zip code {zip_code}."
    # else:
    #     # No age provided
    #     result = "No age information provided."
    
    return result_income_string, result_edu_string

#===============================================================================================================================================================================
# This returns the index for the zipcode
# 
# 
# 
#===============================================================================================================================================================================
def zip_to_puma_index(zip_code, puma_mapping, zip_to_puma):
    state_puma_info = zip_to_puma[zip_to_puma['ZIP'] == int(zip_code)][['STATEFP', 'GEOID_PUMA5_10']].iloc[0]
    state_fips = state_puma_info['STATEFP']
    puma_numbers = state_puma_info['GEOID_PUMA5_10']

    puma_index = puma_mapping[puma_mapping['PUMA'] == int(puma_numbers)][['PUMA_INDEX']].iloc[0]
    puma_index = puma_index['PUMA_INDEX']

    return puma_index

#===============================================================================================================================================================================
# process income and education data concurrently
# 
# 
# 
#===============================================================================================================================================================================
def process_data(data_array, result_list, dimension):
    start_time = time.time()
    print("shape of array before anything:", data_array.shape)
    processed_data_array = np.sum(data_array, axis=dimension)
    print(f"shape of array after collapsing dimension {dimension}:", processed_data_array.shape)
    result_list.append(processed_data_array)
    print(f"Time taken for data operations on dimension {dimension}: %.2f seconds" % (time.time() - start_time))

#===============================================================================================================================================================================
# find the range for income
# 
# 
# 
#===============================================================================================================================================================================

def find_smallest_income_range_with_labels(pmf_data_normalized, income_buckets, target_probability=0.85):
    smallest_range = None
    smallest_range_size = float('inf')
    length = len(pmf_data_normalized)

    for start_edge in range(length):
        end_edge = start_edge
        current_probability = 0

        while (current_probability < target_probability) and (end_edge < length):
            current_probability += pmf_data_normalized[end_edge]
            end_edge += 1

        current_range_size = end_edge - start_edge

        if current_probability >= target_probability and current_range_size < smallest_range_size:
            smallest_range = (start_edge, end_edge)
            smallest_range_size = current_range_size

    if smallest_range:
        start_index, end_index = smallest_range
        print(f"The smallest valid income range with {target_probability * 100}% confidence is from index {start_index} to {end_index - 1}.")
        print(f"Income labels in this range: {income_buckets[start_index:end_index]}")
    else:
        print("No valid range for income found.")
    return (f"{income_buckets[start_index:end_index]}")

#===============================================================================================================================================================================
# find the range for education
# 
# 
# 
#===============================================================================================================================================================================

def find_smallest_edu_range_with_labels(pmf_data_normalized, education_buckets_labels, target_probability=0.85):
    sorted_indices = np.argsort(pmf_data_normalized)[::-1]
    cumulative_probability = 0
    selected_indices = []

    for index in sorted_indices:
        cumulative_probability += pmf_data_normalized[index]
        selected_indices.append(index)
        if cumulative_probability >= target_probability:
            break

    # Output education bucket labels corresponding to the selected indices
    selected_labels = [education_buckets_labels[i] for i in selected_indices]
    print(f"The education labels for {target_probability * 100}% confidence: {selected_labels}")
    return(f"{selected_labels}")
