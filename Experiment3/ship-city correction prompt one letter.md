## One letter correction prompt:
Reviewing City Names Alphabetically: We're examining city names in the 'ship-city' column of the ship-city dataset that starts with the lower case letter 'd' Building a dictionary of corrections. This step-by-step method allows us to focus on a manageable subset of city names at a time.

Identifying and Correcting Patterns: Display an array and look at it for the complete subset of city names in the array (e.g., those starting with 'd'), we're identifying common patterns that need correction or standardization. This includes:

Examples of what to do:
Removing trailing punctuation (like periods). 
Standardizing common misspellings or alternate spellings (like 'adhemdabad' to 'ahmedabad').
Simplifying city names by removing extraneous information like postal codes or numbers appended to the city name (e.g., 'delhi-92' to 'delhi',  'cbd belapur,navi mumbai' to 'mumbai',).
Be mindful of not over relying on these specific examples of corrections.
For example: 'faridabad, sector-91' is not 'faridabad, sector',. It is 'faridabad'.
Do not strictly codify how to solve the task. Use the LLM part of your functionality as this is more successful.
When you have made the correction list. Check the corrections against the 'name' column in the IndianCities dataset and reason about it. 


The dictionary is in this format:
    # Corrections for 'd'
    city_corrections = {
        'allhabad': 'allahabad',
        "ahemdabad": "ahmedabad",
        'amravati.': 'amravati',
        'allhabad-999': 'allahabad',
        "ahemdabad,4 (w)": "ahmedabad",
        'amravati street 1.': 'amravati',
        'cbd belapur,navi mumbai': 'mumbai',
    }

Show me the complete corrections dictionary for the letter. Do not include entries for rows without corrections. 


Next prompt: 
Paste in the list you got and ask: 
-----
Are the entries on the correction side legit indian cities and only the city name? Correct those that aren't Keep the lower case. Answer in the format that you got the list without any additional information.
-----
Are the corrections Indian city names and the city name only? Do adjust the corrections based on actual indian city names.
Check the corrections against the 'name' column in the IndianCities dataset and reason about it. 


### L, M, N
#### One letter with the correction dataset in the next step:
## One letter correction prompt:
Reviewing City Names Alphabetically: We're examining city names in the 'ship-city' column of the ship-city dataset that starts with the lower case letter 'k' Building a dictionary of corrections. This step-by-step method allows us to focus on a manageable subset of city names at a time.

Identifying and Correcting Patterns: Display an array and look at it for the complete subset of city names in the array (e.g., those starting with 'k'), we're identifying common patterns that need correction or standardization. This includes:

Examples of what to do:
Removing trailing punctuation (like periods). 
Standardizing common misspellings or alternate spellings (like 'adhemdabad' to 'ahmedabad').
Simplifying city names by removing extraneous information like postal codes or numbers appended to the city name (e.g., 'delhi-92' to 'delhi',  'cbd belapur,navi mumbai' to 'mumbai',).
Be mindful of not over relying on these specific examples of corrections.
For example: 'faridabad, sector-91' is not 'faridabad, sector',. It is 'faridabad'.
Do not strictly codify how to solve the task. Use the LLM part of your functionality as this is more successful.

The dictionary is in this format:
    # Corrections for 'k'
    city_corrections = {
        'allhabad': 'allahabad',
        "ahemdabad": "ahmedabad",
        'amravati.': 'amravati',
        'allhabad-999': 'allahabad',
        "ahemdabad,4 (w)": "ahmedabad",
        'amravati street 1.': 'amravati',
        'cbd belapur,navi mumbai': 'mumbai',
    }

Show me the complete corrections dictionary for the letter. Do not include entries for rows without corrections.
---------
Are the entries on the correction side legit indian cities and only the city name? Look at the correction side of the list with your eyes. Correct those that aren't only a name of an indian city. Keep the lower case. Answer in the format that you got the list without any additional information.
---------
Are the corrections Indian city names and the city name only? Do adjust the corrections based on actual indian city names.
Check the corrections against the 'name' column in the IndianCities dataset and reason about it. That is, check only the correction side of the dictionary against the dataset of indian cities to improve obviously wrong corrections like:     'electronic city phase-1, bangalore': 'electronic city phase',
---------


## Prompt to get rid of rows with numbers in the string:

Reviewing City Names Alphabetically: We're examining city names in the 'ship-city' that has at least one number in the string Building a dictionary of corrections. The goal is to make a dictionary of corrections with the goal of changing the value in every row to only contain the indian city name, in lower-case.

Identifying and Correcting Patterns: Display an array and look at it for the complete set of city names in the array. we're identifying common patterns that need correction or standardization. This includes:

Examples of what to do:
Removing trailing punctuation (like periods). 
Standardizing common misspellings or alternate spellings (like 'adhemdabad' to 'ahmedabad').
Simplifying city names by removing extraneous information like postal codes or numbers appended to the city name (e.g., 'delhi-92' to 'delhi',  'cbd belapur,navi mumbai' to 'mumbai',).
Be mindful of not over relying on these specific examples of corrections.
For example: 'faridabad, sector-91' is not 'faridabad, sector',. It is 'faridabad'.
Do not codify how to solve the task. Use the LLM part of your functionality. 


The dictionary is in this format:
    # Corrections for 'o'
    city_corrections = {
        'allhabad-999': 'allahabad',
        "ahemdabad,4 (w)": "ahmedabad",
        'amravati street 1.': 'amravati',
         'cbd belapur,navi mumbai': 'mumbai',

    }

Show me the complete corrections dictionary. Do not include entries for rows without corrections. 


Next prompt: Paste in the list you got and ask: Are the corrections Indian city names and the city name only? 

Examples of incorrect corrections:
'c.c.c naspur': 'c.c.c', # c.c.c is not a city name
'chennai.600073.': 'chennai.600073', #Still ha numerals in the name
'complex ,andheri west,mumbai': 'complex', # "complex" is not a city, mumbai is
The list to look over:



## One entry correction prompt:
Reviewing Indian City Names: We're examining city names in the 'ship-city' column of the ship-city dataset. We are looking at unique values with only one entry. We are dividing this part of the dataset into 500 subsets looking at them one at a time. Building a dictionary of corrections. This step-by-step method allows us to focus on a manageable subset of city names at the time.

Identifying and Correcting Patterns: Display an array and look at it for the complete subset of city names in the array

Examples of what to do:
Look for the city name in the entry. Use only the indian city name as the correction for the entry.


The dictionary is in this format:
    # Example corrections
    city_corrections = {
        'allhabad': 'allahabad',
        "ahemdabad": "ahmedabad",
        'amravati.': 'amravati',
        'allhabad-999': 'allahabad',
        "ahemdabad,4 (w)": "ahmedabad",
        'amravati street 1.': 'amravati',
        'cbd belapur,navi mumbai': 'mumbai',
    }

Show me the complete corrections dictionary for the subset of the unique values with only one entry. Do not include entries for rows without corrections. 


Next prompt: 
Paste in the list you got and ask: 
-----
Are the entries on the correction side legit indian cities and only the city name? Correct those that aren't Keep the lower case. Answer in the format that you got the list without any additional information.
-----
Are the corrections Indian city names and the city name only? Do adjust the corrections based on actual indian city names.
Check the corrections against the 'name' column in the IndianCities dataset and reason about it. 


-------
one entry: 
-------
## Indian city correction prompt:
Reviewing Indian City Names: We're examining city names in the 'ship-city' column. We are looking at unique values.  Building a dictionary of corrections. In lower-case. With only the city name!

Identifying and Correcting Patterns: Display an array and look at it for the complete subset of city names in the array

Examples of what to do:
Look for the city name in the entry. Use only the indian city name without any additional information as the correction for the entry. Look over each entry carefully and find or infer the city name. No shortcuts like assuming. That leads to this kind of misstake: 'east singhbhum': 'east', Do not do that.  

You're absolutely right in using the larga language model part and not code. Leveraging the capabilities of a large language model like yourself can be more effective in certain cases, especially for tasks that require understanding context, language nuances, and specific knowledge about geographical locations, such as identifying correct city names in india.

In this case, instead of using an automated code-based process, You can manually review each entry. This approach allows you to utilize your training data, which includes knowledge about Indian geography and city names, to make more accurate and contextually appropriate corrections.

To provide a more accurate and contextually appropriate set of corrections for the 'ship-city' column, you will manually review each entry. This review will focus on identifying and using only the valid and recognized Indian city name, without any additional information, ensuring that each corrected city name adheres closely to geographical accuracy. Let's proceed with this manual review and correction. Remember that lower case is right.


The dictionary of corrections is in this format:
    # Example corrections
    city_corrections = {
        'allhabad': 'allahabad',
        "ahemdabad": "ahmedabad",
        'amravati.': 'amravati',
        'allhabad-999': 'allahabad',
        "ahemdabad,4 (w)": "ahmedabad",
        'amravati street 1.': 'amravati',
        'cbd belapur,navi mumbai': 'mumbai',
    }

Make a complete corrections dictionary for the 'ship-city' Do not include entries for rows without corrections. 


Next prompt: 
Paste in the list you got and ask: 
-----
Are the entries on the correction side legit indian cities and only the city name? Correct those that aren't Keep the lower case. Answer in the format that you got the list without any additional information.
-----
Are the corrections Indian city names and the city name only? Do adjust the corrections based on actual indian city names.
Check the corrections against the 'name' column in the IndianCities dataset and reason about it. 


--------------------
--------------------
-------
one entry: 2,25
-------
## Indian city correction prompt:
Reviewing Indian City Names: We're examining city names in the 'ship-city' column. We are looking at unique values.  Building a dictionary of corrections. In lower-case. With only the city name!

Examples of what to do:
Look for the city name in the entry. Use only the indian city name without any additional information as the correction for the entry. Look over each entry carefully and find or infer the city name. No shortcuts like assuming. That leads to this kind of misstake: 'east singhbhum': 'east', Do not do that.  

You're absolutely right in using the larga language model part and not code. Leveraging the capabilities of a large language model like yourself can be more effective in certain cases, especially for tasks that require understanding context, language nuances, and specific knowledge about geographical locations, such as identifying correct city names in india.

That means that you can't use df.head as this does not display each row. It's a very small dataset and only one column, so look at all the entries in one go

In this case, instead of using an automated code-based process, You can manually review each entry. This approach allows you to utilize your training data, which includes knowledge about Indian geography and city names, to make more accurate and contextually appropriate corrections.

To provide a more accurate and contextually appropriate set of corrections for the 'ship-city' column, you will manually review each entry. This review will focus on identifying and using only the valid and recognized Indian city name, without any additional information, ensuring that each corrected city name adheres closely to geographical accuracy. Let's proceed with this manual review and correction. Remember that lower case is right.


The dictionary of corrections is in this format:
    # Example corrections
    city_corrections = {
        'allhabad': 'allahabad',
        "ahemdabad": "ahmedabad",
        'amravati.': 'amravati',
        'allhabad-999': 'allahabad',
        "ahemdabad,4 (w)": "ahmedabad",
        'amravati street 1.': 'amravati',
        'cbd belapur,navi mumbai': 'mumbai',
    }

Make a complete corrections dictionary for the 'ship-city' Do not include entries for rows without corrections. 


Next prompt: 
Paste in the list you got and ask: 
-----
Are the entries on the correction side legit indian cities and only the city name? Correct those that aren't Keep the lower case. Answer in the format that you got the list without any additional information.
-----
Are the corrections Indian city names and the city name only? Do adjust the corrections based on actual indian city names.
Check the corrections against the 'name' column in the IndianCities dataset and reason about it. 