import operator
from collections import defaultdict
import json

def list_to_dictionary(list_1):
    
    # Initialize the dictionary
    dict_1 = defaultdict(list)
    
    # Iterating through the list elements and adding it to the dictionary
    for key,value in list_1:
        dict_1[key].append(value)
    
    
    # Sorting the dictionary based on the key alphabets
    sorted_dict = sorted(dict_1.items(), key=operator.itemgetter(0))
    return sorted_dict

if __name__ == '__main__':
    
    # Initializing the list with tuple elements
    list_1 = [('John',('Physics',80)),('Daniel',('Science',90)),('John',('Science', 95)),('Mark',('Maths',100)),('Daniel',('History',75)),('Mark',('Social',95))]
    
    sorted_dict = list_to_dictionary(list_1)
    
    # Prinitng the final sorted dictionary 
    print(json.dumps(sorted_dict, indent=4))