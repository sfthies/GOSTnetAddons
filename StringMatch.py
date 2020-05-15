import jellyfish as jf
import numpy as np
import pandas as pd
import string

def prepend(list, str): 
    # Using format() 
    str += '{0}'
    list = [str.format(i) for i in list] 
    return(list) 

def clean_name(name, replace_dict = {}, to_upper = True):
    """
    Function takes a single string 'name', removes punctation, replaces words that are contained in replace dict, and returns the cleaned string. If to_upper = True (default) characters are set to upper case.
    """
    new_name = []
    #remove punctuation:
    name = ''.join([str(char) for char in name if str(char) not in string.punctuation])
    #Go through every word in the old name and adjust it
    for word in name.split():
        word = word.strip()
        if word in replace_dict:
            word = replace_dict[word]
        if to_upper == True:
            word = word.upper()
        new_name.append(word)
    new_name = ' '.join(new_name)
    return new_name


def string_NN_map(names1, names2, k, dist_fun = jf.jaro_winkler):
    """
    Function that takes a pd.Series of names1 and returns for each element the closest k names in the pd.Series names2
    as measured by teh specified distance function. Default: jaro winkler
    """
    names1_ind = names1.index
    names2_ind = names2.index
    
    #calculate distances:
    dist_mat = np.array([[jf.jaro_winkler(word1, word2) for word2 in list(names2)] for word1 in list(names1)])
    
    #obtain location indicies of k nearest neighbors (taking into account that sort orders increasingly):
    k_NN_iloc = np.flip(dist_mat.argsort(axis = 1)[:,-k:], axis =1)
    
    #obtain actual indicies of k nearest neighbors:
    k_NN_loc = names2_ind.values[k_NN_iloc]
    
    #translate indicies to names:
    k_NN_names = np.apply_along_axis(lambda x: names2.loc[x], 1, k_NN_loc)
    
    #obtain distance to k NN:
    k_NN_dist = np.array([list(dist_mat[x, k_NN_iloc[x,:]]) for x in np.arange(dist_mat.shape[0])])
    
    #Store results in pandas data frame
    #Prepare column names:
    col_names = prepend(np.arange(1,k+1,1).astype(str), 'index_k_') + prepend(np.arange(1,k+1,1).astype(str), 'name_k_') + prepend(np.arange(1,k+1,1).astype(str), 'dist_k_')
    #prepare result:
    result = pd.DataFrame(np.column_stack([k_NN_loc, k_NN_names, k_NN_dist]), columns = col_names, index = names1_ind)
    
    return result