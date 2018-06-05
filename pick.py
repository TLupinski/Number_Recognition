import pickle
import os
import numpy as np

def translate_array(array,alphabet, del_spaces=False):
    """
    Translate function from classification array to string

    #Argument
        array: Classification array with L entries where each entry consist in an array with size=alphabet with one 1 and all other 0.
        alphabet: dictionnary used to convert from integer to char

    #Output
        res: Corresponding string
    """
    res = ''
    for i in range(len(array)):
        n = np.argmax(array[i])
        if (array[i][n] != 0):
            if not(del_spaces and alphabet[n]==' '):
                res = res + (alphabet[n])
    return res

alphabet = "0123456789 "
tab = np.zeros((8,11),dtype=np.int32)
for i in range(8):
    tab[i][10]=1
res = translate_array(tab,alphabet,False)
print(res,"BRAVO")

