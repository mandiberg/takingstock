import hashlib
import os
import shutil
import re

PATH= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production/gettyimages") 
basepath = '/testimages'

def touch(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def make_hash_folders(basepath):

    #setup alphabet list
    #long to crate full directory structure
    alphabet = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9 0'  
    alphabet2 = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9 0'  
    # alphabet = 'A B C 0 1 2'   #short alphabet for testing purposes
    # alphabet2 = 'A B C 0 1 2'   #short alphabet for testing purposes
    alphabet = alphabet.split()
    alphabet2 = alphabet2.split()

    #helper variable for determining what depth you are at
    # c_depth = alphabet

    #create depth 0
    for letter in alphabet:
        # print (letter)
        pth = os.path.join(PATH+basepath,letter)
        touch(pth)
        for letter2 in alphabet2:
            # print (letter2)

            pth = os.path.join(PATH+basepath,letter,letter+letter2)
            touch(pth)



#touch all new folders (once done once, can comment out)
make_hash_folders(basepath)

