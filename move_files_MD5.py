import hashlib
import os
import shutil
import re

testname = "woman-in-a-music-concert-picture-id505111652.jpg"
PATH= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production/gettyimages") 
# folder ="5GB_testimages"


def get_hash_folders(filename):
    m = hashlib.md5()
    m.update(filename.encode('utf-8'))
    d = m.hexdigest()
    # csvWriter1.writerow(["https://upload.wikimedia.org/wikipedia/commons/"+d[0]+'/'+d[0:2]+'/'+filename])
    return d[0], d[0:2]
# print(get_hash_folders(testname))


def get_dir_files(folder):
    # counter = 1

    # directory = folder
    directory = os.path.join(folder)
    # print(directory)

    meta_file_list = []
    try:
        os.chdir(directory)
        # print(len(os.listdir(directory)))
        for filename in os.listdir(directory):
        # for item in os.listdir(root):
            # print (counter)

            if not filename.startswith('.') and os.path.isfile(os.path.join(directory, filename)):
                meta_file_list.append(filename)

    except Exception as e:
        raise e
    return meta_file_list


def touch(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def make_hash_folders():
    basepath = '/newimages'

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
make_hash_folders()

#loop through all existing folders
basepath = '/images'

#setup alphabet list
#long to crate full directory structure
alphabet = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9 0'  
# alphabet = '0'  
alphabet2 = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9 0'  
# alphabet = 'A B C 0 1 2'   #short alphabet for testing purposes
# alphabet2 = 'A B C 0 1 2'   #short alphabet for testing purposes
alphabet = alphabet.split()
alphabet2 = alphabet2.split()

#helper variable for determining what depth you are at
# c_depth = alphabet

counter = 0

#create depth 0
for letter in alphabet:
    # print (letter)
    # pth = os.path.join(PATH+basepath,letter)
    # print(pth)
    for letter2 in alphabet2:
        # print (letter2)

        pth = os.path.join(PATH+basepath,letter,letter+letter2)
        meta_file_list = get_dir_files(pth)
        for file in meta_file_list:
            #removes jpg in case you did it too many times
            # os.rename(file, file.replace(".jpg",""))
            newfile = file+".jpg"
            #this keeps adding jpg, so only do once!
            os.rename(file, newfile)
            if  re.search(r"\.jpg\.jpg", file):
                print("\n\n\n\n\n\n\n\n\nDOUBLE TROUBLE DOUBLE TROUBLE DOUBLE TROUBLE DOUBLE TROUBLE DOUBLE TROUBLE \n\n\n\n\n\n\n\n\n") 
            print(file)
            a,b = get_hash_folders(newfile)
            currentpathfile = os.path.join(pth,newfile)
            newpathfile = os.path.join(PATH,"newimages",a,b,newfile)

            #if you dont move the files, it will repeate the renaming on about 10% of the files
            shutil.move(currentpathfile, newpathfile)

            print("moved from: ",currentpathfile)
            print("moved to: ",newpathfile)
            print(counter)
            counter = counter+1


#when in each subfolder
#get_dir_files
#for each file in meta_file_list
#rename with jpg
#get_hash_folders(filename)
#move filename to new filename with hashfolders

#TEST ON A SMALL SUBSET FIRST!!!!