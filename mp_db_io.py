import os
from sys import platform


class DataIO:
    """Store key database and file IO info for use across codebase"""

    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 0.3

        # platform specific file folder (mac for michael, win for satyam)
        if platform == "darwin":
            ####### Michael's OS X Credentials ########
            self.db = {
                "host":"localhost",
                "name":"stock1",            
                "user":"root",
                "pass":"XFZ5dPJq2"
            }
            self.ROOT= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production") ## only on Mac
            self.ROOT36= "/Volumes/Test36" ## only on Mac
            self.NUMBER_OF_PROCESSES = 8
        elif platform == "win32":
            ######## Satyam's WIN Credentials #########
            self.db = {
                "host":"localhost",
                "name":"gettytest3",                 
                "user":"root",
                "pass":"SSJ2_mysql"
            }
            self.ROOT= os.path.join("D:/"+"Documents/projects-active/facemap_production") ## SD CARD
            self.NUMBER_OF_PROCESSES = 4


        self.folder_list = [
            "", #0, Empty, there is no site #0 -- starts count at 1
            os.path.join(self.ROOT,"gettyimages/newimages"), #1, Getty
            os.path.join(self.ROOT36,""),
            os.path.join(self.ROOT36,""),
            os.path.join(self.ROOT36,""),
            os.path.join(self.ROOT36,"images_pexels"), #5, Pexels
            os.path.join(self.ROOT36,""),
            os.path.join(self.ROOT36,""),
            os.path.join(self.ROOT36,""),
            os.path.join(self.ROOT36,""),
        ]
