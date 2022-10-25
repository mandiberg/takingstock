# Program to plot 2-D Heat map
# using matplotlib.pyplot.imshow() method
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#for third test
from matplotlib.colors import BoundaryNorm

#fourth
import matplotlib.colors as colors
from matplotlib import cm
import seaborn as sns

#declariing path and image before function, but will reassign in the main loop
ROOT="/Users/michaelmandiberg/Documents/projects-active/facemap_production/"

# folder ="sourceimages"
# FOLDER ="/Users/michaelmandiberg/Dropbox/Photo Scraping/facemesh/facemeshes_commons/"
MAPDATA_FILE = "allmaps_1025.csv"
# size = (750, 750) #placeholder 


# file = "auto-service-workerowner-picture-id931914734.jpg"
# path = "sourceimages/auto-service-workerowner-picture-id931914734.jpg"
# image = cv2.imread(os.path.join(root,folder, file))  # read any image containing a face
# dfallmaps = pd.DataFrame(columns=['name', 'cropX', 'x', 'y', 'z', 'resize', 'newname', 'mouth_gap']) 

# def touch(folder):
#     if not os.path.exists(folder):
#         os.makedirs(folder)


#Do These matter?
FOLDER = os.path.join(ROOT,"5GB_testimages_output")
outputfolderRGB = os.path.join(ROOT,"face_mesh_outputsRGB")
outputfolderBW = os.path.join(ROOT,"face_mesh_outputsBW")
outputfolderMEH = os.path.join(ROOT,"face_mesh_outputsMEH")

try:
    df = pd.read_csv(os.path.join(ROOT,MAPDATA_FILE))
except:
    print('you forgot to change the filename DUH')

if df.empty:
    print('dataframe empty, probably bad path')
    sys.exit()

df['x'] = df['x'] + 180
df['y'] = df['y'] + 180

data = df[['x', 'y']].to_numpy()
npx = df[['x']].to_numpy()
npy = df[['x']].to_numpy()

print(data)
# data = np.random.random(( 12 , 12 ))

# data = df.to_numpy()

# print(data[2:2])
 
# plt.imshow( data, cmap = 'autumn' , interpolation = 'nearest' )

# data = np.random.rand( -15 , 15 )



# plt.pcolormesh(data , cmap = 'summer' )

# plt.title( "2-D Heat Map" )
# plt.show()

print(df['y'].size)

#fourth test
with sns.axes_style('whitegrid'):
	rand_normal_y = np.random.randn(df['y'].size)
	x = np.arange(0,df['y'].size, 1)
	norm = colors.CenteredNorm()
	rand_normal_y_norm = norm(rand_normal_y)
	print(rand_normal_y)
	print(rand_normal_y_norm)
	cmap = cm.coolwarm(rand_normal_y_norm)
	# sns.scatterplot(x = x, y = rand_normal_y , c=cmap,  )
	sns.scatterplot(x = df['y'], y = df['x'] , c=cmap,  )
	plt.plot(np.linspace(120,180, 200), np.repeat(100, 200), color = 'black', ls = "-")
	plt.show()

# #third test
# # a=np.random.randn(2500).reshape((50,50))

# # define the colormap
# cmap = plt.cm.jet
# # extract all colors from the .jet map
# cmaplist = [cmap(i) for i in range(cmap.N)]
# # create the new map
# cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

# # define the bins and normalize
# bounds = np.linspace(np.min(data),np.max(data),5)
# norm = BoundaryNorm(bounds, cmap.N)

# plt.imshow(data,interpolation='none',norm=norm,cmap=cmap)
# plt.colorbar()
# plt.show()
#   