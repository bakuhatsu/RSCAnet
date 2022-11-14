
path = "/home/sven/data/ctw1500/"
subfolder = "train"
file = "trainval.txt"
file_path = f"{path}{subfolder}/{file}"

# Open file
img_paths_file = open(file_path, "r")

# Read data
data = img_paths_file.read()

# Convert to list with one item per line (and strip whitespace)
img_paths = data.split("\n")
# Make sure any whitespace around path is stripped
img_paths = [item.strip() for item in img_paths]

print(img_paths[0:20])

def load_paths(subfolder, file, path="/home/sven/data/ctw1500/", prepend="../../"):
    file_path = f"{path}{subfolder}/{file}"
    # Open file
    img_paths_file = open(file_path, "r")

    # Read data
    data = img_paths_file.read()

    # Convert to list with one item per line (and strip whitespace)
    img_paths = data.split("\n")
    # Make sure any whitespace around path is stripped
    img_paths = [item.strip() for item in img_paths]
    # Prepend relative path to data folder so that paths work
    img_paths = [prepend + item for item in img_paths]

    return img_paths




# train_images = load_paths(subfolder="train", file="trainval.txt")
# train_curvebbox = load_paths(subfolder="train", file="trainval_label_curve.txt")

# test_images = load_paths(subfolder="test", file="test.txt")
# test_curvebbox = load_paths(subfolder="test", file="test_label_curve.txt")

train_images = load_paths(subfolder="train", file="img_paths.txt")
train_curvebbox = load_paths(subfolder="train", file="label_curve_paths.txt")

test_images = load_paths(subfolder="test", file="img_paths.txt")
test_curvebbox = load_paths(subfolder="test", file="label_curve_paths.txt")

print(train_images[0:10])
print(train_curvebbox[0:10])
print(test_images[0:10])
print(test_curvebbox[0:10])

#print(["../../" + item for item in train_images[0:10]])


import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

img_HW = Image.open(train_images[0]).convert("RGB")
print(img_HW.size)

display(img_HW)

polygons = open(train_curvebbox[0], "r")

# Read data
polygon_data = polygons.read()

# Convert to list with one item per line 
polygon_list = polygon_data.split("\n")

print(polygon_list[0].split(","))

if (polygon_list[len(polygon_list)-1] == ""):
    polygon_list = polygon_list[:-1]

for polygon in polygon_list:
    polygon = polygon.split(",")
    # Make sure any whitespace around path is stripped
    polygon = [item.strip() for item in polygon]
    polygon = [int(num) for num in polygon]

print(eval(polygon_list[0]))
type(eval(polygon_list[0]))
print(polygon_list)

polygn = zip(*[iter(eval(polygon_list[0]))]*2)
pg = list(polygn)
print(pg)

width = img_HW.size[0]
height = img_HW.size[1]
img = Image.new("L", (width, height), 0)
ImageDraw.Draw(img).polygon(pg, outline=1, fill=1)  ## Must be a list of tuples
mask = np.array(img)

plt.imshow(mask, cmap="gray")
plt.show()

display(img_HW)

