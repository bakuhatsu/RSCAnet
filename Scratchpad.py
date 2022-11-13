
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




train_images = load_paths(subfolder="train", file="trainval.txt")
train_curvebbox = load_paths(subfolder="train", file="trainval_label_curve.txt")

test_images = load_paths(subfolder="test", file="test.txt")
test_curvebbox = load_paths(subfolder="test", file="test_label_curve.txt")

print(train_images[0:10])
print(train_curvebbox[0:10])
print(test_images[0:10])
print(test_curvebbox[0:10])

print(["../../" + item for item in train_images[0:10]])