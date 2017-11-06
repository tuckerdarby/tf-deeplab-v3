import os
from defaults import *

all_images = os.listdir(DATA_DIRECTORY)
all_masks = os.listdir(MASK_DIRECTORY)

trainable = []

for filename in all_images:
    name = filename.split('.')[0]
    if (name + '.png') in all_masks:
        try:
            trainable.append(name)
        except:
            continue


with file(INDEX_FILE, 'wb') as f:
    for name in trainable:
        f.write("%s\n"%name)
