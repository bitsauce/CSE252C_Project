
# coding: utf-8

# In[ ]:

get_ipython().magic('matplotlib inline')
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
pylab.rcParams["figure.figsize"] = (8.0, 10.0)


# In[ ]:

data_dir  = "../data/coco"
data_type = "train2017"
ann_file  = "{}/annotations/instances_{}.json".format(data_dir, data_type)

# Initialize COCO api for instance annotations
coco = COCO(ann_file)


# In[ ]:

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat["name"] for cat in cats]
print("COCO categories: \n{}\n".format(" ".join(nms)))

nms = set([cat["supercategory"] for cat in cats])
print("COCO supercategories: \n{}".format(" ".join(nms)))


# In[ ]:

imgIds = set()
catIds = coco.getCatIds()[:10]
for catId in catIds:
    for imgId in coco.getImgIds(catIds=[catId])[:700]:
        imgIds.add(imgId)
imgIds = list(imgIds)


# In[ ]:

# load and display image
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
I = io.imread(os.path.join(data_dir, data_type, img["file_name"]))

# load and display instance annotations
plt.imshow(I); plt.axis("off")
annIds = coco.getAnnIds(imgIds=img["id"], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns, show_bbox=True)

