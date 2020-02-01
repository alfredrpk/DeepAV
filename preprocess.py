import tensorflow as tf
#import numpy as np
import copy
from lyft_dataset_sdk.lyftdataset import LyftDataset
level5data = LyftDataset(data_path='D:/LEVEL5/v1.01-train', json_path='D:/LEVEL5/v1.01-train/v1.01-train', verbose=True)

#list of scenes -> list of samples in scene -> list of every vehicle in sample

annodict = {
  "rotation": [],
  "size": [],
  "translation": [],
  "name": []
}
trash = []
raw = []
instannos = []
#my_scene = level5data.scene[0]
for my_scene in level5data.scene:
    trash=[] #empty trash bc instances are not kept across scenes
    firstsampletoken = my_scene["first_sample_token"]
    samp = level5data.get('sample', firstsampletoken)
    nextexists=True
    while (nextexists):
        anns=samp['anns']
        for ann in anns:
            annotation =  level5data.get('sample_annotation', ann)
            if annotation['instance_token'] in trash:
            else:
                instance = level5data.get('instance', annotation['instance_token'])
                anno = level5data.get('sample_annotation', instance['first_annotation_token'])
                annonum = instance['nbr_annotations']
                instannos = []
                namae = level5data.get('category', instance['category_token'])['name']
                for x in range(annonum):
                    annodict['rotation']=[]
                    annodict['size']=[]
                    annodict['translation']=[]
                    annodict['name']=[]
                    annodict['rotation'].append(anno['rotation'])
                    annodict['size'].append(anno['size'])
                    annodict['translation'].append(anno['translation'])
                    annodict['name'].append(namae)
                    instannos.append(copy.deepcopy(annodict))
                    if (x<(annonum-1)):
                        anno = level5data.get('sample_annotation', anno['next'])
                raw.append(instannos)
                trash.append(annotation['instance_token'])
        if (samp['next'] == ""):
            tf.logging.info('reached end')
            nextexists=False
        else:
            samp = level5data.get('sample', samp['next'])

import pickle
pickle_in = open("C:/DeepSDV/raw.pickle","rb")
raw = pickle.load(pickle_in)

sorryallnames = []
for scene in raw:
    sorryallnames.append(scene[0]['name'][0])

peds=[]
for nam in sorryallnames:
    if (nam=='pedestrian'):
        peds.append(nam )
    if (nam=='animal'):
        peds.append(nam)
        
vrooms=[]
for vroom in sorryallnames:
    if (vroom=='car'):
        vrooms.append(vroom)