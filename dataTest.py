import tensorflow as tf
#import numpy as np
import copy
from lyft_dataset_sdk.lyftdataset import LyftDataset
level5data = LyftDataset(data_path='D:/LEVEL5/v1.01-train', json_path='D:/LEVEL5/v1.01-train/v1.01-train', verbose=True)

#level5data.list_scenes()
# =============================================================================
# level5data.render_sample(firstsampletoken)
# first = level5data.get('sample', firstsampletoken)
# next = level5data.get('sample', first['next'])
# level5data.render_sample(next['token'])
# timediff = next['timestamp']-first['timestamp'] #should show time difference, not working right now, needed for displacement calc
# 
# sensor_channel = 'LIDAR_TOP'  # also try this e.g. with 'LIDAR_TOP'
# my_sample_data = level5data.get('sample_data', first['data'][sensor_channel])
# 
# my_annotation_token = first['anns'][6]
# my_annotation =  level5data.get('sample_annotation', my_annotation_token)
# level5data.render_annotation(my_annotation_token)
# 
# #LOOK AT ANNOTATION STUFF TO GET COORDINATES AND OTHER STUFF
# 
# my_ego_pose = level5data.get('ego_pose', my_sample_data['ego_pose_token'])
# 
# my_sample_data2 = level5data.get('sample_data', next['data'][sensor_channel])
# my_annotation_token2 = next['anns'][5]
# ann2img =  my_sample_data.get('sample_annotation', my_annotation_token2)
# level5data.render_annotation(ann2img)
# =============================================================================
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
                tf.logging.info('duplicate found')
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

#import pickle
#pickle_in = open("C:/DeepSDV/raw.pickle","rb")
#raw = pickle.load(pickle_in)
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
        
