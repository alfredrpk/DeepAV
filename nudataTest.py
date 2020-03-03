import numpy as np
import copy
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='D:/NuScenes', verbose=True)
#Train dataset is used because test dataset doesn't have annotations.
import pickle
import random

raw = []
count=1
sampcount=0
instances = []
loglist = []
scenes = nusc.scene
randscenes = random.sample(scenes, len(scenes))
#randscenes = randscenes[:int(len(randscenes)/4)]
#pickle.dump( data, open( "randscenes.cpkl", "wb" ) )
trainscenes = pickle.load( open( "trainscenes.cpkl", "rb" ) )
testscenes=[]
for x in range(len(randscenes)):
    if randscenes[x] not in trainscenes:
        testscenes.append(randscenes[x])

testscenes = testscenes[:int(len(testscenes)/10)]
pickle.dump( testscenes, open( "testscenes.cpkl", "wb" ) )
testscenes = pickle.load( open( "testscenes.cpkl", "rb" ) )

for my_scene in testscenes:
    trash=[] #empty trash bc instances are not kept across scenes
    
    log = nusc.get('log', my_scene['log_token'])
    loglist.append(log['location'])
    if (log['location'] != 'singapore-hollandvillage'):
        if (log['location'] == 'singapore-onenorth'):
            contextnum = 1
        if (log['location'] == 'boston-seaport'):
            contextnum = 2
        if (log['location'] == 'singapore-queenstown'):
            contextnum = 3
        
        firstsampletoken = my_scene["first_sample_token"]
        samp = nusc.get('sample', firstsampletoken)
        nextexists=True
        scenelist = np.array([0,0,0,0,0], dtype='float64').reshape((1,5))
        sampcount=0
        while (nextexists):
            samprow = np.array([0,0,0,0,0], dtype='float64').reshape((1,5))
            
            for ann in samp['anns']:
                anno = nusc.get('sample_annotation', ann)
                #vehicle or pedestrian
                instoken = anno['instance_token']
                if instoken not in instances:
                    instances.append(instoken)
                if 'vehicle' in anno['category_name']:
                    pedid = instances.index(instoken)+1
                    annadd = np.array([sampcount, pedid, contextnum, anno['translation'][0], anno['translation'][1]], dtype='float64').reshape((1,5))
                    samprow = np.concatenate((samprow, annadd))
            samprow = samprow[1:]
            samprow.sort(axis=0)
            scenelist = np.concatenate((scenelist, samprow))
            
            if (samp['next'] == ""):
                print('completed', count, 'out of', len(testscenes))
                count=count+1
                nextexists=False
            else:
                samp = nusc.get('sample', samp['next'])
                sampcount=sampcount+1
        scenelist = scenelist[1:]
        data.append(scenelist)

min_position_x = 1000
max_position_x = -1000
min_position_y = 1000
max_position_y = -1000

all_frame_data = []
valid_frame_data = []
frameList_data = []
numPeds_data = []
val_fraction = 0.3
dataset_index = 0

#THE REST OF THIS IS FORKED CODE FROM HUANG-XX'S TRAFFICPREDICT REPO

 # all_frame_data would be a list of list of numpy arrays corresponding to each dataset
# Each numpy array will correspond to a frame and would be of size (numPeds, 3) each row
# containing pedID, x, y
all_frame_data = []
# Validation frame data
valid_frame_data = []
# frameList_data would be a list of lists corresponding to each dataset
# Each list would contain the frameIds of all the frames in the dataset
frameList_data = []
# numPeds_data would be a list of lists corresponding to each dataset
# Ech list would contain the number of pedestrians in each frame in the dataset
numPeds_data = []
# Index of the current dataset
dataset_index = 0

min_position_x = 1000
max_position_x = -1000
min_position_y = 1000
max_position_y = -1000

for data in raw:
    min_position_x = min(min_position_x, min(data[:, 3]))
    max_position_x = max(max_position_x, max(data[:, 3]))
    min_position_y = min(min_position_y, min(data[:, 4]))
    max_position_y = max(max_position_y, max(data[:, 4]))
    
# For each dataset
for data in raw:
    data[:, 3] = (
        (data[:, 3] - min_position_x) / (max_position_x - min_position_x)
    ) * 2 - 1
    data[:, 4] = (
        (data[:, 4] - min_position_y) / (max_position_y - min_position_y)
    ) * 2 - 1

    data = data[~(data[:, 2] == 5)]

    # Frame IDs of the frames in the current dataset
    frameList = np.unique(data[:, 0]).tolist()
    numFrames = len(frameList)

    # Add the list of frameIDs to the frameList_data
    frameList_data.append(frameList)
    # Initialize the list of numPeds for the current dataset
    numPeds_data.append([])
    # Initialize the list of numpy arrays for the current dataset
    all_frame_data.append([])
    # Initialize the list of numpy arrays for the current dataset
    valid_frame_data.append([])

    skip = 1

    for ind, frame in enumerate(frameList):

        ## NOTE CHANGE
        if ind % skip != 0:
            # Skip every n frames
            continue

        # Extract all pedestrians in current frame
        pedsInFrame = data[data[:, 0] == frame, :]

        # Extract peds list
        pedsList = pedsInFrame[:, 1].tolist()

        # Add number of peds in the current frame to the stored data
        numPeds_data[dataset_index].append(len(pedsList))

        # Initialize the row of the numpy array
        pedsWithPos = []
        # For each ped in the current frame
        for ped in pedsList:
            # Extract their x and y positions
            current_x = pedsInFrame[pedsInFrame[:, 1] == ped, 3][0]
            current_y = pedsInFrame[pedsInFrame[:, 1] == ped, 4][0]
            current_type = pedsInFrame[pedsInFrame[:, 1] == ped, 2][0]
            # print('current_type    {}'.format(current_type))
            # Add their pedID, x, y to the row of the numpy array
            pedsWithPos.append([ped, current_x, current_y, current_type])

        if (ind > numFrames * val_fraction):
            # At inference time, no validation data
            # Add the details of all the peds in the current frame to all_frame_data
            all_frame_data[dataset_index].append(
                np.array(pedsWithPos)
            )  # different frame (may) have diffenent number person
        else:
            valid_frame_data[dataset_index].append(np.array(pedsWithPos))

    dataset_index += 1
# Save the tuple (all_frame_data, frameList_data, numPeds_data) in the pickle file
f = open('pedestrian/data/trajectories_TP_pedestrians_test.cpkl', "wb")
#f = open('vehicle/data/trajectories_TP_vehicles_test.cpkl', "wb")
pickle.dump(
    (all_frame_data, frameList_data, numPeds_data, valid_frame_data),
    f,
    protocol=2,
)
f.close()