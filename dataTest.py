from lyft_dataset_sdk.lyftdataset import LyftDataset
level5data = LyftDataset(data_path='D:/LEVEL5/v1.01-train', json_path='D:/LEVEL5/v1.01-train/v1.01-train', verbose=True)

level5data.list_scenes()

my_scene = level5data.scene[100]

firstsampletoken = my_scene["first_sample_token"]
level5data.render_sample(firstsampletoken)
first = level5data.get('sample', firstsampletoken)
next = level5data.get('sample', first['next'])
level5data.render_sample(next['token'])
timediff = next['timestamp']-first['timestamp'] #should show time difference, not working right now, needed for displacement calc

sensor_channel = 'LIDAR_TOP'  # also try this e.g. with 'LIDAR_TOP'
my_sample_data = level5data.get('sample_data', first['data'][sensor_channel])

my_annotation_token = first['anns'][5]
my_annotation =  my_sample_data.get('sample_annotation', my_annotation_token)
level5data.render_annotation(my_annotation_token)

#LOOK AT ANNOTATION STUFF TO GET COORDINATES AND OTHER STUFF