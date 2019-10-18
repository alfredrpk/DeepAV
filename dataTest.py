from lyft_dataset_sdk.lyftdataset import LyftDataset
level5data = LyftDataset(data_path='D:/LEVEL5/v1.01-train', json_path='D:/LEVEL5/v1.01-train/v1.01-train', verbose=True)

level5data.list_scenes()

my_scene = level5data.scene[100]

firstsampletoken = my_scene["first_sample_token"]
level5data.render_sample(firstsampletoken)
first = level5data.get('sample', firstsampletoken)
next = level5data.get('sample', first['next'])
level5data.render_sample(next['token'])
timediff = next['timestamp']-first['timestamp']