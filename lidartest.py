import numpy as np

file_path = 'D:/mini/sweeps/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151605247935.pcd.bin'
bruh = []
with open(file_path, "rb") as f:
    number = f.read(4)
    while number != b"":
        print(np.frombuffer(number, dtype=np.float32))
        bruh.append(np.frombuffer(number, dtype=np.float32))
        number = f.read(4)

plswork = np.array(bruh)