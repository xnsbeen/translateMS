import random

load_file_path = "C:/Users/ahoho/Downloads/Traning_TS_swiss_Carbam.txt"
save_file_path = "C:/Users/ahoho/Downloads/Traning_TS_swiss_Carbam_shuffled.txt"

with open(load_file_path,'r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
print('done loading')

with open(save_file_path,'w') as target:
    for _, line in data:
        target.write( line )