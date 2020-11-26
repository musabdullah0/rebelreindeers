import pandas as pd
import numpy as np
from numpy import linalg as la
import csv

'''
000000.csv
    label, yolo
    label, yolo

000005.csv
    label, yolo

img name
    - class 
        - yolo
        - truth
    - pose
        - yolo
        - truth
    - other
        - yolo
        - truth

'''

'''
returns an array arr
    [1 0 2]
    label[0] <-> yolo_obj[1]
    label[1] <-> yolo_obj[0]
    label[2] <-> yolo_obj[2]
'''
def difference(labels, yolos):
    pairs = []

    while labels.size > 0:
        label = labels[0]
        diffs = [la.norm(label - y) for y in yolos]
        yolo = yolos[np.argmin(diffs)]
        pairs.append([list(label), list(yolo)])
        labels = np.delete(labels, 0, 0)

    return pairs


'''
writes a file in the `merged` directory with both the yolo outputs and corresponding labels together
'''
def write_file(img_name, ground_truth, yolo_outputs):
    root_dir = 'merged/'
    file_name = root_dir + img_name.replace('png', 'csv')[-10:] # get the name of the file we're writing to
    print('writing', file_name)

    truth = np.array(ground_truth)
    yolos = np.array(yolo_outputs)
    pairs = difference(truth, yolos)

    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        for pair in pairs:
            writer.writerow(pair)



labels_dir = './dataset/training/label_2/'
images_dir = './dataset/training/image_2/'
yolo_csv = 'yolo-outputs.csv'

yolodf = pd.read_csv('yolo-outputs.csv')

# key - file name, value - yolo outputs
files = {}

for i in range(len(yolodf)):
    row = yolodf.iloc[i].to_numpy()

    # getting image name
    img_name = row[0]

    # getting yolo outputs
    yolo_outputs = row[2]
    yolo_outputs = yolo_outputs.split(',')
    yolo_outputs = [float(x) for x in yolo_outputs]
    yolo_outputs = yolo_outputs[:-1] # ignore the last value (classification uncertainty from yolo)

    # put in files dict
    if img_name not in files:
        files[img_name] = [yolo_outputs]
    else:
        files[img_name].append(yolo_outputs)
        


types = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2, 'Tram': 0, 'Person_sitting': 1, 'Misc': 3, 'DontCare': 3, 'Van': 0, 'Truck': 0}
image_names = sorted(files.keys())

for j, img_name in enumerate(image_names):
    label_name = img_name.replace('image_2', 'label_2').replace('png', 'txt')
    labels = []
    with open(label_name, 'r') as f:
        labels = f.readlines()

    truth = []
    for i, l in enumerate(labels):
        label = l.split()
        label[0] = types[label[0]]
        if label[0] == 3: # don't care or misc classification
            continue
        label = [float(x) for x in label]
        truth.append(label)

    write_file(img_name, truth, files[img_name])
