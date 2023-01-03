import glob
from src.config import basic_config
from tqdm import tqdm
from tf_pose import common
from src.core.utils.model_loader import Model
import pandas as pd
import numpy as np


class Loader:
    def __init__(self, train):
        self.train = train
        self.x, self.y = self.__loader()

    def __loader(self):
        def humanToDict(hum):
            resultdict = {}
            parts = hum.body_parts.keys()
            for p in parts:
                resultdict[str(p) + '_x'] = hum.body_parts[p].x
                resultdict[str(p) + '_y'] = hum.body_parts[p].y
                resultdict[str(p) + '_score'] = hum.body_parts[p].score
            return resultdict

        if self.train:
            filenames = glob.glob(basic_config['DEFAULT']['train_path'])
        else:
            filenames = glob.glob(basic_config['DEFAULT']['test_path'])

        m = Model()
        model = m.loader(model_name='pose_estimator')
        keys = []
        images = []
        img_names = []

        pbar = tqdm(total=len(filenames))
        for img in filenames:
            image = common.read_imgfile(img, None, None)
            humans = model.inference(npimg=image, upsample_size=4.0)
            if len(humans) > 0:
                keys.append(humans[0])
                images.append(image)
                img_names.append(img)
            pbar.update(1)
        pbar.close()
        print("train images: ", len(images))
        keys_list = []
        for k in keys:
            keys_list.append(humanToDict(k))
        data = pd.DataFrame(keys_list)

        labels = []
        for img in img_names:
            if img.split("/")[-1].count("no_action") > 0:
                labels.append(0)
            if img.split("/")[-1].count("no_ball") > 0:
                labels.append(1)
            if img.split("/")[-1].count("wide") > 0:
                labels.append(2)
            if img.split("/")[-1].count("sixes") > 0:
                labels.append(3)
            if img.split("/")[-1].count("out") > 0:
                labels.append(4)
        labels = np.array(labels)
        labels = pd.DataFrame(labels)
        if self.train:
            data.to_csv('training_x.csv', index=False)
            labels.to_csv('training_y.csv', index=False)
            x = pd.read_csv('training_x.csv')
            y = pd.read_csv('training_y.csv')
        else:
            data.to_csv('test_x.csv', index=False)
            labels.to_csv('test_y.csv', index=False)
            x = pd.read_csv('test_x.csv')
            y = pd.read_csv('test_y.csv')

        return x, y





