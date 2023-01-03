from src.core.pipelines.train_pipeline import TrainPipeline
from src.core.utils.model_loader import Model
from src.core.utils.data_loader import Loader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


m = Model()
model = m.model
# declaring the classes for classification
classes = ['no action', 'no ball', 'wide', 'sixes', 'out']
#getting the training data from the data loader
loader = Loader(train=True)
train_x = loader.x
train_y = loader.y
#getting the test data from the data loader
loader = Loader(train=False)
test_x = loader.x
test_y = loader.y
train_x = train_x.reindex(sorted(train_x.columns), axis=1)
test_x = test_x.reindex(sorted(test_x.columns), axis=1)
#preprocessing the keypoints
scaler = MinMaxScaler()
scaler.fit(train_x)
train_x = pd.DataFrame(scaler.transform(train_x))
scaler.fit(test_x)
test_x = pd.DataFrame(scaler.transform(test_x))
train_x = train_x.fillna(0)
test_x = test_x.fillna(0)
# making an object of TrainPipeline to initiate training
IP = TrainPipeline(model, train_x, train_y, test_x, test_y)