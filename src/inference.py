import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.core.utils.model_loader import Model
from src.core.utils.data_loader import Loader


m = Model(train=False)
model = m.model
# loading test data from data loader
loader = Loader(train=False)
test_x = loader.x
test_y = loader.y
test_x = test_x.reindex(sorted(test_x.columns), axis=1)
# using minmax scaler
scaler = MinMaxScaler()
scaler.fit(test_x)
test_x = pd.DataFrame(scaler.transform(test_x))
test_x = test_x.fillna(0)

model.evaluate(test_x, test_y)
