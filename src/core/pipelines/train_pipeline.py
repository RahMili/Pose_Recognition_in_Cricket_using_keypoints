import numpy as np
import matplotlib.pyplot as plt
from src.core.pipelines.base_pipeline import BasePipeline
from src.core.utils.model_loader import Best_Weights


class TrainPipeline(BasePipeline):
    def __init__(self, model, train_x, train_y, test_x, test_y):
        self.model = model
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def __train(self):
        # function for plotting roc
        from sklearn.metrics import roc_curve, auc
        def plot_keras_multi_roc(test_y_one_hot, y_pred):
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(5):
                fpr[i], tpr[i], _ = roc_curve(test_y_one_hot[:, i], y_pred[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            fpr["micro"], tpr["micro"], _ = roc_curve(test_y_one_hot.ravel(), y_pred.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            n_classes = 5
            from itertools import cycle

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)

            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            colors = cycle(['blue', 'red', 'green', 'grey', 'purple'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                               ''.format(classes[i], roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Extending the ROC Curve for ANN to Multi-Class')
            plt.legend(loc="lower right")
            plt.show()
        #loading the model from the Model class

        print(self.model.summary())
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(self.train_x, self.train_y, epochs=200, validation_data=[self.test_x, self.test_y], callbacks=[Best_Weights()])
        self.model.save('pose_recognition.h5')

    def execute(self, configs):
        self.__train(self)
