import os
import typing

from sklearn.gaussian_process.kernels import RBF
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import WhiteKernel
from scipy.spatial.distance import cdist
from scipy.stats import norm
import math


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 100  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 25  # Number of points displayed in 3D during evaluation


# Cost function constants
THRESHOLD = 35.5
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 5.0
COST_W_THRESHOLD = 20.0

SQRT_NUM_OF_CENTERS = 20
NUM_OF_CENTERS = SQRT_NUM_OF_CENTERS**2
NUM_OF_CENTERS_TO_ASSIGN = 4


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        # TODO: Add custom initialization for your model here if necessary
        self.rng = np.random.RandomState(seed=0)
        np.random.seed(0)

        self.gprs = [GaussianProcessRegressor(
                           kernel=RBF() + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-6, 10)),
                           n_restarts_optimizer=20,
                           normalize_y=True,
                           random_state=self.rng,
                        ) for _ in range(NUM_OF_CENTERS)]
        self.centers = np.mgrid[0:1:SQRT_NUM_OF_CENTERS*1j, 0:1:SQRT_NUM_OF_CENTERS*1j].reshape(2,-1).T  # grid

    def predict(self, x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param x: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """
        # TODO: Use your GP to estimate the posterior mean and stddev for each location here

        closest_center = np.argmin(cdist(x, self.centers), axis=1)
        gp_mean, gp_std = np.zeros(x.shape[0]), np.zeros(x.shape[0])
        predictions=np.zeros(gp_mean.shape)
        for i in range(NUM_OF_CENTERS):
            assigned = np.where(closest_center == i)[0]
            if assigned.size:
                mean, std = self.gprs[i].predict(x[assigned], return_std=True)
                gp_mean[assigned] = mean
                gp_std[assigned] = std - math.exp(self.gprs[i].kernel_.theta[1])

        # TODO: Use the GP posterior to form your predictions here

        # predictions = np.where((THRESHOLD - 3 < gp_mean) & (gp_mean < THRESHOLD + 1),
        #                        THRESHOLD,
        #                        gp_mean - 1)
        for i in range(0,len(gp_mean)):
            space=np.linspace(gp_mean[i]-4,gp_mean[i]+4,300)
            costs=np.zeros(space.shape)
            prob_array=norm.pdf(space,gp_mean[i],1)
            for j in range(0,len(space)):
                weight=np.ones((space.shape))
                weight[space<space[j]]=5 
                if(space[j]<THRESHOLD):
                    weight[(space>THRESHOLD)]=20
                distances=space-space[j]
                distances=[number**2 for number in distances]
                costs[j]=np.dot(np.transpose(weight),(np.multiply(distances,prob_array)))
            predictions[i]=space[np.argmin(costs)]
        return predictions, gp_mean, gp_std

    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        # TODO: Fit your model here

        distances = cdist(train_x, self.centers)
        closest_centers = np.argsort(distances)[:,:NUM_OF_CENTERS_TO_ASSIGN]
        for i in range(NUM_OF_CENTERS):
            assigned = [j for j, centers in enumerate(closest_centers) if i in centers]
            print(f"assigned {len(assigned)} data points to center {i}")
            if assigned:
                x = train_x[assigned]
                y = train_y[assigned]
                self.gprs[i].fit(x, y)


def cost_function(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param y_true: Ground truth pollution levels as a 1d NumPy float array
    :param y_predicted: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert y_true.ndim == 1 and y_predicted.ndim == 1 and y_true.shape == y_predicted.shape

    # Unweighted cost
    cost = (y_true - y_predicted) ** 2
    weights = np.zeros_like(cost)

    # Case i): overprediction
    mask_1 = y_predicted > y_true
    weights[mask_1] = COST_W_OVERPREDICT

    # Case ii): true is above threshold, prediction below
    mask_2 = (y_true >= THRESHOLD) & (y_predicted < THRESHOLD)
    weights[mask_2] = COST_W_THRESHOLD

    # Case iii): everything else
    mask_3 = ~(mask_1 | mask_2)
    weights[mask_3] = COST_W_NORMAL

    # Weigh the cost and return the average
    return np.mean(cost * weights)


def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
   # Load the training dateset and test features
    # x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    # y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)

    # TRAIN = 13000
    # TEST = 2100
    # train_x = x[:TRAIN]
    # train_y = y[:TRAIN]
    # test_x = x[TRAIN:TRAIN+TEST]
    # test_y = y[TRAIN:TRAIN+TEST]

    # # Fit the model
    # print('Fitting model')
    # model = Model()
    # model.fit_model(train_x, train_y)

    # # Predict on the test features
    # print('Predicting on test features')
    # all_predicted_y, predicted_mean, predicted_std = model.predict(test_x)
    # print("MSE", mean_squared_error(test_y, predicted_mean))
    # # for predicted_y in all_predicted_y:
    # #     print(cost_function(test_y, predicted_y))
    # print(predicted_std.mean(), predicted_std.min(), predicted_std.max())

    # if EXTENDED_EVALUATION:
    #     perform_extended_evaluation(model, output_dir='.')
    
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fit_model(train_x, train_y)

    # Predict on the test features
    print('Predicting on test features')
    predicted_y = model.predict(test_x)
    print(predicted_y)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
