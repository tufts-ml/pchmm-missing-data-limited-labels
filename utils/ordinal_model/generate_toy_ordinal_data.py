#!/usr/bin/env python3
"""_summary_

TODO
----
* Create method of generating cresent data
* generate method for generating stacked gaussian data
* 
"""
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import argparse


class ToyOrdinalData:
    """_summary_
    """

    def __init__(self, num_ordinal_labels: int, density: float = 1, seed: int = None) -> None:
        """_summary_

        Parameters
        ----------
        num_ordinal_labels : int
            _description_
        type : str
            _description_

        TODO
        ----
        * specify num ordinal labels--Done
        * choose toy data type--Done
        * Set up switch statement to generate appropriate data--Working
        """
        # Parameters
        self.num_ordinal_labels = num_ordinal_labels
        self.density = density
        self.rs = np.random.RandomState(seed)
        self.export_path = Path(__file__).parents[2].joinpath(
            'toydata', 'ordinal_data')
        self.mean_scale = None

    def generate_data(self, toy_type: str, N_scale: int = 500):
        self.N_scale = N_scale
        if toy_type == 'stacked':
            self.stacked_gaussian()
        elif toy_type == 'crescents':
            self.crescent_shapes()
        elif toy_type == 'circles':
            self.concentric_circles()
        elif toy_type == 'semi-circles':
            self.concentric_circles(circle_type='semi')
        elif toy_type == 'quarter-circles':
            self.concentric_circles(circle_type='quarter')
        else:
            pass

    def stacked_gaussian(self, mean_scale: float = 2):
        self.filename = 'stacked_gaussian'
        self.plot_title = 'Stacked Gaussian'
        self.mean_scale = mean_scale
        mean_D2 = np.hstack((np.zeros((self.num_ordinal_labels, 1)),
                             np.arange(0, self.num_ordinal_labels*mean_scale, mean_scale)[:, np.newaxis]))
        cov_D22 = np.stack([np.diag([1, 2])
                           for i in range(self.num_ordinal_labels)])
        data_N2 = np.zeros((0, 2))
        labels_N = np.zeros((0, 1))
        for i in range(self.num_ordinal_labels):
            n = int(self.N_scale*self.density)
            data_i2 = self.rs.multivariate_normal(
                mean_D2[i, :], cov_D22[i, :, :], size=n)
            labels_i = np.ones((n, 1)) * i
            data_N2 = np.vstack((data_N2, data_i2))
            labels_N = np.vstack((labels_N, labels_i))
        self.data = data_N2.copy()
        self.labels = labels_N.squeeze().copy()

    def crescent_shapes(self, mean_scale: float = 2.5):
        self.filename = 'crescent_shapes'
        self.plot_title = 'Crescent Shapes'
        mean_D2 = np.hstack((np.arange(0, self.num_ordinal_labels*mean_scale, mean_scale)[
                            :, np.newaxis], np.zeros((self.num_ordinal_labels, 1))))
        cov_D22 = np.stack([np.diag([(i+1)/2, (i+1)/2])
                           for i in range(self.num_ordinal_labels)])
        data_N2 = np.zeros((0, 2))
        labels_N = np.zeros((0, 1))
        for i in range(self.num_ordinal_labels):
            n = int(self.N_scale*(i+1)**self.density)
            data_i2 = self.rs.multivariate_normal(
                mean_D2[i, :], cov_D22[i, :, :], size=n)
            labels_i = np.ones((n, 1)) * i
            radius_mask = np.sqrt(np.sum(data_i2**2, axis=1)) < i*mean_scale
            if i == 0:
                data_N2 = np.vstack((data_N2, data_i2))
                labels_N = np.vstack((labels_N, labels_i))
            else:
                data_N2 = np.vstack((data_N2, data_i2[radius_mask]))
                labels_N = np.vstack((labels_N, labels_i[radius_mask]))
        self.data = data_N2.copy()
        self.labels = labels_N.squeeze().copy()

    def concentric_circles(self, padding: float = 0.2, circle_type: str = None):
        self.filename = 'concentric_circles'
        self.plot_title = 'Concentric Circles'
        mean_12 = np.array([0, 0])
        cov_D22 = np.stack([np.diag([(i+2), (i+2)])
                           for i in range(self.num_ordinal_labels)])
        data_N2 = np.zeros((0, 2))
        labels_N = np.zeros((0, 1))
        for i in range(self.num_ordinal_labels):
            n = int(self.N_scale*(i+1)**self.density)
            data_i2 = self.rs.multivariate_normal(
                mean_12, cov_D22[i, :, :], size=n)
            labels_i = np.ones((n, 1)) * i
            radius_from_origin = np.sqrt(np.sum(data_i2**2, axis=1))
            radius_mask = (radius_from_origin >
                           i-padding) & (radius_from_origin < i+1+padding)
            if circle_type == 'semi':
                self.filename = 'concentric_semi_circles'
                self.plot_title = 'Concentric Semi-Circles'
                radius_mask = (radius_mask) & (data_i2[:, 0] > 0)
            if circle_type == 'quarter':
                self.filename = 'concentric_quarter_circles'
                self.plot_title = 'Concentric Quarter-Circles'
                radius_mask = (radius_mask) & (
                    data_i2[:, 0] > 0) & (data_i2[:, 1] > 0)
            # if i == 0:
            #     data_N2 = np.vstack((data_N2, data_i2))
            #     labels_N = np.vstack((labels_N, labels_i))
            # else:
            data_N2 = np.vstack((data_N2, data_i2[radius_mask]))
            labels_N = np.vstack((labels_N, labels_i[radius_mask]))
        self.data = data_N2.copy()
        self.labels = labels_N.squeeze().copy()

    def sigmoid_shape(self):
        pass

    def plot_data(self, export: bool = False):
        # Data
        data_N2 = self.data.copy()
        labels_N = self.labels.copy()

        # Plotting
        fig, ax = plt.subplots()
        for i in range(self.num_ordinal_labels):
            data_i = data_N2[labels_N == i]
            ax.scatter(data_i[:, 0], data_i[:, 1],
                       marker='x', linewidths=1, label=f'y={i}')
        formatted_title = f'{self.plot_title} Ordinal Data with {self.num_ordinal_labels} labels\nDensity factor = {self.density}'
        if self.mean_scale is not None:
            formatted_title += f'\nMean scale = {self.mean_scale}'
        ax.set_title(formatted_title)
        ax.grid(True)
        ax.legend()
        ax.axis('equal')

        formatted_filename = f'{self.filename}_{self.num_ordinal_labels}_labels_{self.density}_densityfactor'
        if self.mean_scale is not None:
            formatted_filename += f'_{self.mean_scale}_meanscale'
        formatted_filename += '.png'

        if export:
            plt.savefig(self.export_path.joinpath(formatted_filename),
                        bbox_inches='tight', pad_inches=0)
        else:
            plt.show()

    def export_data(self):
        data_labels_df = pd.DataFrame(
            self.data, columns=['x1', 'x2'])
        data_labels_df['ordinal_label'] = self.labels.astype(int)

        formatted_filename = f'{self.filename}_{self.num_ordinal_labels}_labels_{self.density}_densityfactor'
        if self.mean_scale is not None:
            formatted_filename += f'_{self.mean_scale}_meanscale'
        formatted_filename += '.csv'

        data_labels_df.to_csv(self.export_path.joinpath(
            formatted_filename), index=False)


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_ordinal_labels', type=int, default=2,
                        help='number of ordinal labels desired')
    parser.add_argument('--density_factor', type=float, default=1,
                        help='Desired density of points at each label')
    parser.add_argument('--seed', type=int, default=10,
                        help='whether to seed the data generation')
    parser.add_argument('--toy_type', type=str,
                        help='what sort of toy data to generate')
    # parser.add_argument('--export_plot', type=bool, default=True,
    #                     help='Whether to export the plot')
    args = parser.parse_args()

    num_ordinal_labels = args.num_ordinal_labels
    density_factor = args.density_factor
    seed = args.seed
    toy_type = args.toy_type
    # export_plot = args.export_plot

    # Generate data
    generator = ToyOrdinalData(
        num_ordinal_labels=num_ordinal_labels, density=density_factor, seed=seed)
    generator.generate_data(toy_type=toy_type)
    generator.export_data()
    generator.plot_data(export=True)
