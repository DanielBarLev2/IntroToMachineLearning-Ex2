#################################
# Your name: Daniel Bar Lev (211992425)
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    @staticmethod
    def sample_from_D(m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two-dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        data = np.random.uniform(low=0, high=1, size=m)
        labels = np.zeros_like(data)

        for index in range(m):
            if 0 <= data[index] <= 0.2 or 0.4 <= data[index] <= 0.6 or 0.8 <= data[index] <= 1:
                labels[index] = np.random.binomial(n=1, p=0.8)
            else:
                labels[index] = np.random.binomial(n=1, p=0.1)

        samples = np.column_stack((data, labels))

        return samples

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two-dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        es_avg = []
        ep_avg = []

        ns = np.arange(m_first, m_last + 1, step)

        for n in ns:
            es_sum, ep_sum = 0.0, 0.0
            print(n/5)
            for _ in range(T):
                samples = self.sample_from_D(n)
                samples = samples[np.argsort(samples[:, 0])]
                data, labels = samples[:, 0], samples[:, 1]
                best_intervals, es = intervals.find_best_interval(xs=data, ys=labels, k=k)
                es_sum += (1 / n) * es

                true_error = self.compute_true_error(best_intervals)
                ep_sum += true_error

            es_avg.append(es_sum / T)
            ep_avg.append(ep_sum / T)

        plt.plot(ns, es_avg, label='Empirical Error')
        plt.plot(ns, ep_avg, label='True Error')
        plt.xlabel('Sample Size (n)')
        plt.ylabel('Error')
        plt.legend()
        plt.show()

        return np.array([es_avg, ep_avg]).T

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        # TODO: Implement the loop
        pass

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        # TODO: Implement the loop
        pass

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # TODO: Implement me
        pass

    #################################
    # Place for additional methods
    #################################

    @staticmethod
    def compute_true_error(interval_list):
        """Calculate the true error ep(hI) for a given list of intervals I.
        Input: intervals - a list of tuples representing the intervals.

        Returns: The true error e_P(hI).
        """
        error = 0.0
        high_prob_intervals = [(0, 0.2), (0.4, 0.6), (0.8, 1)]  # P[y=1|x] = 0.8 / P[y=0|x] = 0.2
        low_prob_intervals = [(0.2, 0.4), (0.6, 0.8)]           # P[y=1|x] = 0.1 / P[y=0|x] = 0.9

        # create intervals for 0 guesses
        inverse_interval_list = []
        i_start, i_end = 0, 0
        for r in range(len(interval_list)):
            start, end = interval_list[r]
            i_end = start
            inverse_interval_list.append((i_start, i_end))
            i_start = end
        inverse_interval_list.append((i_start, 1))

        def compute_prob(interval, true_interval_list, prob):
            total_prob = 0.0
            start, end = interval

            for true_interval in true_interval_list:
                low, high = true_interval

                # the true_interval is fully contained in the interval
                if start <= low <= high <= end:
                    total_prob += (high - low) * prob

                # the start point of true_interval contained in the interval
                elif start <= low <= end <= high:
                    total_prob += (end - low) * prob

                # the end point of true_interval contained in the interval
                elif low <= start <= high <= end:
                    total_prob += (high - start) * prob

                # the true_interval is not contained in the interval at all
                else:
                    continue

            return total_prob

        for interval in interval_list:
            # mistake penalty for guess 1 in high_prob_intervals
            error += compute_prob(interval, high_prob_intervals, 0.2)
            # mistake penalty for guess 1 in low_prob_intervals
            error += compute_prob(interval, low_prob_intervals, 0.9)

        for interval in inverse_interval_list:
            # mistake penalty for guess 0 in high_prob_intervals
            error += compute_prob(interval, high_prob_intervals, 0.8)
            # mistake penalty for guess 0 in low_prob_intervals
            error += compute_prob(interval, low_prob_intervals, 0.1)

        return error


if __name__ == '__main__':
    ass = Assignment2()
    ass.sample_from_D(100)
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
