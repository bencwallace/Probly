import numpy as np
import probly as pr


class Human(object):
    def __init__(self, gender, height, weight):
        self.gender = gender
        self.height = height
        self.weight = weight


class randomHuman(pr.Distr):
    def __init__(self, female_stats, male_stats):
        self.female_stats = female_stats
        self.male_stats = male_stats

    def sampler(self, seed=None):
        np.random.seed(seed)
        gender = np.random.choice(2, p=[0.5, 0.5])
        if gender == 0:
            height_mean, weight_mean, cov = self.female_stats
        else:
            height_mean, weight_mean, cov = self.male_stats

        means = [height_mean, weight_mean]
        np.random.seed(seed)
        height, weight = np.random.multivariate_normal(means, cov)

        return Human(gender, height, weight)


def example():
    # Set desired female and male human statistics
    f_cov = np.array([[80, 5], [5, 99]])
    f_stats = [160, 65, f_cov]
    m_cov = np.array([[70, 4], [4, 110]])
    m_stats = [180, 75, m_cov]

    # Initialize a `randomHuman` object, sample from it,
    # and print his/her gender
    return randomHuman(f_stats, m_stats)


# Define a decorated BMI function
@pr.Lift
def BMI(self):
    return self.weight / (self.height / 100) ** 2


# Declare and sample frm the BMI of a random human
H = example()
B = BMI(H)
print(B(0))
