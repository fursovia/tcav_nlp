from sklearn.linear_model import LogisticRegression


class CAV:

    def __init__(self):

        self.cav = None

    def fit(self, X, y):
        """
        Args:
            X: f(concepts) and f(random concepts)
            y: 0 or 1
        """

        self.lm = LogisticRegression()
        self.lm.fit(X, y)

        self.cav = self.lm.coef_

        return self.cav
