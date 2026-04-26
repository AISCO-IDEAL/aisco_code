


class LossNormalisationManager:
    """
    Class to normalise the loss values.

    Should give values to the TotalLossV2 class which it can then use to further weight the values there

    """

    def __init__(self):
        pass


    def calibrate_loss_factors(self):
        """
        Calibrate the loss factors.

        This method should be called at least after the first epoch of training, to calibrate the loss factors based on the observed loss values.
        """


    def calibrate_gradient_factors(self):
        """
        Calibrate the gradient factors.

        This method should be called at least after the first epoch of training, to calibrate the gradient factors based on the observed gradient values.
        """
