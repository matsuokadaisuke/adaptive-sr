class MinMaxNormalizer:
    """
    Normalize images in the range from 0 to 1

    Attributes:
        min: min value
        max: max value
    """

    def __init__(self, xmin, xmax, ymin, ymax):
        """
        Args:
            min: min value
            max: max value
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def normalize(self, imgs, min, max):
        """
        Normalize images in the range from 0 to 1

        Args:
            imgs: input images

        Returns:
            normalized images in the range from 0 to 1
        """
        return (imgs - min) / (max - min)
        #return (imgs - self.min) / (self.max - self.min)

    def denormalize_x(self, imgs):
        """
        Denormalize images

        Args:
            imgs: input images

        Returns:
            denormalized images
        """
        return (imgs * (self.xmax - self.xmin)) + self.xmin

    def denormalize_y(self, imgs):
        """
        Denormalize images

        Args:
            imgs: input images

        Returns:
            denormalized images
        """
        return (imgs * (self.ymax - self.ymin)) + self.ymin