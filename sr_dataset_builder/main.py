from dataset import Dataset

if __name__ == '__main__':
    """
    Build a randomly-block-separated bathymetric chart dataset
    """

    # low-resolution filename
    lr_fname = 'input/data_LR.pkl'
    # high-resolution filename
    hr_fname = 'input/data_HR.pkl'
    # output directory
    output_dir = 'output'

    # build a dataset
    Dataset(lr_fname, hr_fname).save(output_dir)
