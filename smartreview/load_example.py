def get_path_and_file_names(example):
    """
    Input:  either 'ex1' or ex2'
    """
    #root_dir = "/home/ubuntu/app_demo"
    root_dir = "/Users/rick/Git/insight/smartreview"
    pkl_dir = "/static/pkl"
    image_file = example+"_image.pkl"

    return root_dir, pkl_dir, image_file

def load_example(example):
    import pandas as pd
    import pickle

    root_dir, pkl_dir, image_file = get_path_and_file_names(example)

    df_image = pd.read_pickle(root_dir + pkl_dir + '/' + image_file)

    return df_image