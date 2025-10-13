import os

ROOT_DIR = {
    'sceneflow':        '/data/xp/Scene_Flow/',
    'kitti2015':        '/data/xp/KITTI_2015/',
    'kitti2012':        '/data/xp/KITTI_2012/',
    'middlebury':       '/data/xp/Middlebury/MiddleburyH/',
    'eth3d':            '/data/xp/ETH3D/',
    'drivingstereo':    '/data/xp/Driving_Stereo/',
    'booster':          '/data/xp/Booster/'
}

class Datasets_List():
    def __init__(self, *datasets):
        """
        - sceneflow:
            - train_cleanpass
            - test_cleanpass
            - train_finalpass
            - test_final_pass
        - kitti2015:
            - train
            - val
            - test
        - kitti2012:
            - train
            - val
            - test
        - middlebury:
            - train
            - val
            - test
        - eth3d:
            - train
            - val
            - test
        - drivingstereo:
            - train_half
            - test_half
            - test_full
            - {train/test}_{half/full}_{cloudy/foggy/rainy/sunny} 
        """
        self.dirname = os.path.dirname(os.path.abspath(__file__))
        self.left_images = []
        self.right_images = []
        self.disp_images = []

        for dataset, mode in datasets:
            left_images, right_images, disp_images = self.read_lines(os.path.join(self.dirname, dataset, mode+'.txt'))
            self.left_images += [os.path.join(ROOT_DIR[dataset], line) for line in left_images]
            self.right_images += [os.path.join(ROOT_DIR[dataset], line) for line in right_images]
            self.disp_images += [os.path.join(ROOT_DIR[dataset], line) if line else None for line in disp_images ]

    def read_lines(self, filename: str):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:
            return left_images, right_images, [None] * len(left_images)
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images
        