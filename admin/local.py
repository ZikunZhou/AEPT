class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = ''
        self.lasot_lmdb_dir = ''
        self.lasot_meta_dir = ''
        self.got10k_lmdb_dir = ''
        self.got10k_meta_dir = ''
        self.trackingnet_lmdb_dir = ''
        self.trackingnet_meta_dir = ''
        self.coco_dir = ''
        self.imagenet_lmdb_dir = ''
        self.imagenet_meta_dir = ''
        self.imagenet_dir = ''
        self.det_lmdb_dir = ''
        self.det_meta_dir = ''
        self.youtubebb_root = ''
        self.youtubebb_anno = ''
