class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'ddd17':
            return '../Semantic_Segmentation/DDD17/dataset_our_codification/'
        elif dataset == 'ddd17_images':
            return '../Semantic_Segmentation/DDD17/dataset_our_codification/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError