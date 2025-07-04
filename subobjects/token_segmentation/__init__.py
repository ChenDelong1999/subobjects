
def get_segmenter(**kwargs):
    if kwargs['type'] == 'patch':
        from .patch import PatchSegmenter
        return PatchSegmenter(**kwargs)
    
    elif kwargs['type'] == 'superpixel':
        from .superpixel import SuperpixelSegmenter
        return SuperpixelSegmenter(**kwargs)

    elif kwargs['type'] == 'epoc':
        from .epoc import EPOCSegmenter
        return EPOCSegmenter(**kwargs)
    
    elif kwargs['type'] == 'panoptic':
        from .panoptic import PanopticSegmenter
        return PanopticSegmenter(**kwargs)
    
    else:
        raise NotImplementedError