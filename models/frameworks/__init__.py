def get_model(args):       
    if args.model.framework == 'UNISURF':
        from .unisurf import get_model
    elif args.model.framework == 'NeuS':
        from .neus import get_model
    elif args.model.framework == 'VolSDF':
        from .volsdf import get_model
    elif args.model.framework == 'nerfpp':
        from .nerfpp import get_model
    elif args.model.framework == 'NeuSSegm':
        from .neussegm import get_model
    elif args.model.framework == 'GarmentNerf':
        from .garmnerf import get_model
    elif args.model.framework == 'NeuralBody':
        from .neuralbody import get_model
    else:
        raise NotImplementedError
    return get_model(args)