def get_data(args, return_val=False, val_downscale=4.0, **overwrite_cfgs):
    dataset_type = args.data.get('type', 'DTU')
    cfgs = {
        'scale_radius': args.data.get('scale_radius', -1),
        'downscale': args.data.downscale,
        'data_dir': args.data.data_dir,
        'train_cameras': False
    }
    
    if dataset_type == 'DTU':
        from .DTU import SceneDataset
        cfgs['cam_file'] = args.data.get('cam_file', None)
    elif dataset_type == 'custom':
        from .custom import SceneDataset
    elif dataset_type == 'BlendedMVS':
        from .BlendedMVS import SceneDataset
    elif dataset_type == 'MviewTemporalSMPL':
        from .MviewTemporalSMPL import SceneDataset
    else:
        raise NotImplementedError

    cfgs.update(overwrite_cfgs)
    if dataset_type == 'MviewTemporalSMPL':
        cfgs['views'] = args.data.get('train_views', [0, 4, 8, 12, 16, 20])
        cfgs['subjects'] = args.data.get('subjects', [])
        cfgs['uv_size'] = args.data.get('uv_size', 256)
        cfgs['select_frame'] = args.data.get('select_frame', 'uniform')
        cfgs['num_frame'] = args.data.get('num_frame', 1)
        cfgs['start_frame'] = args.data.get('start_frame', 0)
        cfgs['end_frame'] = args.data.get('end_frame', -1)
    if dataset_type == 'DTU':
        cfgs['views'] = args.data.get('train_views', [])

    dataset = SceneDataset(**cfgs)
    if return_val:
        if dataset_type == 'MviewTemporalSMPL':
            cfgs['views'] = args.data.get('test_views', [5, 10, 15])
        if dataset_type == 'DTU':
            cfgs['views'] = args.data.get('test_views', [])

        cfgs['downscale'] = val_downscale
        val_dataset = SceneDataset(**cfgs)
        return dataset, val_dataset
    else:
        return dataset