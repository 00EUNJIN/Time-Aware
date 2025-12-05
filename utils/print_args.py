def print_args(args):
    print('Arguments:')
    for arg in vars(args):
        print(f'    {arg}: {getattr(args, arg)}')