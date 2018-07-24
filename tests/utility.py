import pkg_resources

resource_package = 'mclust'


def apply_resource(directory, file, func):
    resource_path = '/'.join(('resources', directory, file))
    with pkg_resources.resource_stream(resource_package, resource_path) as f:
        return func(f)
