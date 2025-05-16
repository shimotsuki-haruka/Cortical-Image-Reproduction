import tempfile


workdir = None


def default_workdir():
    return tempfile.mkdtemp(prefix="fluoromind_workbench_")


def update_workdir(_workdir):
    global workdir
    workdir = _workdir


def get_workdir():
    global workdir
    if workdir is None:
        workdir = default_workdir()
    return workdir
