from .zjumocap import ZJUMoCapDataset
from .people_snapshot import PeopleSnapshotDataset
from .prox import Prox
from .prox_add_scene import ProxAddScene
from .egobody_add_scene import EgobodyAddScene
from .i3db_add_scene import I3dbAddScene
from .emdb_add_scene import EMDBAddScene

def load_dataset(cfg, split='train', is_coarse=False):
    dataset_dict = {
        'zjumocap': ZJUMoCapDataset,
        'people_snapshot': PeopleSnapshotDataset,
        "prox": Prox,
        "prox_add_scene": ProxAddScene,
        "egobody_add_scene": EgobodyAddScene,
        "i3db_add_scene": I3dbAddScene,
        "emdb_add_scene": EMDBAddScene,
    }
    return dataset_dict[cfg.name](cfg, split, is_coarse)
