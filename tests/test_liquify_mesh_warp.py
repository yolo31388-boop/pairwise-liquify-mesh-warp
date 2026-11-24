import pytest
import imaging.liquify_mesh_warp as mod

CASES = [
    ("radius_in_image_space", (1,)),
    ("feather_boundary", (1,)),
    ("normalize_grid", (1,)),
    ("deep_copy_snapshot", (1,)),
    ("double_buffer", (1,)),
    ("undo_step", (1,)),
    ("load", (1,)),
    ("warp", (1,)),
    ("preview", (1,)),
    ("export", (1,)),
    ("undo", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
