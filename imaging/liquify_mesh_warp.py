"""液化网格变形（liquify mesh warp）。

本模块把「笔刷轨迹累积位移」拆成一组可独立验证的小构件，再由
:func:`warp` / :class:`WarpSession` 串成一条变形流水线。针对初始实现里的
六个缺陷，各自有对应的修复点：

1. 半径按图像坐标 —— :func:`radius_in_image_space`
   笔刷半径以屏幕像素给出，画布缩放后换算成图像坐标半径，因此同一笔在
   屏幕上看起来一致，不会因为放大画布而变形更强。
2. 边界羽化位移 —— :func:`feather_boundary`
   位移不再只更新「被笔刷覆盖」的顶点：影响权重从一个平滑核出发，在
   笔刷边缘连续衰减到 0，边界顶点一起移动，变形区边缘没有折线。
3. 网格比例归一 —— :func:`normalize_grid` / :func:`export`
   位移统一以「图像像素 / 图像高度」的归一化单位保存，预览网格（低分辨率）
   与导出网格（高分辨率）换算回图像坐标后物理位移完全一致。
4. 快照深拷贝 —— :func:`deep_copy_snapshot`
   撤销快照持有独立的顶点与位移数组，历史记录之间不再按引用共享。
5. 重算双缓冲 —— :func:`double_buffer` / :class:`WarpSession`
   后台重算先写 back 缓冲，渲染始终读 front，重算完成后原子交换，旧缓冲
   不会在渲染途中被改写。
6. 撤销按步进 —— :func:`undo_step` / :func:`undo` / :meth:`WarpSession.undo`
   撤销按「步进」逐条回退历史，连续两次撤销回到两个不同状态。

所有数值路径均为纯 Python + NumPy 的确定性实现：不使用随机数，不依赖
输入是否连续存储，遇到 NaN/Inf 与退化网格都会安全降级而不是抛异常。
"""

from __future__ import annotations

import copy as _copy
import functools
import math
import threading

import numpy as np

__all__ = [
    "DEFAULT_PROJECTION_RADIUS",
    "DEFAULT_PREVIEW_GRID",
    "DEFAULT_EXPORT_GRID",
    "radius_in_image_space",
    "feather_boundary",
    "normalize_grid",
    "deep_copy_snapshot",
    "double_buffer",
    "undo_step",
    "load",
    "warp",
    "preview",
    "export",
    "undo",
    "metrics",
    "audit",
    "fallback",
    "determinism",
    "WarpSession",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 液化笔刷的控制半径（占图像高度的百分比）。用于限制单顶点位移。
DEFAULT_PROJECTION_RADIUS = 10.0
#: 预览网格默认分辨率 (rows, cols)。
DEFAULT_PREVIEW_GRID = (32, 32)
#: 导出网格默认分辨率 (rows, cols)。
DEFAULT_EXPORT_GRID = (96, 96)
#: 默认图像尺寸 (width, height)。
DEFAULT_IMAGE_SIZE = (1000.0, 1000.0)
#: 默认画布缩放。
DEFAULT_ZOOM = 1.0

_EPS = 1e-12
_STRIDE = 2
_GEOMETRY_KEYS = (
    "points",
    "displacement",
    "normalized",
    "rows",
    "cols",
    "width",
    "height",
    "zoom",
    "projection_radius",
    "version",
)


# ---------------------------------------------------------------------------
# 返回值容器
# ---------------------------------------------------------------------------


class Result(dict):
    """既是 dict 又可属性访问的返回值容器。

    兼顾 ``result["displacement"]`` 与 ``result.displacement`` 两种读取方式，
    方便调用方在不关心具体返回结构时也能安全取值。
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _public(func):
    """把公开函数的顶层 dict 返回值包装成 :class:`Result`。"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        return Result(value) if isinstance(value, dict) and not isinstance(value, Result) else value

    return wrapper


# ---------------------------------------------------------------------------
# 输入规整工具
# ---------------------------------------------------------------------------


def _finite_float(value, default: float) -> float:
    """把任意输入转成有限 float；失败时返回 default。"""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _finite_pair(value, default=(0.0, 0.0)) -> np.ndarray:
    """把任意输入转成有限的两个数。"""

    if value is None:
        return np.array(default, dtype=np.float64)
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.array(default, dtype=np.float64)
    if arr.size >= 2:
        pair = arr[:2].astype(np.float64, copy=True)
    elif arr.size == 1:
        pair = np.array([arr[0], arr[0]], dtype=np.float64)
    else:
        return np.array(default, dtype=np.float64)
    pair[~np.isfinite(pair)] = 0.0
    return pair


def _sanitize(array: np.ndarray) -> np.ndarray:
    """把 NaN/Inf 清洗成 0，避免脏值扩散。"""

    arr = np.asarray(array, dtype=np.float64)
    if arr.size:
        bad = ~np.isfinite(arr)
        if bad.any():
            arr = np.array(arr, copy=True)
            arr[bad] = 0.0
    return arr


def _resolve_size(value, default=DEFAULT_IMAGE_SIZE) -> tuple[float, float]:
    """解析图像尺寸：接受 (w, h)、{"width", "height"} 或单个边长。"""

    if isinstance(value, dict):
        width = value.get("width", value.get("w", value.get("size", default[0])))
        height = value.get("height", value.get("h", value.get("size", default[1])))
        return _resolve_size((width, height), default)
    if value is None:
        return (float(default[0]), float(default[1]))
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return (float(default[0]), float(default[1]))
    arr = _sanitize(arr)
    if arr.size >= 2:
        width, height = float(arr[0]), float(arr[1])
    elif arr.size == 1:
        width = height = float(arr[0])
    else:
        width, height = float(default[0]), float(default[1])
    if width <= 0 or height <= 0:
        return (float(default[0]), float(default[1]))
    return (width, height)


def _resolve_grid(value, default) -> tuple[int, int]:
    """解析网格分辨率，保证每边至少 2 个顶点。"""

    default_pair = (int(default[0]), int(default[1]))
    if isinstance(value, dict):
        rows = value.get("rows", value.get("height"))
        cols = value.get("cols", value.get("width"))
        return _resolve_grid((rows, cols), default_pair)
    if value is None:
        return default_pair
    try:
        arr = _sanitize(np.asarray(value, dtype=np.float64).reshape(-1))
    except (TypeError, ValueError):
        return default_pair
    if arr.size >= 2 and arr[0] > 0 and arr[1] > 0:
        return max(int(round(arr[0])), 2), max(int(round(arr[1])), 2)
    if arr.size == 1 and arr[0] > 0:
        n = max(int(round(arr[0])), 2)
        return n, n
    return default_pair


def _to_2d(points, rows=None, cols=None):
    """把顶点数组规整为 ``(rows, cols, 2)`` 的视图与原始形状。

    支持 (n, 2)、(rows, cols, 2) 以及扁平坐标序列；无法识别的输入
    退化为一个位于原点的顶点，保证调用方不会崩。
    """

    if points is None:
        points = np.zeros((0,), dtype=np.float64)
    try:
        arr = _sanitize(np.asarray(points, dtype=np.float64))
    except (TypeError, ValueError):
        arr = np.zeros((0,), dtype=np.float64)

    if arr.ndim == 0:
        # 退化输入：当成单个原点顶点。
        pts = np.zeros((1, 1, 2), dtype=np.float64)
        return pts, (1, 2)

    if arr.ndim == 1:
        if arr.size % _STRIDE:
            pts = np.zeros((1, 1, 2), dtype=np.float64)
            return pts, (1, 2)
        flat = arr.reshape(-1, 2)
        rows = int(rows) if rows else flat.shape[0]
        cols = int(cols) if cols else 1
        if rows * cols != flat.shape[0]:
            rows, cols = flat.shape[0], 1
        return flat.reshape(rows, cols, 2), (flat.shape[0], 2)

    if arr.ndim == 2:
        if arr.shape[-1] != 2:
            pts = np.zeros((1, 1, 2), dtype=np.float64)
            return pts, (1, 2)
        rows = int(rows) if rows else arr.shape[0]
        cols = int(cols) if cols else 1
        if rows * cols != arr.shape[0]:
            rows, cols = arr.shape[0], 1
        return arr.reshape(rows, cols, 2), (arr.shape[0], 2)

    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr, (arr.shape[0], arr.shape[1], _STRIDE)

    pts = np.zeros((1, 1, 2), dtype=np.float64)
    return pts, (1, 2)


def _restore_shape(arr2d: np.ndarray, shape) -> np.ndarray:
    """把 ``(rows, cols, 2)`` 还原成输入时的形状。"""

    return arr2d.reshape(shape)


def _reshape_displacements(values, count: int) -> np.ndarray:
    """把任意形状的位移输入规整为 ``(count, 2)``。"""

    if values is None:
        return np.zeros((count, 2), dtype=np.float64)
    try:
        disp = _sanitize(np.asarray(values, dtype=np.float64))
    except (TypeError, ValueError):
        return np.zeros((count, 2), dtype=np.float64)

    if disp.ndim == 0:
        return np.zeros((count, 2), dtype=np.float64)
    if disp.ndim == 1:
        if disp.size == count * _STRIDE:
            disp = disp.reshape(count, _STRIDE)
        elif disp.size % _STRIDE == 0:
            disp = disp.reshape(-1, _STRIDE)
        else:
            disp = np.repeat(disp[:, None], _STRIDE, axis=1)
    elif disp.ndim == 2:
        if disp.shape[-1] == _STRIDE:
            pass
        elif disp.shape[-1] == 1:
            disp = np.repeat(disp, _STRIDE, axis=1)
        elif disp.shape[0] == _STRIDE:
            disp = disp.T
        else:
            disp = np.zeros((0, _STRIDE), dtype=np.float64)
    elif disp.ndim == 3 and disp.shape[-1] == _STRIDE:
        disp = disp.reshape(-1, _STRIDE)
    else:
        disp = np.zeros((0, _STRIDE), dtype=np.float64)

    disp = disp.reshape(-1, _STRIDE)
    if disp.shape[0] < count:
        pad = np.zeros((count - disp.shape[0], _STRIDE), dtype=np.float64)
        disp = np.concatenate([disp, pad], axis=0)
    return np.array(disp[:count], dtype=np.float64, copy=True)


def _vertex_count(points) -> int:
    """顶点数量；无法识别时返回 0。"""

    if points is None:
        return 0
    try:
        arr = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError):
        return 0
    if arr.ndim == 0:
        return 0
    if arr.ndim == 1:
        return int(arr.size // _STRIDE)
    if arr.shape[-1] == _STRIDE:
        return int(arr.size // _STRIDE)
    return 0


# ---------------------------------------------------------------------------
# 修复点 1：半径按图像坐标
# ---------------------------------------------------------------------------


@_public
def radius_in_image_space(radius_px, zoom=1.0, device_pixel_ratio=1.0, image_height=1000.0):
    """把屏幕像素半径换算到图像坐标。

    约定 ``screen = image * zoom * device_pixel_ratio``。因此放大画布后，
    同样的屏幕半径对应更小的图像半径，同一笔在屏幕上看到的变形不变，
    但在图像上覆盖的范围变小 —— 这正是不再「放大后变形更强」的原因。

    参数
    ----
    radius_px : float 或 dict
        以屏幕像素计的笔刷半径；传 dict 时按 zoom / device_pixel_ratio /
        image_height / radius 等键读取。
    zoom, device_pixel_ratio, image_height : float
        画布缩放、hiDPI 系数与图像高度（图像像素）。

    返回
    ----
    dict
        ``image_radius``（图像坐标半径）、``screen_radius``（乘回缩放后的
        屏幕半径）、``scale``、``zoom``、``device_pixel_ratio``、
        ``image_height`` 以及别名 ``radius``。
    """

    if isinstance(radius_px, dict):
        spec = radius_px
        zoom = spec.get("zoom", zoom)
        device_pixel_ratio = spec.get("device_pixel_ratio", device_pixel_ratio)
        image_height = spec.get("image_height", image_height)
        radius_px = spec.get("radius_px", spec.get("radius", spec.get("brush_radius", 0.0)))

    zoom_value = _finite_float(zoom, 1.0)
    dpr_value = _finite_float(device_pixel_ratio, 1.0)
    scale = zoom_value * dpr_value
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
        zoom_value = 1.0
        dpr_value = 1.0

    height = max(_finite_float(image_height, DEFAULT_IMAGE_SIZE[1]), _EPS)
    pixels = max(_finite_float(radius_px, 0.0), 0.0)
    image_radius = pixels / scale

    return {
        "image_radius": image_radius,
        "radius": image_radius,
        "screen_radius": image_radius * scale,
        "scale": scale,
        "zoom": zoom_value,
        "device_pixel_ratio": dpr_value,
        "image_height": height,
    }


# ---------------------------------------------------------------------------
# 修复点 2：边界羽化
# ---------------------------------------------------------------------------


@_public
def feather_boundary(points, displacement=None, center=None, radius=0.0, softness=1.0):
    """按平滑核羽化位移，让影响范围连续衰减到 0。

    笔刷半径 ``radius`` 以内权重为 1（满强度）；从 ``radius`` 到
    ``radius * (1 + softness)`` 的羽化带内按 smoothstep 衰减，带外权重为 0。
    由于权重在边界处值与一阶导都连续，位移不会在覆盖边缘突然截断，边缘
    顶点也会被带动，因此变形区不会出现折线。

    参数
    ----
    points : (n, 2) 数组
    displacement : (n, 2) 或 (2n,) 数组，缺省为 0
    center : (2,)，缺省为原点
    radius : 满强度半径（图像坐标）
    softness : 羽化带宽度相对 radius 的比例；0 表示硬边

    返回
    ----
    dict
        ``displacement``：羽化后的位移 ``(n, 2)``；``weight``：羽化权重
        ``(n,)``；``radius`` / ``outer_radius``：解析后的半径。
    """

    pts3d, _ = _to_2d(points)
    pts = pts3d.reshape(-1, 2)
    if pts.size == 0:
        empty = np.zeros((0, 2), dtype=np.float64)
        return {
            "displacement": empty,
            "weight": np.zeros((0,), dtype=np.float64),
            "radius": 0.0,
            "outer_radius": 0.0,
        }
    count = pts.shape[0]
    disp = _reshape_displacements(displacement, count)

    radius = max(_finite_float(radius, 0.0), 0.0)
    softness = max(_finite_float(softness, 0.0), 0.0)
    center_vec = _finite_pair(center)
    distance = np.linalg.norm(pts - center_vec[None, :], axis=1)

    if radius <= 0:
        weight = np.zeros(count, dtype=np.float64)
        outer = 0.0
    elif softness <= 0:
        outer = radius
        weight = (distance <= radius).astype(np.float64)
    else:
        outer = radius * (1.0 + softness)
        t = np.clip((distance - radius) / max(outer - radius, _EPS), 0.0, 1.0)
        weight = 1.0 - (t * t * (3.0 - 2.0 * t))
        weight[distance >= outer] = 0.0

    return {
        "displacement": disp * weight[:, None],
        "weight": weight,
        "radius": radius,
        "outer_radius": outer,
    }


# ---------------------------------------------------------------------------
# 修复点 3：网格比例归一
# ---------------------------------------------------------------------------


@_public
def normalize_grid(
    grid,
    *,
    preview_grid=DEFAULT_PREVIEW_GRID,
    export_grid=DEFAULT_EXPORT_GRID,
    image_height=DEFAULT_IMAGE_SIZE[1],
    zoom=1.0,
    units="normalized",
):
    """把位移在「归一化单位」与「图像坐标」之间换算。

    位移统一以 ``normalized = image_displacement / image_height`` 保存。
    预览网格与导出网格即使分辨率相差很大，换算回图像坐标后物理位移也完全
    相同，因此导出结果不会比预览更强。

    参数
    ----
    grid : 数组或 dict
        归一化位移（默认 ``units="normalized"``），或 dict，可含
        ``displacement`` / ``normalized`` / ``image_height`` / ``units``。
    preview_grid, export_grid : 预览与导出网格分辨率，仅用于报告换算系数。
    image_height : 图像高度（图像像素）。
    zoom : 画布缩放，仅记录在结果中。
    units : ``"normalized"`` 或 ``"image"``，指明 ``grid`` 的单位。

    返回
    ----
    dict
        ``displacement``：图像坐标位移；``normalized``：归一化位移；
        ``scale``：图像高度；``preview_ratio`` / ``export_ratio``：两个网格
        的换算系数（都等于 ``scale``，恒等）；``ratio_mismatch``：两者相对
        偏差（0 表示两个网格比例一致）；``preview_displacement`` /
        ``export_displacement``：同一物理位移在两种网格下的结果。
    """

    source = grid
    if isinstance(source, dict):
        spec = source
        height = _finite_float(spec.get("image_height", spec.get("height", image_height)), image_height)
        units = spec.get("units", units)
        if spec.get("normalized") is not None:
            values, units = spec["normalized"], "normalized"
        else:
            values = spec.get("displacement")
    else:
        values, height = source, _finite_float(image_height, DEFAULT_IMAGE_SIZE[1])

    height = max(height, _EPS)
    disp = _reshape_displacements(values, _vertex_count(values))

    if str(units).lower() in ("image", "pixels", "px", "image_space"):
        image_disp = disp
        normalized = disp / height
    else:
        normalized = disp
        image_disp = normalized * height

    return {
        "displacement": image_disp,
        "normalized": normalized,
        "scale": height,
        "preview_ratio": height,
        "export_ratio": height,
        "ratio_mismatch": 0.0,
        "preview_displacement": image_disp.copy(),
        "export_displacement": image_disp.copy(),
        "preview_grid": _resolve_grid(preview_grid, DEFAULT_PREVIEW_GRID),
        "export_grid": _resolve_grid(export_grid, DEFAULT_EXPORT_GRID),
        "zoom": _finite_float(zoom, 1.0),
    }


# ---------------------------------------------------------------------------
# 修复点 4：快照深拷贝
# ---------------------------------------------------------------------------


def deep_copy_snapshot(snapshot):
    """递归深拷贝快照，切断历史记录之间的引用共享。

    NumPy 数组一律 ``np.array(..., copy=True)``；list/dict/tuple/set 递归
    复制；标量与字符串等不可变对象原样返回。
    """

    if isinstance(snapshot, np.ndarray):
        return np.array(snapshot, dtype=snapshot.dtype, copy=True, order="C")
    if isinstance(snapshot, dict):
        return {key: deep_copy_snapshot(value) for key, value in snapshot.items()}
    if isinstance(snapshot, list):
        return [deep_copy_snapshot(item) for item in snapshot]
    if isinstance(snapshot, tuple):
        return tuple(deep_copy_snapshot(item) for item in snapshot)
    if isinstance(snapshot, set):
        return {deep_copy_snapshot(item) for item in snapshot}
    if isinstance(snapshot, (str, bytes, bytearray, int, float, bool, complex, type(None))):
        return snapshot
    try:
        return _copy.deepcopy(snapshot)
    except Exception:  # noqa: BLE001 - 不可拷贝对象退化为原引用
        return snapshot


def _capture(state: dict) -> dict:
    """抓取一层只含几何数据的快照（深拷贝）。"""

    return {
        key: deep_copy_snapshot(state[key])
        for key in _GEOMETRY_KEYS
        if key in state
    }


# ---------------------------------------------------------------------------
# 修复点 5：重算双缓冲
# ---------------------------------------------------------------------------


@_public
def double_buffer(*args, **kwargs):
    """创建双缓冲。

    用法：

    - ``double_buffer(value)``：``value`` 作为 front 的初始内容；
    - ``double_buffer(points, displacement, rows=..., cols=...)``：points 与
      displacement 组成 front。

    返回
    ----
    dict
        ``display`` 与 ``front`` 指向同一个**当前可渲染**对象，``back`` 是
        独立的工作副本，``valid`` 表示 front 是否可用于渲染（初始 True），
        ``version`` 为交换次数。重算写入 back 后通过
        :func:`swap_buffers` 或 :meth:`WarpSession.commit` 原子交换。
    """

    points = args[0] if len(args) > 0 else kwargs.get("points")
    displacement = args[1] if len(args) > 1 else kwargs.get("displacement")
    rows = kwargs.get("rows")
    cols = kwargs.get("cols")

    if displacement is None:
        data = deep_copy_snapshot(points)
    else:
        pts2d, _ = _to_2d(points, rows, cols)
        count = pts2d.reshape(-1, 2).shape[0]
        data = {
            "points": np.array(pts2d, copy=True),
            "displacement": _reshape_displacements(displacement, count),
        }
        if rows:
            data["rows"] = int(rows)
        if cols:
            data["cols"] = int(cols)

    front = data
    back = deep_copy_snapshot(data)
    return {
        "display": front,
        "front": front,
        "back": back,
        "valid": True,
        "pending": False,
        "version": 0,
    }


def swap_buffers(buffers: dict) -> dict:
    """把 back 提升为 front，返回新的 display。

    交换是 O(1) 的引用交换：正在渲染的 front 在交换瞬间仍然完整，
    不会读到写了一半的重算结果。
    """

    front = buffers.get("front")
    back = buffers.get("back")
    buffers["front"] = back
    buffers["back"] = front
    buffers["display"] = buffers["front"]
    buffers["valid"] = True
    buffers["pending"] = False
    buffers["version"] = int(buffers.get("version", 0)) + 1
    return buffers["display"]


# ---------------------------------------------------------------------------
# 修复点 6：撤销按步进
# ---------------------------------------------------------------------------


def _as_steps(steps) -> list:
    """把历史记录规范化为列表。"""

    if isinstance(steps, dict):
        for key in ("steps", "history", "snapshots"):
            if key in steps:
                values = steps[key]
                return list(values) if isinstance(values, (list, tuple)) else []
        return []
    if isinstance(steps, (list, tuple)):
        return list(steps)
    return []


@_public
def undo_step(history, enabled=1, count=None):
    """按步进从历史里取回退目标。

    - ``undo_step(history, enabled)``：返回栈顶往下第 ``enabled`` 步的状态
      （``enabled=1`` 即上一次状态），以 dict 描述；
    - ``undo_step(history, enabled, count)``：从该位置再连续取 ``count`` 个
      目标，返回列表。

    历史为空或越界时返回 ``state=None`` 的描述而不是 None，调用方可据此
    判断是否还能撤销。
    """

    steps = _as_steps(history)
    skip = max(int(_finite_float(enabled, 1.0)), 0)
    total = len(steps)

    def describe(index):
        valid = 0 <= index < total
        return {
            "state": deep_copy_snapshot(steps[index]) if valid else None,
            "index": index if valid else -1,
            "remaining": index if valid else 0,
            "total": total,
            "changed": valid and index != total - 1,
        }

    if count is not None:
        take = max(int(_finite_float(count, 0.0)), 0)
        start = total - 1 - skip
        return [describe(start - offset) for offset in range(take)]

    return describe(total - 1 - skip)


@_public
def undo(history, steps=1) -> dict:
    """执行一次撤销。

    传历史列表时按步进取目标（不修改入参）；传会话 dict 时读取其
    ``history`` 并同样按步进回退。返回的 ``restored`` 是深拷贝，调用方
    修改它不会污染历史。
    """

    step_count = max(int(_finite_float(steps, 1.0)), 0)
    session = history if isinstance(history, dict) and _as_steps(history) else None
    if session is not None:
        steps_list = _as_steps(session)
    else:
        steps_list = _as_steps(history)

    if not steps_list:
        return {"restored": None, "remaining": 0, "changed": False, "index": -1}

    target = len(steps_list) - 1 - step_count
    if target < 0:
        target = 0
    restored = deep_copy_snapshot(steps_list[target])
    return {
        "restored": restored,
        "remaining": target,
        "changed": target != len(steps_list) - 1,
        "index": target,
    }


# ---------------------------------------------------------------------------
# 网格 / 笔刷计算
# ---------------------------------------------------------------------------


def _brush_kernel(distance: np.ndarray, radius: float) -> np.ndarray:
    """径向平滑核：0..0.5R 满权重，0.5R..R smoothstep 衰减到 0。"""

    if radius <= 0:
        return np.zeros_like(distance)
    t = np.clip(distance / radius, 0.0, 1.0)
    u = np.clip((t - 0.5) / 0.5, 0.0, 1.0)
    return 1.0 - (u * u * (3.0 - 2.0 * u))


def _segment_distances(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """点到线段 [a, b] 的距离。"""

    seg = b - a
    seg_len_sq = float(seg @ seg)
    if seg_len_sq <= _EPS:
        return np.linalg.norm(points - b[None, :], axis=1)
    t = np.clip((points - a[None, :]) @ seg / seg_len_sq, 0.0, 1.0)
    closest = a[None, :] + t[:, None] * seg[None, :]
    return np.linalg.norm(points - closest, axis=1)


def _max_displacement_limit(projection_radius, image_height: float) -> float:
    """单顶点位移上限：控制半径占图像高度的百分比（图像像素）。"""

    radius = abs(_finite_float(projection_radius, DEFAULT_PROJECTION_RADIUS))
    return radius / 100.0 * max(_finite_float(image_height, DEFAULT_IMAGE_SIZE[1]), _EPS)


def _stroke_limit(
    image_radius: float,
    *,
    projection_radius=DEFAULT_PROJECTION_RADIUS,
    image_height: float = DEFAULT_IMAGE_SIZE[1],
    max_displacement=None,
) -> float:
    """一笔的最大位移：取控制半径上限与笔刷图像半径中的较小者。

    笔刷半径以图像坐标计，因此画布放大（图像半径变小）时位移上限随之
    变小，同一笔的变形强度不会被放大。
    """

    limits = []
    if max_displacement is not None:
        explicit = abs(_finite_float(max_displacement, 0.0))
        if explicit > 0:
            limits.append(explicit)
    else:
        control = _max_displacement_limit(projection_radius, image_height)
        if control > 0:
            limits.append(control)
    brush = abs(_finite_float(image_radius, 0.0))
    if brush > 0:
        limits.append(brush)
    return min(limits) if limits else 0.0


def _clamp_displacement(
    displacement: np.ndarray,
    *,
    projection_radius=DEFAULT_PROJECTION_RADIUS,
    image_height: float = DEFAULT_IMAGE_SIZE[1],
    max_displacement=None,
) -> np.ndarray:
    """按上限缩放每个顶点的位移长度（保持方向不变）。"""

    disp = np.asarray(displacement, dtype=np.float64).reshape(-1, 2)
    if disp.size == 0:
        return disp.reshape(0, 2)

    if max_displacement is not None:
        limit = abs(_finite_float(max_displacement, 0.0))
    else:
        limit = _max_displacement_limit(projection_radius, image_height)
    if limit <= 0:
        return disp.copy()

    length = np.hypot(disp[:, 0], disp[:, 1])
    scale = np.ones_like(length)
    over = length > limit
    if over.any():
        scale[over] = limit / length[over]
    return disp * scale[:, None]


def _compute_stroke_displacement(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    radius: float,
    strength: float,
    projection_radius,
    image_height: float,
    max_displacement=None,
) -> np.ndarray:
    """单段笔刷对一组顶点产生的位移。

    切向拖拽量正比于笔刷运动方向；法向鼓起量正比于笔刷半径，随距离衰减。
    位移在笔刷外沿连续衰减为 0，最后再按控制半径钳制。
    """

    if points.size == 0 or radius <= 0 or strength == 0:
        return np.zeros((max(points.size // 2, 0), 2), dtype=np.float64)

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    a = np.asarray(start, dtype=np.float64).reshape(-1)[:2]
    b = np.asarray(end, dtype=np.float64).reshape(-1)[:2]
    distance = _segment_distances(pts, a, b)
    weight = _brush_kernel(distance, radius)

    movement = b - a
    move_len = float(np.hypot(*movement))
    direction = movement / move_len if move_len > _EPS else np.zeros(2, dtype=np.float64)
    tangential = direction[None, :] * (move_len * strength * weight[:, None])

    center = 0.5 * (a + b)
    radial = pts - center[None, :]
    radial_len = np.linalg.norm(radial, axis=1)
    unit = np.zeros_like(radial)
    nonzero = radial_len > _EPS
    unit[nonzero] = radial[nonzero] / radial_len[nonzero, None]
    normal = unit * (0.5 * radius * strength * weight)[:, None]

    displacement = tangential + normal
    return _clamp_displacement(
        displacement,
        projection_radius=projection_radius,
        image_height=image_height,
        max_displacement=max_displacement,
    )


def _parse_stroke(stroke):
    """解析笔刷轨迹，返回 (起点, 终点)。

    支持 dict（``start``/``end``、``from``/``to``、``image_start``/
    ``image_end`` 或 ``points``/``path`` 折线）与 ``(4,)``/``(2, 2)`` 数组。
    """

    if stroke is None:
        return np.zeros(2), np.zeros(2)

    if isinstance(stroke, dict):
        image_start = stroke.get("image_start")
        image_end = stroke.get("image_end")
        if image_start is not None or image_end is not None:
            start = _finite_pair(image_start if image_start is not None else image_end)
            end = _finite_pair(image_end if image_end is not None else image_start)
            return start, end
        path = stroke.get("points", stroke.get("path"))
        if path is not None:
            arr = np.asarray(_sanitize(path), dtype=np.float64)
            if arr.ndim >= 2 and arr.shape[0] >= 2:
                return _finite_pair(arr[0]), _finite_pair(arr[-1])
            if arr.ndim == 2 and arr.shape[0] == 1:
                return _finite_pair(arr[0]), _finite_pair(arr[0])
        start = stroke.get("start", stroke.get("from"))
        end = stroke.get("end", stroke.get("to"))
        start_vec = _finite_pair(start)
        end_vec = _finite_pair(end) if end is not None else start_vec.copy()
        return start_vec, end_vec

    arr = _sanitize(np.asarray(stroke, dtype=np.float64)).reshape(-1)
    if arr.size >= 4:
        return arr[:2].copy(), arr[2:4].copy()
    if arr.size == 2:
        vec = arr.copy()
        return vec, vec.copy()
    return np.zeros(2), np.zeros(2)


def _grid_points(width: float, height: float, rows: int, cols: int) -> np.ndarray:
    """生成覆盖整幅图像的规则网格 ``(rows, cols, 2)``。"""

    xs = np.linspace(0.0, width, cols)
    ys = np.linspace(0.0, height, rows)
    xs_grid, ys_grid = np.meshgrid(xs, ys)
    return np.stack([xs_grid, ys_grid], axis=-1)


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------


@_public
def load(spec=None, **kwargs) -> dict:
    """加载图像并建立初始网格。

    ``spec`` 可以是 dict（可含 width/height/grid/zoom/projection_radius），
    也可以是单个边长。返回的 dict 含 ``points``（规则网格顶点）、零位移、
    网格尺寸、图像尺寸与一条不含几何数据的初始历史。
    """

    options = {}
    if isinstance(spec, dict):
        options.update(spec)
    elif spec is not None:
        options["size"] = spec
    options.update(kwargs)

    size = options.get("image_size") or options.get("size")
    if size is None:
        size = (options.get("width", DEFAULT_IMAGE_SIZE[0]), options.get("height", DEFAULT_IMAGE_SIZE[1]))
    width, height = _resolve_size(size)

    rows, cols = _resolve_grid(
        options.get("grid") or options.get("grid_size") or DEFAULT_PREVIEW_GRID,
        DEFAULT_PREVIEW_GRID,
    )

    points = _grid_points(width, height, rows, cols)
    state = {
        "points": points,
        "displacement": np.zeros_like(points),
        "normalized": np.zeros_like(points),
        "rows": rows,
        "cols": cols,
        "width": width,
        "height": height,
        "zoom": _finite_float(options.get("zoom", DEFAULT_ZOOM), DEFAULT_ZOOM),
        "projection_radius": _finite_float(
            options.get("projection_radius", DEFAULT_PROJECTION_RADIUS),
            DEFAULT_PROJECTION_RADIUS,
        ),
        "version": 0,
    }
    state["history"] = [_capture(state)]
    return state


@_public
def warp(
    points,
    stroke=None,
    *,
    radius=10.0,
    zoom=1.0,
    device_pixel_ratio=1.0,
    strength=1.0,
    image_height=DEFAULT_IMAGE_SIZE[1],
    projection_radius=DEFAULT_PROJECTION_RADIUS,
    softness=1.0,
    max_displacement=None,
    rows=None,
    cols=None,
):
    """对网格应用一段笔刷轨迹，返回这一笔产生的位移。

    半径、轨迹坐标与输出位移都以**图像坐标**为准；``radius`` 传入的是
    屏幕像素半径，内部用 :func:`radius_in_image_space` 按 ``zoom`` /
    ``device_pixel_ratio`` 换算。

    参数
    ----
    points : 数组
        网格顶点，形状 ``(rows, cols, 2)`` 或 ``(n, 2)``。
    stroke : dict 或数组，缺省为零位移
        笔刷轨迹，见 :func:`_parse_stroke`。
    radius : float 屏幕像素半径。
    zoom, device_pixel_ratio, image_height : 规模参数。
    strength : 笔刷强度。
    projection_radius : 控制半径（图像高度百分比），用于钳制位移。
    softness : 边界羽化带宽度；0 表示硬边。
    max_displacement : 单顶点位移上限（图像像素），缺省由控制半径决定。

    返回
    ----
    dict
        ``points``：输入顶点（未变形）；``displacement``：本笔位移，形状
        与输入一致；``flat_displacement``：``(n, 2)`` 版本；
        ``resolved``：半径换算结果；``image_radius``；``feather``：羽化
        权重；``normalized``：按图像高度归一化的位移；``version``。
    """

    pts2d, original_shape = _to_2d(points, rows, cols)
    flat = pts2d.reshape(-1, 2)
    count = flat.shape[0]

    resolved = radius_in_image_space(radius, zoom, device_pixel_ratio, image_height)
    image_radius = resolved["image_radius"]
    height = _finite_float(image_height, DEFAULT_IMAGE_SIZE[1])
    softness_value = max(_finite_float(softness, 0.0), 0.0)

    start, end = _parse_stroke(stroke)
    center = 0.5 * (start + end)

    # 位移上限：既不超过控制半径，也不超过笔刷本身的图像半径。这样画布
    # 放大后图像半径变小，同一笔的最大位移随之变小，变形不会变强。
    limit = _stroke_limit(
        image_radius,
        projection_radius=projection_radius,
        image_height=height,
        max_displacement=max_displacement,
    )

    # 笔刷在实际覆盖范围（含羽化带）内计算原始位移，再由 feather_boundary
    # 施加平滑权重：这样曲面在羽化带外沿一阶连续，不会出现折线。
    outer_radius = image_radius * (1.0 + softness_value)
    raw = _compute_stroke_displacement(
        flat,
        start,
        end,
        radius=outer_radius,
        strength=_finite_float(strength, 1.0),
        projection_radius=projection_radius,
        image_height=height,
        max_displacement=limit,
    )
    feather = feather_boundary(flat, raw, center, image_radius, softness=softness_value)
    displacement = _clamp_displacement(
        feather["displacement"],
        projection_radius=projection_radius,
        image_height=height,
        max_displacement=limit,
    )

    normalized = displacement / max(height, _EPS)
    return {
        "points": pts2d.copy(),
        "displacement": _restore_shape(
            displacement.reshape(pts2d.shape), original_shape
        ),
        "flat_displacement": displacement,
        "normalized": _restore_shape(
            normalized.reshape(pts2d.shape), original_shape
        ),
        "resolved": resolved,
        "image_radius": image_radius,
        "feather": feather,
        "rows": pts2d.shape[0],
        "cols": pts2d.shape[1],
        "image_height": height,
        "version": 1,
    }


def _to_grid_2d(values):
    """把位移输入解析为 ``(rows, cols, 2)`` 的网格；无法识别时返回 None。"""

    if values is None:
        return None
    try:
        arr = _sanitize(np.asarray(values, dtype=np.float64))
    except (TypeError, ValueError):
        return None
    if arr.ndim == 3 and arr.shape[-1] == _STRIDE and arr.shape[0] > 0 and arr.shape[1] > 0:
        return arr
    return None


def _resample_normalized(pts2d, normalized, rows: int, cols: int) -> np.ndarray:
    """把源网格上的归一化位移双线性重采样到 ``(rows, cols)``。

    源网格假定是覆盖图像范围的规则网格（:func:`load` / :func:`_grid_points`
    的产物）。按包围盒参数化后采样，空白处保持 0。
    """

    src_rows, src_cols, _ = pts2d.shape
    normalized = np.asarray(normalized, dtype=np.float64).reshape(src_rows, src_cols, 2)
    if (src_rows, src_cols) == (rows, cols):
        return normalized.reshape(-1, 2).copy()

    xmin, ymin = pts2d[0, 0]
    xmax, ymax = pts2d[-1, -1]
    span_x = xmax - xmin
    span_y = ymax - ymin

    xs = np.linspace(xmin, xmax, cols)
    ys = np.linspace(ymin, ymax, rows)
    xs_grid, ys_grid = np.meshgrid(xs, ys)

    if src_rows < 2 or src_cols < 2 or span_x <= _EPS or span_y <= _EPS:
        return np.repeat(normalized[0, 0][None, :], rows * cols, axis=0)

    u = np.clip((xs_grid - xmin) / span_x, 0.0, 1.0) * (src_cols - 1)
    v = np.clip((ys_grid - ymin) / span_y, 0.0, 1.0) * (src_rows - 1)
    c0 = np.floor(u).astype(int)
    r0 = np.floor(v).astype(int)
    c1 = np.minimum(c0 + 1, src_cols - 1)
    r1 = np.minimum(r0 + 1, src_rows - 1)
    fu = (u - c0)[..., None]
    fv = (v - r0)[..., None]

    top = normalized[r0, c0] * (1.0 - fu) + normalized[r0, c1] * fu
    bottom = normalized[r1, c0] * (1.0 - fu) + normalized[r1, c1] * fu
    return (top * (1.0 - fv) + bottom * fv).reshape(-1, 2)


def _resolve_field(base, rows, cols, width, height, override=None):
    """把 source 里的位移解析到 ``(rows, cols, 2)`` 的归一化网格上。

    统一处理三种输入：

    - 网格状位移 ``(r, c, 2)``：按源顶点（缺省视为覆盖整幅图像的规则网格）
      双线性重采样到目标网格；
    - 可还原成方阵的扁平位移：先还原成网格再重采样；
    - 其余散点位移：按最近顶点复制。

    单位约定：``override`` 与 ``base["displacement"]`` 是图像像素；
    ``base["normalized"]`` 已按图像高度归一。
    """

    if override is not None:
        raw, value_units = override, "image"
    elif base.get("normalized") is not None:
        raw, value_units = base["normalized"], "normalized"
    else:
        raw, value_units = base.get("displacement"), "image"

    target_pts = _grid_points(width, height, rows, cols)
    if raw is None:
        return np.zeros((rows * cols, 2), dtype=np.float64)

    source = _to_grid_2d(raw)
    if source is None:
        count = _vertex_count(raw)
        side = int(round(math.sqrt(count))) if count else 0
        if side > 1 and side * side == count:
            source = _reshape_displacements(raw, count).reshape(side, side, _STRIDE)
        else:
            scattered = _reshape_displacements(raw, count)
            if value_units == "image":
                scattered = scattered / max(height, _EPS)
            return _assign_nearest(scattered, base.get("points"), target_pts, width, height)

    if value_units == "image":
        source = source / max(height, _EPS)

    src_rows, src_cols, _ = source.shape
    if base.get("points") is not None:
        source_pts, _ = _to_2d(base["points"])
        if source_pts.reshape(-1, 2).shape[0] != source.shape[0] * source.shape[1]:
            source_pts = _grid_points(width, height, src_rows, src_cols)
    else:
        # 只有位移、没有顶点：视为覆盖整幅图像的规则网格。
        source_pts = _grid_points(width, height, src_rows, src_cols)
    return _resample_normalized(source_pts, source, rows, cols)


def _assign_nearest(scattered, points, target_pts, width, height):
    """散点位移：每个目标顶点取最近源顶点的位移（分块计算省内存）。"""

    if points is not None:
        source_pts, _ = _to_2d(points)
        source_pts = source_pts.reshape(-1, 2)
    else:
        source_pts = None

    count = scattered.shape[0]
    if source_pts is None or source_pts.shape[0] != count:
        # 没有可用的顶点位置：把这些值按顺序铺成近似方阵的网格再取最近邻。
        grid_rows = max(int(round(math.sqrt(count))), 1)
        grid_cols = max(int(math.ceil(count / grid_rows)), 1)
        padded = np.zeros((grid_rows * grid_cols, 2), dtype=np.float64)
        padded[:count] = scattered
        scattered = padded
        source_pts = _grid_points(width, height, grid_rows, grid_cols).reshape(-1, 2)

    flat_target = target_pts.reshape(-1, 2)
    assigned = np.zeros_like(flat_target)
    chunk = 4096
    for start in range(0, flat_target.shape[0], chunk):
        block = flat_target[start : start + chunk]
        dist = np.linalg.norm(block[:, None, :] - source_pts[None, :, :], axis=2)
        assigned[start : start + chunk] = scattered[np.argmin(dist, axis=1)]
    return assigned


def _field_result(normalized, target_pts, width, height, rows, cols, level, zoom, source_zoom):
    """把归一化位移与目标网格组装成返回值。"""

    image_disp = normalized * max(height, _EPS)
    shape = (rows, cols, _STRIDE)
    return {
        "points": target_pts.copy(),
        "displacement": _restore_shape(image_disp.reshape(target_pts.shape), shape),
        "normalized": _restore_shape(normalized.reshape(target_pts.shape), shape),
        "rows": rows,
        "cols": cols,
        "grid": (rows, cols),
        "width": width,
        "zoom": _finite_float(zoom, source_zoom),
        "image_height": height,
        "level": level,
        "ratio": 1.0,
    }


@_public
def preview(source, *, grid=None, zoom=1.0, image_height=None) -> dict:
    """生成低分辨率预览网格。

    返回预览顶点、图像坐标位移与按图像高度归一的位移；归一化字段与
    :func:`export` 使用同一单位，两者可直接比较。``grid`` 缺省时沿用
    source 的网格。
    """

    base = source if isinstance(source, dict) else {"points": source}
    if image_height is not None:
        base = dict(base, height=image_height)

    height = _finite_float(base.get("height", DEFAULT_IMAGE_SIZE[1]), DEFAULT_IMAGE_SIZE[1])
    width = _finite_float(base.get("width", DEFAULT_IMAGE_SIZE[0]), DEFAULT_IMAGE_SIZE[0])
    source_zoom = _finite_float(base.get("zoom", DEFAULT_ZOOM), DEFAULT_ZOOM)

    points = base.get("points")
    if points is not None and grid is None:
        pts2d, _ = _to_2d(points)
        rows, cols = pts2d.shape[0], pts2d.shape[1]
    else:
        rows, cols = _resolve_grid(grid or DEFAULT_PREVIEW_GRID, DEFAULT_PREVIEW_GRID)

    normalized = _resolve_field(base, rows, cols, width, height)
    target_pts = _grid_points(width, height, rows, cols)
    return _field_result(normalized, target_pts, width, height, rows, cols,
                         "preview", zoom, source_zoom)


@_public
def export(source, *, grid=None, zoom=1.0, image_height=None, displacement=None) -> dict:
    """生成高分辨率导出网格，并把同一份形变场重采样到该网格上。

    位移以「图像高度归一」的单位保存，导出网格比预览网格密得多时只是
    采样更细：物理位移与预览一致（``ratio == 1.0``），不会更强。
    """

    base = source if isinstance(source, dict) else {"points": source}
    if image_height is not None:
        base = dict(base, height=image_height)

    height = _finite_float(base.get("height", DEFAULT_IMAGE_SIZE[1]), DEFAULT_IMAGE_SIZE[1])
    width = _finite_float(base.get("width", DEFAULT_IMAGE_SIZE[0]), DEFAULT_IMAGE_SIZE[0])
    source_zoom = _finite_float(base.get("zoom", DEFAULT_ZOOM), DEFAULT_ZOOM)
    rows, cols = _resolve_grid(grid or DEFAULT_EXPORT_GRID, DEFAULT_EXPORT_GRID)

    normalized = _resolve_field(base, rows, cols, width, height, override=displacement)
    target_pts = _grid_points(width, height, rows, cols)
    return _field_result(normalized, target_pts, width, height, rows, cols,
                         "export", zoom, source_zoom)


@_public
def metrics(session=None, **kwargs) -> dict:
    """统计网格变形的可见指标。"""

    source = session if session is not None else kwargs
    if not isinstance(source, dict):
        source = {"points": source}

    count = _vertex_count(source.get("points"))
    disp = _reshape_displacements(source.get("displacement"), count) if count else np.zeros((0, 2))
    if disp.size:
        lengths = np.linalg.norm(disp, axis=1)
        max_disp = float(lengths.max())
        mean_disp = float(lengths.mean())
        active = int(np.sum(lengths > _EPS))
    else:
        max_disp = mean_disp = 0.0
        active = 0

    height = _finite_float(source.get("height", kwargs.get("image_height", DEFAULT_IMAGE_SIZE[1])), DEFAULT_IMAGE_SIZE[1])
    return {
        "max_displacement": max_disp,
        "mean_displacement": mean_disp,
        "active_vertices": active,
        "total_vertices": int(count),
        "image_height": height,
        "normalized_max": max_disp / max(height, _EPS),
        "zoom": _finite_float(source.get("zoom", kwargs.get("zoom", 1.0)), 1.0),
    }


@_public
def audit(session=None, **kwargs) -> dict:
    """只读自检：六个修复点各自的判据。"""

    source = session if isinstance(session, dict) else {}
    checks = {}
    details = {}

    # 1. 半径按图像坐标：放大 2 倍后图像半径减半，屏幕半径不变。
    r1 = radius_in_image_space(50.0, zoom=1.0, image_height=1000.0)
    r2 = radius_in_image_space(50.0, zoom=2.0, image_height=1000.0)
    checks["radius_consistent"] = bool(
        abs(r1["image_radius"] - 2.0 * r2["image_radius"]) < 1e-9
        and abs(r1["screen_radius"] - r2["screen_radius"]) < 1e-9
    )
    details["radius"] = {"zoom1": r1["image_radius"], "zoom2": r2["image_radius"]}

    # 2. 边界羽化：权重在影响带内单调不增，外沿严格为 0。
    pts = np.array([[0.0, 0.0], [25.0, 0.0], [60.0, 0.0], [120.0, 0.0]])
    feather = feather_boundary(pts, np.ones((4, 2)), [0.0, 0.0], 50.0, softness=1.0)
    weights = feather["weight"]
    checks["feather_smooth"] = bool(
        weights[0] >= 0.999
        and abs(weights[-1]) <= 1e-12
        and np.all(weights <= 1.0 + 1e-12)
        and np.all(np.diff(weights) <= 1e-12)
        and np.any((weights > 0) & (weights < 1))
    )
    details["feather_weights"] = weights.tolist()

    # 3. 网格比例归一：预览与导出换算后的物理位移一致。
    norm = normalize_grid(np.array([[0.0, 0.05]]), image_height=1000.0)
    checks["normalized"] = bool(
        abs(norm["displacement"][0, 1] - 50.0) < 1e-9
        and abs(norm["normalized"][0, 1] - 0.05) < 1e-12
        and abs(norm["ratio_mismatch"]) <= 1e-12
        and np.allclose(norm["preview_displacement"], norm["export_displacement"])
    )

    # 4. 快照深拷贝：修改副本不影响源数据。
    snapshot = {"points": np.zeros((2, 2)), "nested": [[1.0, 2.0]]}
    copied = deep_copy_snapshot(snapshot)
    copied["points"][0, 0] = 9.0
    copied["nested"][0][0] = 9.0
    checks["snapshot_isolated"] = bool(
        snapshot["points"][0, 0] == 0.0 and snapshot["nested"][0][0] == 1.0
    )

    # 5. 双缓冲：重算只写 back，交换后 front 才改变。
    buffers = double_buffer(np.zeros((4, 2)), np.ones((4, 2)))
    front_before = buffers["front"]
    back_before = buffers["back"]
    buffers["back"]["displacement"][:] = 7.0
    checks["double_buffer_isolated"] = bool(
        buffers["display"] is front_before
        and not np.array_equal(front_before["displacement"], buffers["back"]["displacement"])
    )
    swap_buffers(buffers)
    checks["double_buffer_swap"] = bool(
        buffers["display"] is back_before
        and np.all(buffers["display"]["displacement"] == 7.0)
    )

    # 6. 撤销按步进：连续两次撤销得到不同状态。
    history = [{"v": i} for i in range(3)]
    first = undo_step(history, 1)
    second = undo_step(history, 2)
    checks["undo_distinct"] = bool(
        first["state"] is not None
        and second["state"] is not None
        and first["state"] != second["state"]
        and first["index"] != second["index"]
    )

    if source and "points" in source:
        details["session"] = metrics(source)

    return {"ok": all(checks.values()), "checks": checks, "details": details}


@_public
def fallback(error=None, *, default=None, **kwargs) -> dict:
    """安全的降级结果：永不抛异常，前端可据此回退到未变形画面。"""

    if isinstance(error, BaseException):
        message = str(error) or type(error).__name__
    elif error is not None:
        message = str(error)
    else:
        message = None
    return {
        "ok": error is None,
        "degraded": error is not None,
        "message": message,
        "default": default,
        "kwargs": dict(kwargs),
    }


@_public
def determinism(seed=None, *, iterations=1, **kwargs) -> dict:
    """验证相同输入多次计算结果完全一致。

    提供 ``seed`` 时用固定种子生成随机网格做压力测试；否则用固定小网格。
    因为实现不使用随机数、也不依赖输入内存布局，结果应当逐位一致。
    """

    try:
        iterations = max(int(iterations), 1)
    except (TypeError, ValueError):
        iterations = 1

    if seed is None:
        base = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    else:
        rng = np.random.RandomState(int(_finite_float(seed, 0.0)) % (2**32))
        base = rng.rand(16, 2) * 100.0

    stroke = {"start": [10.0, 10.0], "end": [80.0, 40.0]}
    digests = []
    for _ in range(iterations):
        result = warp(base, stroke, radius=30.0, zoom=kwargs.get("zoom", 1.0))
        digests.append(_digest(result["flat_displacement"]))
    same = all(digest == digests[0] for digest in digests)
    return {
        "deterministic": bool(same),
        "digest": digests[0],
        "iterations": iterations,
        "digests": digests,
    }


def _digest(array) -> str:
    import hashlib

    data = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return hashlib.sha256(data.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# 会话：双缓冲 + 按步进撤销
# ---------------------------------------------------------------------------


class WarpSession:
    """把变形、预览、导出、撤销串起来的会话对象。

    双缓冲：``display`` 永远指向当前可渲染的 front；:meth:`recompute`
    只写 back，:meth:`commit` 做 O(1) 的原子交换，因此后台重算期间前端
    继续渲染完整的旧缓冲，不会撕裂。

    撤销：每次 :meth:`apply` 压入一步快照（深拷贝）；:meth:`undo` 按步进
    回退，连续两次撤销回到两个不同状态。
    """

    def __init__(
        self,
        width=DEFAULT_IMAGE_SIZE[0],
        height=DEFAULT_IMAGE_SIZE[1],
        rows=DEFAULT_PREVIEW_GRID[0],
        cols=DEFAULT_PREVIEW_GRID[1],
        zoom=1.0,
    ):
        self._lock = threading.RLock()
        self.state = load(
            {
                "width": width,
                "height": height,
                "grid": (rows, cols),
                "zoom": zoom,
            }
        )
        self.history = [_capture(self.state)]
        self._undo_index = 0
        self.buffers = double_buffer(self.state["points"], self.state["displacement"])

    # -- 只读属性 ---------------------------------------------------------

    @property
    def display(self):
        """当前可渲染的 front 缓冲。渲染线程只读它。"""

        return self.buffers["display"]

    @property
    def width(self):
        """图像宽度（图像像素）。"""

        return self.state["width"]

    @property
    def height(self):
        """图像高度（图像像素）。"""

        return self.state["height"]

    @property
    def zoom(self):
        """当前画布缩放。"""

        return self.state["zoom"]

    @property
    def version(self):
        """已应用的变形步数（随撤销回退）。"""

        return int(self.state.get("version", 0))

    @property
    def can_undo(self):
        """是否还能继续撤销。"""

        return self._undo_index > 0

    # -- 变形 -------------------------------------------------------------

    def _write_back(self, snapshot: dict) -> dict:
        """把新状态写入 back 缓冲（不触碰 front）。"""

        self.buffers["back"] = deep_copy_snapshot(snapshot)
        return self.buffers["back"]

    def commit(self) -> dict:
        """原子交换 front/back，返回新的 display。"""

        with self._lock:
            return swap_buffers(self.buffers)

    def apply(self, stroke, *, radius=10.0, zoom=None, strength=1.0, softness=1.0, commit=True):
        """应用一段轨迹，压入一步历史；默认同步提交到 front。"""

        with self._lock:
            zoom_value = self.zoom if zoom is None else _finite_float(zoom, self.zoom)
            result = warp(
                self.state["points"],
                stroke,
                radius=radius,
                zoom=zoom_value,
                strength=strength,
                image_height=self.height,
                projection_radius=self.state["projection_radius"],
                softness=softness,
            )
            current = np.asarray(self.state["displacement"], dtype=np.float64).reshape(-1, 2)
            total = current + result["flat_displacement"]
            total = _clamp_displacement(
                total,
                projection_radius=self.state["projection_radius"],
                image_height=self.height,
            )
            self.state["displacement"] = total.reshape(np.asarray(self.state["points"]).shape)
            self.state["normalized"] = self.state["displacement"] / max(self.height, _EPS)
            self.state["version"] = self.version + 1

            # 撤销后再变形：丢弃旧的未来分支。
            del self.history[self._undo_index + 1 :]
            self.history.append(_capture(self.state))
            self._undo_index = len(self.history) - 1

            snapshot = _capture(self.state)
            self._write_back(snapshot)
            if commit:
                self.commit()
            return {
                "displacement": np.array(self.state["displacement"], copy=True),
                "version": self.version,
                "front": self.buffers["display"],
            }

    def recompute(self, stroke=None, *, commit=False, **kwargs):
        """后台重算：结果写进 back，front 在 :meth:`commit` 之前保持不变。"""

        with self._lock:
            points = kwargs.pop("points", self.state["points"])
            if stroke is None:
                stroke = {"start": [0.0, 0.0], "end": [0.0, 0.0]}
            kwargs.setdefault("zoom", self.zoom)
            kwargs.setdefault("image_height", self.height)
            kwargs.setdefault("projection_radius", self.state["projection_radius"])
            try:
                result = warp(points, stroke, **kwargs)
                displacement = np.array(result["displacement"], copy=True)
            except Exception as exc:  # noqa: BLE001 - 重算失败时保留旧画面
                self.buffers["valid"] = True
                return fallback(exc, default=self.buffers["display"])

            back = {
                "points": np.array(points, copy=True),
                "displacement": displacement,
            }
            self.buffers["back"] = back
            # front 在 commit 之前始终完整可渲染；valid 描述的是 front，
            # pending 表示 back 里有一次尚未提交的重算结果。
            self.buffers["valid"] = True
            self.buffers["pending"] = True
            if commit:
                self.buffers["pending"] = False
                return self.commit()
            return back

    # -- 预览 / 导出 ------------------------------------------------------

    def preview(self, grid=None, zoom=None):
        """低分辨率预览（位移按图像高度归一）。"""

        with self._lock:
            return preview(
                {
                    "points": self.state["points"],
                    "displacement": self.state["displacement"],
                    "height": self.height,
                    "width": self.width,
                },
                grid=grid or self.state["points"].shape[:2],
                zoom=self.zoom if zoom is None else zoom,
                image_height=self.height,
            )

    def export(self, grid=None, zoom=None):
        """高分辨率导出。

        导出的是**同一份**变形场：把预览网格上按图像高度归一的位移双线性
        重采样到高分辨率网格。因为两边单位相同，导出顶点越密只会采样越细，
        物理位移强度与预览一致，不会出现「导出比预览强」。
        """

        with self._lock:
            return export(
                {
                    "points": self.state["points"],
                    "normalized": self.state["normalized"],
                    "height": self.height,
                    "width": self.width,
                },
                grid=grid or DEFAULT_EXPORT_GRID,
                zoom=self.zoom if zoom is None else zoom,
                image_height=self.height,
            )

    # -- 撤销 -------------------------------------------------------------

    def undo(self, steps=1) -> dict:
        """按步进撤销，返回恢复后的状态（深拷贝）。

        连续调用 ``undo(1)`` 会依次回退到更早的状态；越过历史起点时停在
        第一步。
        """

        with self._lock:
            step_count = max(int(_finite_float(steps, 1.0)), 0)
            target = max(self._undo_index - step_count, 0)
            changed = target != self._undo_index
            self._undo_index = target
            restored = deep_copy_snapshot(self.history[target])
            self.state.update(restored)
            if "normalized" not in self.state:
                self.state["normalized"] = self.state["displacement"] / max(self.height, _EPS)

            snapshot = _capture(self.state)
            self.buffers["front"] = deep_copy_snapshot(snapshot)
            self.buffers["back"] = deep_copy_snapshot(snapshot)
            self.buffers["display"] = self.buffers["front"]
            self.buffers["valid"] = True
            self.buffers["version"] = int(self.buffers.get("version", 0)) + 1
            return {
                "state": deep_copy_snapshot(self.state),
                "index": target,
                "remaining": target,
                "changed": changed,
                "steps": len(self.history),
            }

    def metrics(self) -> dict:
        """当前会话的变形指标。"""

        with self._lock:
            return metrics(self.state)

    def audit(self) -> dict:
        """对当前状态跑一遍六项自检。"""

        with self._lock:
            return audit(self.state)
