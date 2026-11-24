# pairwise-liquify-mesh-warp

## 领域

游戏、图形与多媒体

## 主题

液化网格变形

## 说明

图像坐标半径、边界羽化与双缓冲。

初始状态下 `imaging/liquify_mesh_warp.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
