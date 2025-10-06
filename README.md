# 给模型提供者的 Hearts 游戏接入说明

这个仓库包含一个简单的 Hearts(黑桃)纸牌游戏实现, 和一个最小的 policy(策略)接口, 方便你把外部 AI 模型或启发式策略接入为玩家. 

本文档说明仓库结构、policy 的 '契约'(模型会接收什么、需要返回什么)、示例, 以及如何在本地运行和验证. 

## 仓库结构

- `game.py` — 核心游戏逻辑与环境, 定义了 `Card`、`Player`、`GameState`、游戏循环以及辅助函数. 
- `Zhuiy_samplepolicy.py` — 一个最小示例策略, 展示了期望的函数签名. 
- `data/` —(可选)可以放数据集或其他资源的文件夹. 

## Policy 接口

在 `game()` 中会传入一个 `policies` 列表, 每个元素都是一个可调用的策略函数. 

当前 `game.py` 使用的签名为：

```py
def policy(player: Player, player_info: dict, actions: List[Card], order: int) -> Card
```

参数说明：
- `player`(`game.Player`): 对应当前玩家的 Player 对象, 包含 `hand`(手牌)、`points`(分数)、`table`(玩家的出牌记录). 这是运行时的对象, 供本地策略快速访问. 
- `player_info`(`dict`): 为模型准备的“干净视图”, 包含键：`hand`, `points`, `table`, `round`(在 `game.py` 的 `player_info()` 中构造). 当你把状态发送到远程模型或不希望模型直接操作 `Player` 对象时, 应使用此字段. 
- `actions`(`List[Card]`): 当前允许出的合法动作列表(Card 对象). 策略必须从中返回一个. 
- `order`(`int`): 本轮中的出牌顺序(0 表示本轮第一个出牌的玩家, 1/2/3 表示后手). 

返回值：
- 必须返回一个 `Card` 对象(或在进程内返回 Card 类型), 且该 Card 必须包含在 `actions` 中. 引擎会从玩家手牌中移除该卡并把它放到桌面上. 

为什么会传入 `player`：
- `player` 便于获取内存中的完整状态(性能和便利性). 但模型提供者可以只使用 `player_info`, 完全忽略 `player`. 
