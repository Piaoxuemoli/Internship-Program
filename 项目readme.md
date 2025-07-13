# AlphaZero Gomoku 项目报告

## 项目概述

本项目实现了基于 AlphaZero 算法的五子棋/四子棋AI系统，通过深度强化学习训练智能游戏AI模型。项目采用蒙特卡洛树搜索（MCTS）结合深度神经网络的方法，实现了从零开始的自我对弈训练，无需人类棋谱数据。

## 1. 强化学习训练游戏AI模型

### 1.1 AlphaZero算法核心
项目基于AlphaZero算法框架，主要包含以下关键组件：

#### **策略价值网络（Policy-Value Network）**
- **网络架构**：使用卷积神经网络处理棋盘状态
  - 输入层：4层棋盘状态表示（当前玩家、对手、空位、最后一步）
  - 卷积层：3层CNN（32、64、128个过滤器）
  - 策略头：输出每个位置的落子概率
  - 价值头：输出当前局面的胜率评估

```python
# 网络结构示例（from policy_value_net.py）
network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3, 3))
network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(3, 3))
network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(3, 3))

# 策略网络输出落子概率
policy_net = lasagne.layers.DenseLayer(policy_net, 
                                      num_units=board_width*board_height,
                                      nonlinearity=softmax)

# 价值网络输出局面评估
value_net = lasagne.layers.DenseLayer(value_net, num_units=1,
                                     nonlinearity=tanh)
```

#### **蒙特卡洛树搜索（MCTS）**
- **选择阶段**：基于UCB公式选择最优子节点
- **扩展阶段**：使用神经网络评估新节点
- **模拟阶段**：通过神经网络直接估值，无需随机模拟
- **反向传播**：更新路径上所有节点的访问次数和价值

```python
# UCB公式计算节点价值（from mcts_alphaZero.py）
def get_value(self, c_puct):
    self._u = (c_puct * self._P * 
               np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
    return self._Q + self._u
```

### 1.2 自我对弈训练流程

#### **数据收集**
- 每局游戏通过MCTS进行400次模拟决策
- 记录每步的棋盘状态、MCTS概率分布、最终胜负结果
- 数据增强：通过旋转和翻转扩充训练数据8倍

#### **网络更新**
- 批量大小：512个样本
- 学习率自适应调整：基于KL散度动态调节
- 损失函数：`loss = (z-v)² - π^T log(p) + c||θ||²`
  - 价值损失：预测胜率与实际结果差异
  - 策略损失：MCTS概率与网络输出交叉熵
  - 正则化项：防止过拟合

```python
# 训练流程（from train.py）
def run(self):
    for i in range(self.game_batch_num):
        # 1. 收集自我对弈数据
        self.collect_selfplay_data(self.play_batch_size)
        
        # 2. 更新神经网络
        if len(self.data_buffer) > self.batch_size:
            loss, entropy = self.policy_update()
            
        # 3. 评估模型性能
        if (i+1) % self.check_freq == 0:
            win_ratio = self.policy_evaluate()
            if win_ratio > self.best_win_ratio:
                self.policy_value_net.save_model('./best_policy.model')
```

## 2. 智能决策和动态策略调整

### 2.1 智能决策机制

#### **多层次决策架构**
1. **神经网络快速评估**：提供初始策略和价值估计
2. **MCTS深度搜索**：通过树搜索优化决策质量
3. **温度参数控制**：平衡探索与利用

#### **动态搜索深度**
- 标准模式：400次MCTS模拟
- 快速模式：可调整为更少模拟次数
- 关键局面：自动增加搜索深度

```python
# 动态决策过程（from mcts_alphaZero.py）
def get_move_probs(self, state, temp=1e-3):
    for n in range(self._n_playout):
        state_copy = copy.deepcopy(state)
        self._playout(state_copy)
    
    # 基于访问次数计算概率分布
    act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
    acts, visits = zip(*act_visits)
    act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
    return acts, act_probs
```

### 2.2 策略自适应调整

#### **学习率动态调整**
```python
# 基于KL散度调整学习率（from train.py）
if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
    self.lr_multiplier /= 1.5  # 降低学习率
elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
    self.lr_multiplier *= 1.5  # 提高学习率
```

#### **对手强度渐进提升**
- 初始对手：纯MCTS（1000次模拟）
- 性能评估：每50轮训练评估一次
- 自动升级：当胜率达到100%时，增加对手模拟次数

## 3. 支持多种游戏场景和复杂任务

### 3.1 多游戏模式支持

#### **游戏配置灵活性**
- **五子棋模式**：8×8棋盘，5子连珠获胜
- **四子棋模式**：6×6棋盘，4子连珠获胜
- **自定义规则**：支持任意棋盘大小和连珠数量

```python
# 灵活的游戏配置（from train.py）
class TrainPipeline():
    def __init__(self, init_model=None):
        self.board_width = 6      # 可配置棋盘宽度
        self.board_height = 6     # 可配置棋盘高度
        self.n_in_row = 4         # 可配置获胜条件
```

#### **多框架支持**
- **Theano/Lasagne**：原始实现版本
- **PyTorch**：支持GPU加速训练
- **TensorFlow**：工业级部署方案
- **Keras**：快速原型开发

### 3.2 复杂任务处理能力

#### **状态空间复杂度**
- 8×8五子棋：约10^29个可能状态
- 6×6四子棋：约10^18个可能状态
- 通过神经网络有效处理高维状态空间

#### **长序列决策**
- 平均对局长度：30-50步
- 每步需要考虑未来多步后果
- 通过价值网络评估长期收益

## 4. 可视化界面和AI表现观察

### 4.1 图形化游戏界面

#### **Pygame可视化系统**
项目提供了完整的图形化界面，支持：

- **实时棋盘显示**：清晰的网格和棋子渲染
- **鼠标交互**：点击落子，直观操作
- **游戏状态提示**：当前玩家、游戏进度显示
- **结果展示**：胜负判定和重新开始选项

```python
# 可视化界面实现（from human_play.py）
class Game_UI(object):
    def draw(self):
        # 绘制棋盘网格
        for i in range(self.width):
            pg.draw.line(self.screen, (0, 0, 0), ...)
        
        # 绘制棋子
        for move, player in self.board.states.items():
            if player == 1:
                pg.draw.circle(self.screen, (0, 0, 0), (x, y), 15)  # 黑子
            else:
                pg.draw.circle(self.screen, (255, 255, 255), (x, y), 15)  # 白子
```

#### **多模式游戏体验**
- **人类 vs AI**：测试AI性能，提供娱乐体验
- **AI自对弈**：观察AI策略演化，分析决策过程
- **难度选择**：不同模型文件提供不同AI强度

### 4.2 训练过程监控

#### **实时性能指标**
```python
# 训练监控指标（from train.py）
print(("kl:{:.5f},"           # KL散度
       "lr_multiplier:{:.3f}," # 学习率倍数
       "loss:{},"              # 总损失
       "entropy:{},"           # 策略熵
       "explained_var_old:{:.3f},"  # 价值预测准确度
       "explained_var_new:{:.3f}"
       ).format(...))
```

#### **AI表现分析**
- **胜率统计**：与基准MCTS的对战胜率
- **决策质量**：策略熵值变化趋势  
- **学习效率**：损失函数收敛情况
- **泛化能力**：不同起始位置的表现

### 4.3 模型演示效果

#### **训练成果展示**
- **400步MCTS演示**：项目提供了训练好的模型对战GIF
- **性能基准**：
  - 6×6四子棋：500-1000局训练达到良好水平（约2小时）
  - 8×8五子棋：2000-3000局训练达到高水平（约2天）

#### **实际对战表现**
- AI展现出明确的战略思维
- 能够识别攻守转换时机
- 具备多步棋局规划能力
- 在复杂局面下保持稳定决策

## 技术创新点

### 1. **零知识学习**
完全从随机策略开始，无需人类棋谱数据，通过自我对弈不断提升。

### 2. **端到端训练**
策略网络和价值网络联合训练，决策和评估能力同步提升。

### 3. **数据效率优化**
通过数据增强和经验回放，最大化训练样本利用率。

### 4. **自适应学习**
基于训练过程反馈，动态调整学习参数和对手强度。

## 项目成果与应用价值

### 1. **算法验证**
成功验证了AlphaZero算法在中等复杂度游戏中的有效性。

### 2. **教育价值**
提供了完整的强化学习项目实现，适合学习和研究。

### 3. **扩展潜力**
架构设计支持扩展到其他棋类游戏和决策问题。

### 4. **工程实践**
展示了从算法研究到产品实现的完整工程流程。

## 结论

本项目成功实现了基于AlphaZero算法的智能五子棋/四子棋AI系统，在强化学习训练、智能决策、多场景支持和可视化界面方面都取得了良好效果。项目不仅验证了深度强化学习在游戏AI中的强大能力，也为相关领域的研究和应用提供了宝贵的参考价值。

通过自我对弈训练，AI从零开始学习游戏策略，最终能够战胜传统的MCTS算法，展现出了强大的学习能力和决策水平。可视化界面的加入使得用户能够直观地观察和体验AI的表现，大大提升了项目的实用性和教育价值。
