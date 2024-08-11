from graphviz import Digraph

dot = Digraph(comment='Optimization Process', format='png')
dot.attr(rankdir='TB', size='16,12')  # 增加图像尺寸

# 全局字体设置
dot.attr('graph', fontname='SimHei', fontsize='14')
dot.attr('node', fontname='SimHei', fontsize='14')  # 使用黑体，增大字号
dot.attr('edge', fontname='SimHei', fontsize='12')  # 边的标签也使用黑体

# 节点样式
dot.attr('node', shape='rectangle', style='filled', color='black', fillcolor='lightgray')

# 添加节点
dot.node('A', '开始', shape='ellipse')
dot.node('B', '生成训练数据')
dot.node('C', '训练随机森林模型')
dot.node('D', '优化过程', shape='diamond')
dot.node('E1', 'PSO优化')
dot.node('E2', 'SA优化')
dot.node('E3', 'ACO优化')
dot.node('F1', 'RF-PSO优化')
dot.node('F2', 'RF-SA优化')
dot.node('F3', 'RF-ACO优化')
dot.node('G', '比较所有优化结果')
dot.node('H', '选择最优解')
dot.node('I', '结束', shape='ellipse')

# 添加边
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E1', label='原始目标函数')
dot.edge('D', 'E2', label='原始目标函数')
dot.edge('D', 'E3', label='原始目标函数')
dot.edge('D', 'F1', label='RF预测目标函数')
dot.edge('D', 'F2', label='RF预测目标函数')
dot.edge('D', 'F3', label='RF预测目标函数')
for node in ['E1', 'E2', 'E3', 'F1', 'F2', 'F3']:
    dot.edge(node, 'G')
dot.edge('G', 'H')
dot.edge('H', 'I')

# 数据准备子图
with dot.subgraph(name='cluster_0') as c:
    c.attr(label='数据准备', fontname='SimHei', fontsize='16')  # 子图标签使用更大字号
    c.node('B1', '生成随机输入')
    c.node('B2', '计算原始目标函数值')
    c.node('B3', '过滤无效数据')
    c.edge('B', 'B1')
    c.edge('B1', 'B2')
    c.edge('B2', 'B3')
    c.edge('B3', 'C')

# 保存图片
dot.render('optimization_process', format='png', cleanup=True)
print("流程图已保存为 'optimization_process.png'")
