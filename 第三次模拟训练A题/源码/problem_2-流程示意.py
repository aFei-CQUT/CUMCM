import graphviz

# 创建一个有向图
dot = graphviz.Digraph(format='png', engine='dot')

# 设置图的属性
dot.attr(rankdir='TB', size='10,10')

# 设置全局字体属性
dot.attr(fontsize='16', fontname='SimHei')  # 设置字体大小和字体类型

# 添加节点
dot.node('A', '开始', fontsize='16', fontname='SimHei')
dot.node('B', '数据准备', fontsize='16', fontname='SimHei')
dot.node('C', '加载数据', fontsize='16', fontname='SimHei')
dot.node('D', '选择基学习器', fontsize='16', fontname='SimHei')

dot.node('E', '线性回归模型', fontsize='16', fontname='SimHei')
dot.node('F', '决策树模型', fontsize='16', fontname='SimHei')
dot.node('G', '随机森林模型', fontsize='16', fontname='SimHei')
dot.node('H', '梯度提升树模型', fontsize='16', fontname='SimHei')
dot.node('I', '支持向量机模型', fontsize='16', fontname='SimHei')

dot.node('J', '训练线性回归模型', fontsize='16', fontname='SimHei')
dot.node('K', '训练决策树模型', fontsize='16', fontname='SimHei')
dot.node('L', '训练随机森林模型', fontsize='16', fontname='SimHei')
dot.node('M', '训练梯度提升树模型', fontsize='16', fontname='SimHei')
dot.node('N', '训练支持向量机模型', fontsize='16', fontname='SimHei')

dot.node('O', '对新样本进行预测', fontsize='16', fontname='SimHei')
dot.node('P', '集成方法', fontsize='16', fontname='SimHei')
dot.node('Q', '结合多个基学习器的预测结果', fontsize='16', fontname='SimHei')
dot.node('R', '输出最终预测结果', fontsize='16', fontname='SimHei')
dot.node('S', '模型评估', fontsize='16', fontname='SimHei')
dot.node('T', '结束', fontsize='16', fontname='SimHei')

# 添加边
dot.edges(['AB', 'BC', 'CD'])
dot.edge('D', 'E')
dot.edge('D', 'F')
dot.edge('D', 'G')
dot.edge('D', 'H')
dot.edge('D', 'I')

dot.edges(['EJ', 'FK', 'GL', 'HM', 'IN'])
dot.edge('J', 'O')
dot.edge('K', 'O')
dot.edge('L', 'O')
dot.edge('M', 'O')
dot.edge('N', 'O')

dot.edge('O', 'P')
dot.edge('P', 'Q')
dot.edge('Q', 'R')
dot.edge('R', 'S')
dot.edge('S', 'T')

# 渲染并保存图像
dot.render('./res/png/2', cleanup=True)