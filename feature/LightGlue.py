from feature import my_feature_class

class lightglue(my_feature_class.feature):
    '''
        LightGlue特征点提取、描述、匹配
        :param imgpaths: 全部图片路径数组
        :param imgnum: 图片数量
    '''
    def __init__(self, imgpaths: list, imgnum: int):
        # 调用父类的初始化方法
        super().__init__(imgpaths, imgnum)