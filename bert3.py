class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """生成一个InputExample.
        Args:
            guid: 每个example的独有id.
            text_a: 字符串，也就是输入的未分割的句子A，对于单句分类任务来说，text_a是必须有的
            text_b: 可选的输入字符串，单句分类任务不需要这一项，在预测下一句或者阅读理解任务中需要输入text_b，
            text_a和text_b中间使用[SEP]分隔
            label: 也是可选的字符串，就是对应的文本或句子的标签，在训练和验证的时候需要指定，但是在测试的时候可以不选
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
