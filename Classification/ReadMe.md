# 基于bert的文本分类,两种方式使用Bert
1. 直接改run_classifier.py
    1. 加入了MyTextClassification类
    2. processors中增加该类
    3. 看run.sh
2. 外部调用bert,额。。其实就是把run_classifier.py重写了一遍,便于调用