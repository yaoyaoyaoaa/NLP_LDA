# _*_ coding:utf-8 _*_

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#使得词语不能被分开
jieba.suggest_freq('负荷位置', True)
'''
jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('京州', True)
'''
#读取文件
with open('./nlp_test6.txt', encoding='UTF-8') as f:
    document = f.read()

    document_decode = document.encode('GBK')
    document_cut = jieba.cut(document_decode)
    # print  ' '.join(jieba_cut)  //如果打印结果，则分词效果消失，后面的result无法显示
    result = ' '.join(document_cut)
    result = result.encode('utf-8')
    with open('./nlp_test7.txt', 'wb') as f2:
        f2.write(result)
f.close()
f2.close()

#从文件导入停用词表
stpwrdpath = "stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()
with open('./nlp_test7.txt', encoding='UTF-8') as f3:
    res1 = f3.read()
print(res1)
with open('./nlp_test8.txt', encoding='UTF-8') as f:
    document2 = f.read()

    document2_decode = document2.encode('GBK')
    document2_cut = jieba.cut(document2_decode)
    # print  ' '.join(jieba_cut)
    result = ' '.join(document2_cut)
    result = result.encode('utf-8')
    with open('./nlp_test9.txt', 'wb') as f2:
        f2.write(result)
f.close()
f2.close()
with open('./nlp_test10.txt', encoding='UTF-8') as f4:
    res2 = f4.read()
print(res2)
jieba.suggest_freq('负荷位置', True)
with open('./nlp_test10.txt', encoding='UTF-8') as f:
    document3 = f.read()

    document3_decode = document3.encode('GBK')
    document3_cut = jieba.cut(document3_decode)
    # print  ' '.join(jieba_cut)
    result = ' '.join(document3_cut)
    result = result.encode('utf-8')
    with open('./nlp_test11.txt', 'wb') as f3:
        f3.write(result)
f.close()
f3.close()
with open('./nlp_test11.txt', encoding='UTF-8') as f5:
    res3 = f5.read()
print(res3)

corpus = [res1,res2]
#构建句子 list 得词向量
vector = TfidfVectorizer(stop_words=stpwrdlst)
#得到词频矩阵得TF-IDF权重矩阵
tfidf = vector.fit_transform(corpus)
print(tfidf)
wordlist = vector.get_feature_names()#获取词袋模型中的所有词
# tf-idf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
weightlist = tfidf.toarray()
#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
for i in range(len(weightlist)):
    print("-------第",i,"段文本的词语tf-idf权重------")
    for j in range(len(wordlist)):
        print(wordlist[j],weightlist[i][j])

corpus = [res1,res2,res3]
cntVector = CountVectorizer(stop_words=stpwrdlst)
cntTf = cntVector.fit_transform(corpus)
print(cntTf)
#两个主题，迭代5次
lda = LatentDirichletAllocation(n_topics=2, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
docres = lda.fit_transform(cntTf)
print(lda.components_)
print(docres)