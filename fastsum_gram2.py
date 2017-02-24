
import nltk
from numpy import *
from nltk import *
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from sklearn import svm
import re
import os
import random


stopwords = nltk.corpus.stopwords.words('english')
porter = nltk.PorterStemmer()

#句子的预处理，小写，去标点符号，去停用词P64
#clear \n \t <P> </P> lower() 去标点符号
#输入raw_text，即未处理的文本（string），输出句子（list的字符串），处理过文本的单词组（lsit的字符串）
def Preprocess_Raw(raw_text):
    raw_text.replace('<P>',' ')
    sents = sent_tokenize(raw_text)
    #print(sents)
    unraw_text = raw_text.replace('<P>','').replace('</P>', '').replace('\n', '').replace('\t', '').lower()
    unraw_text = re.sub('[^a-zA-Z0-9 ]', '', unraw_text) #去标点符号
    unraw_words = nltk.word_tokenize(unraw_text)
    for i in range(len(sents)):
        sents[i] = sents[i].replace('<P>','')
        sents[i] = sents[i].replace('</P>', '')
        sents[i] = sents[i].replace('\n', '')
        sents[i] = sents[i].replace('\t', '')
        sents[i] = sents[i].lower()        #sent[i] string类型
        #print(i)
        #print(sents[i])
    #print(sents)
    for i in range(len(sents)):
        sents[i] = re.sub('[^a-zA-Z0-9 ]','',sents[i])
    #print("sents = ",sents)
    #print(unraw_words)
    unraw_words1 = []
    for w in unraw_words:
        if w not in stopwords:
            unraw_words1.append(porter.stem(w))
    unraw_words1 = list( nltk.bigrams(unraw_words1))
    return sents,unraw_words1

def Preprocess_Raw_one(raw_text):
    raw_text.replace('<P>',' ')
    sents = sent_tokenize(raw_text)
    #print(sents)
    unraw_text = raw_text.replace('<P>','').replace('</P>', '').replace('\n', '').replace('\t', '').lower()
    unraw_text = re.sub('[^a-zA-Z0-9 ]', '', unraw_text) #去标点符号
    unraw_words = nltk.word_tokenize(unraw_text)
    for i in range(len(sents)):
        sents[i] = sents[i].replace('<P>','')
        sents[i] = sents[i].replace('</P>', '')
        sents[i] = sents[i].replace('\n', '')
        sents[i] = sents[i].replace('\t', '')
        sents[i] = sents[i].lower()        #sent[i] string类型
        #print(i)
        #print(sents[i])
    #print(sents)
    for i in range(len(sents)):
        sents[i] = re.sub('[^a-zA-Z0-9 ]','',sents[i])
    #print("sents = ",sents)
    #print(unraw_words)

    unraw_words1 = list( nltk.bigrams(unraw_words))
    return sents,unraw_words1


#统计单词频率 text应该为分词后的，不能是未处理的文章，即raw
#,headline,title,description
#返回 句子的总数,每个句子的词频，topic title ,description, headline, 句子长度,sentence_binary,sentence_value
def Word_Frequency(text_sents,text_word,title,description,headline):
    #print(text_word)
    word_fdist = FreqDist(text_word)
    #print('len=',len(text_word))
    #print(word_fdist)
    sents_word_frequency = []
    topic_title_frequency = []
    description_frequency = []
    headline_frequency = []
    sentence_length = []
    sentence_binary = []
    sentence_value = []
    #print("sentence length=",len(text_sents))
    for i in range(len(text_sents)):
        num = 0
        sentence_words = nltk.word_tokenize(text_sents[i])
        sentence_words = [nltk.PorterStemmer().stem(w) for w in sentence_words if w not in stopwords]
        sentence_words = list(nltk.bigrams(sentence_words))
        #print(sentence_words)
        if(len(sentence_words) >= 2):
            for j in range(len(sentence_words)):
                num += word_fdist[sentence_words[j]]
            num_average = num/len(sentence_words)
            sents_word_frequency.insert(i,num_average)
    #print(sents_word_frequency)
    #print(len(sents_word_frequency))
    #topci title frequency
    title = re.sub('[^a-zA-Z0-9 ]', '', title)
    title_words = nltk.word_tokenize(title)
    title_words = [nltk.PorterStemmer().stem(w) for w in title_words if w not in stopwords]
    title_words = list(nltk.bigrams(title_words))
    title_fdist = FreqDist(title_words)
    #print("def_title =",title_words)
    for i in range(len(text_sents)):
        num = 0
        sentence_words = nltk.word_tokenize(text_sents[i])
        sentence_words = [nltk.PorterStemmer().stem(w) for w in sentence_words if w not in stopwords]
        sentence_words = list(nltk.bigrams(sentence_words))
        #print(sentence_words)
        if(len(sentence_words) >= 2):
            for j in range(len(sentence_words)):
                num += title_fdist[sentence_words[j]]
            num_average = num/len(sentence_words)
            topic_title_frequency.insert(i,num_average)
    #print(topic_title_frequency)
    #print(len(topic_title_frequency))
    #topic description frequency
    description = re.sub('[^a-zA-Z0-9 ]', '', description)
    description_words = nltk.word_tokenize(description)
    description_words = [nltk.PorterStemmer().stem(w) for w in description_words if w not in stopwords]
    description_words = list(nltk.bigrams(description_words))
    description_fdist = FreqDist(description_words)
    #print("def_description =", description_words)
    for i in range(len(text_sents)):
        num = 0
        sentence_words = nltk.word_tokenize(text_sents[i])
        sentence_words = [nltk.PorterStemmer().stem(w) for w in sentence_words if w not in stopwords]
        sentence_words = list(nltk.bigrams(sentence_words))
        # print(sentence_words)
        if (len(sentence_words) >= 2):
            for j in range(len(sentence_words)):
                num += description_fdist[sentence_words[j]]
            num_average = num / len(sentence_words)
            description_frequency.insert(i, num_average)
    #print(description_frequency)
    #print(len(description_frequency))
    # headline frequency
    headline = re.sub('[^a-zA-Z0-9 ]', '', headline)
    headline_words = nltk.word_tokenize(headline)
    headline_words = [nltk.PorterStemmer().stem(w) for w in headline_words if w not in stopwords]
    headline_words = list(nltk.bigrams(headline_words))
    headline_fdist = FreqDist(headline_words)
    #print("def_headline =", headline_words)
    for i in range(len(text_sents)):
        num = 0
        sentence_words = nltk.word_tokenize(text_sents[i])
        sentence_words = [nltk.PorterStemmer().stem(w) for w in sentence_words if w not in stopwords]
        sentence_words = list(nltk.bigrams(sentence_words))
        # print(sentence_words)
        if (len(sentence_words) >= 2):
            for j in range(len(sentence_words)):
                num += headline_fdist[sentence_words[j]]
            num_average = num / len(sentence_words)
            headline_frequency.insert(i, num_average)
    #print(headline_frequency)
    #print(len(headline_frequency))
    #句子的长度
    for i in range(len(text_sents)):
        sentence_words = nltk.word_tokenize(text_sents[i])
        sentence_words = [nltk.PorterStemmer().stem(w) for w in sentence_words if w not in stopwords]
        sentence_words = list(nltk.bigrams(sentence_words))
        if (len(sentence_words) >= 2):
            sentence_length.insert(i, len(sentence_words))
    #print(sentence_length)
    #print(len(sentence_length))
    #sentence_binary
    num = len(sents_word_frequency)
    for i in range(len(sents_word_frequency)):
        if(i == 0):
            sentence_binary.insert(i, 1)
        else:
            sentence_binary.insert(i, 0.1)
    #print("sentence_binary=",sentence_binary)
    #sentence_value
    for i in range(num):
            sentence_value.insert(i, (num-i)/num)
    #print("sentence_value=",sentence_value)
    if(len(sents_word_frequency) == len(topic_title_frequency) == len(description_frequency) == len(headline_frequency) == \
               len(sentence_length) == len(sentence_binary) == len(sentence_value)):
        sss = 0 #不输出了
        # print("the size of all input string is the same")
    else:
        print("wrong input size")

    return len(sents_word_frequency),sents_word_frequency,topic_title_frequency,description_frequency,headline_frequency,\
           sentence_length,sentence_binary,sentence_value


#统计
def content_word_frequency(sents,corpus_root):
    #corpus_root = 'D:\\李卓聪\\DUC\\DUC data\\DUC2006_Summarization_Documents\\duc2006_docs\\D0601A'
    wordlist = PlaintextCorpusReader(corpus_root, '.*')
    all_raw = wordlist.fileids()
    all_raw_text = []
    for i in range(len(all_raw)):
        raw_text = wordlist.raw(all_raw[i])
        text_start = raw_text.find("<TEXT>")
        text_end = raw_text.find("</TEXT>")
        raw_text = raw_text[text_start + 6:text_end]
        all_raw_text.insert(i, raw_text)       #输出一个主题的25个documents，all_raw_text是个list，每个元素是每个document的text
    #print("all_raw_text",all_raw_text[0])
    #print("sents = ",sents)
    unraw_word = []
    for i in range(len(all_raw_text)):
        useless_data,unraw_word_n = Preprocess_Raw(all_raw_text[i])
        unraw_word.insert(i,unraw_word_n)
    #print("ddddddd",all_raw_text[0])
    #print("ddddddd", len(unraw_word))
    content_word_frequency = []
    for i in range(len(sents)):
        num_sents = 0
        sentence_words = nltk.word_tokenize(sents[i])
        sentence_words = [nltk.PorterStemmer().stem(w) for w in sentence_words if w not in stopwords]
        sentence_words = list(nltk.bigrams(sentence_words))
        #print(sentence_words)
        if(len(sentence_words) >= 2):
            for j in range(len(sentence_words)):
                num_word = 0
                for k in range(len(unraw_word)):
                    if (unraw_word[k].count(sentence_words[j]) >= 1):
                        num_word += 1
                #print("num = ",num_word)
                #print(sentence_words)
                num_sents += num_word
            num = num_sents/len(unraw_word)
            num_average = num/len(sentence_words)
            content_word_frequency.insert(i,num_average)
    #print("content_word_frequency = ",content_word_frequency)
    #print("content_word_frequency of length = ", len(content_word_frequency))
    return content_word_frequency


#查找headline,正文text的内容
def String_Headline_Text(first_raw):
    # corpus_root = 'D:\\李卓聪\\DUC\\DUC data\\DUC2006_Summarization_Documents\\duc2006_docs\\D0601A'
    # wordlist = PlaintextCorpusReader(corpus_root, '.*')
    # all_raw = wordlist.fileids()
    # first_raw = wordlist.raw(all_raw[0])
    #print(first_raw)
    #查找headline句子的
    headline_start = first_raw.find("<HEADLINE>")
    headline_end = first_raw.find("</HEADLINE>")
    first_headline = first_raw[headline_start + 10:headline_end]
    #print(first_headline)
    #查找Text句子的
    text_start = first_raw.find("<TEXT>")
    text_end = first_raw.find("</TEXT>")
    first_text = first_raw[text_start + 6:text_end]
    #print(first_text)
    return first_headline,first_text


#查找headline，title的句子   进行一些头尾处理只留下真正的内容
#DUC2006_topics
def String_Title_Description():
    i_corpus_root = 'D:\\李卓聪\\DUC\\DUC topics'
    topics__wordlist = PlaintextCorpusReader(i_corpus_root,'.*')
    #j = topics__wordlist.fileids()
    topics_raw = topics__wordlist.raw('duc2006_topics.sgml')
    #print(topics_raw)
    #查找title句子的
    title_start = topics_raw.find("<title>")
    #print(title_start)
    title_end = topics_raw.find("</title>")
    #print(title_end)
    first_title = topics_raw[title_start+7:title_end]
    #print(first_title)
    #查找description句子的
    description_start = topics_raw.find("<narr>")
    description_end = topics_raw.find("</narr>")
    first_description = topics_raw[description_start + 6:description_end]
    #print(first_description)
    return first_title,first_description



#统计DUC result
def Data_Model(i_num):
    corpus_root = 'D:\\李卓聪\\DUC\\DUC results\\2006-NISTeval\\ROUGE\\models'
    wordlist = PlaintextCorpusReader(corpus_root, '.*')
    a = wordlist.fileids()
    model_raw = [[]] * 4
    model_raw[0] = wordlist.raw(a[i_num*4])
    useless, model_raw[0] = Preprocess_Raw_one(model_raw[0])
    for i in range(3):
        model_raw[i+1] = wordlist.raw(a[i_num*4 + i + 1])
        useless,model_raw[i+1] = Preprocess_Raw_one(model_raw[i+1])
    #print("awewewewew",model_raw)
    return model_raw

def Data_Model_2005(corpus_root,i_num):
    #corpus_root = 'D:\\李卓聪\\DUC\\DUC results\\2006-NISTeval\\ROUGE\\models'
    wordlist = PlaintextCorpusReader(corpus_root, '.*')
    a = wordlist.fileids()
    i_num_list = [4,4,9,4,4,9,\
                  4,4,4,9,4,4,4,4,\
                  4,4,9,4,4,9,4,\
                  4,4,9,9,4,9,\
                  4,4,9,4,9,9,9,4,\
                  9,4,9,4,4,4,4,\
                  9,9,9,9,9,4,9,4]
    before_sum = 0
    for k in range(i_num):
        before_sum += i_num_list[k]
    model_raw = re.sub('[^a-zA-Z0-9 .!?]', '', wordlist.raw(a[before_sum]))
    for i in range(i_num_list[i_num] - 1):
        print("before_sum + 1 + i = ",before_sum + 1 + i)
        model_raw = model_raw + re.sub('[^a-zA-Z0-9 .!?]', '', wordlist.raw(a[before_sum + 1 + i]))
    useless,model_raw = Preprocess_Raw(model_raw)
    #print("awewewewew",model_raw)
    return model_raw

#sentence word overlap score
#medel_raw经过预处理的单词组，sents是句子的string
def overlap_score(sents,model_raw):
    sents_score = []
    for i in range(len(sents)):
        overlap_num = 0
        sentence_words = nltk.word_tokenize(sents[i])
        sentence_words = [nltk.PorterStemmer().stem(w) for w in sentence_words if w not in stopwords]
        sentence_words = list(nltk.bigrams(sentence_words))
        if (len(sentence_words) >= 2):
            for j in range(len(sentence_words)):
                if(model_raw.count(sentence_words[j]) >=1):
                    overlap_num += 1
                    #print(sentence_words[j])
            overlap_num = overlap_num/len(sentence_words)
            #print(overlap_num)
            sents_score.insert(i, overlap_num)
    #print("sents_score =", len(sents_score))
    #print("sents_score = ",sents_score)
    return sents_score

def turn_to_sample(length,sents_word_frequency,topic_title_frequency,description_frequency,\
                   headline_frequency,sentence_length,sentence_binary,sentence_value,content_word_frequency):
    all_sample = []
    for i in range(length):
        sample_data = []
        sample_data.insert(0, sents_word_frequency[i])
        sample_data.insert(1, topic_title_frequency[i])
        sample_data.insert(2, description_frequency[i])
        sample_data.insert(3, headline_frequency[i])
        sample_data.insert(4, sentence_length[i])
        sample_data.insert(5, sentence_binary[i])
        sample_data.insert(6, sentence_value[i])
        sample_data.insert(7, content_word_frequency[i])
        all_sample.insert(i, sample_data)
    #print("all_sample", all_sample)
    return all_sample

#ROUGE分数
# def rouge_score(auto_summary,model_summary):
#     sss


def Rouge(Ref_by_all_Words, Text_S_by_all_Words):  # ref是标准摘要，S是算法提出的摘要,都已经转化成n-gram的链表，其中Ref可能有多个标准文档
    # 先全部转化成Text类型
    Ref_by_all_Words = Text(Ref_by_all_Words)
    Text_S_by_all_Words = Text(Text_S_by_all_Words)
    id_no = len(Ref_by_all_Words)  # 标准文档的篇数。
    sum_ref = 0  # Ref总的n-gram数（所有篇的）

    # 把Text_S_by_all_Words转化成集合类型，使得其n-gram唯一，不重复计算

    # print('Text_S_by_all_Words:', Text_S_by_all_Words)
    Text_S_by_all_Words_Set = set(Text_S_by_all_Words)
    Text_S_by_all_Words_Set = Text(Text_S_by_all_Words_Set)
    # print('Text_S_by_all_Words_Set AFTER SET:', Text_S_by_all_Words_Set)


    S_len = len(Text_S_by_all_Words_Set)
    sum_match = 0
    for i in range(id_no):
        sum_ref = sum_ref + len(Ref_by_all_Words[i])

    for i in range(S_len):
        term = Text_S_by_all_Words_Set[i]
        for j in range(id_no):
            no = min(Ref_by_all_Words[j].count(term), Text_S_by_all_Words.count(term))
            sum_match = sum_match + no
    rouge = sum_match / sum_ref
    print('sum_match = ', sum_match)
    print('sum_ref = ', sum_ref)
    print('rouge =', rouge)
    return sum_match, sum_ref, rouge

def Rouge_new(Ref_by_all_Words, Text_S_by_all_Words):
    id_no = len(Ref_by_all_Words)
    S_len = len(Text_S_by_all_Words)
    all_rouge = 0
    for i in range(id_no):
        sum_match = 0
        for j in range(len(Ref_by_all_Words[i])):
            term = Ref_by_all_Words[i][j]
            no = (1 if Text_S_by_all_Words.count(term)>=1 else 0)
            sum_match = sum_match + no
        rouge = sum_match / len(Ref_by_all_Words[i])
        print("rouge = ",rouge)
        #all_rouge = ( rouge if rouge<all_rouge else all_rouge)
        all_rouge = (rouge if rouge > all_rouge else all_rouge)
    return all_rouge


#SVM的使用
def SVR_data(input_seven,ouput_one,test_data):
    X = input_seven
    y = ouput_one
    clf = svm.SVR()
    svr_parameter = clf.fit(X, y)
    data_predict = [[]] * len(test_data)
    #print(svr_parameter)
    #SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',\
    #    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    for i in range(len(test_data)):
        data_predict[i] = clf.predict(test_data[i])
    # print(data_predict)
    return data_predict
    #array([ 1.5])

#corpus_root = 'D:\\李卓聪\\DUC\\DUC data\\DUC2005_Summarization_Documents\\duc2005_docs\\d301i'


def getsomething():
    dir_names=[]
    k = 0
    str_file="D:\\李卓聪\\DUC\\DUC data\\DUC2005_Summarization_Documents\\duc2005_docs"
    corpus_root_model = "D:\\李卓聪\\DUC\\DUC results\\2005-results\\ROUGE\\models"
    for dirpaths, dirnames, filenames in os.walk(str_file):
        for dirname in dirnames:
            #print(dirname)
            dir_names.insert(k,dirname)
            k += 1
    print("dir_names",dir_names)
    print("dir_names",len(dir_names))
    for i in range(len(dir_names)):
        print("the i is =",i)
        corpus_root = 'D:\\李卓聪\\DUC\\DUC data\\DUC2005_Summarization_Documents\\duc2005_docs\\' + dir_names[i]
        wordlist = PlaintextCorpusReader(corpus_root, '.*')
        all_raw = wordlist.fileids()
        model_raw = Data_Model_2005(corpus_root_model,i)
        for j in range(len(all_raw)):
            the_raw = wordlist.raw(all_raw[j])
            the_headline, the_text = String_Headline_Text(the_raw)
            the_title, the_description = String_Title_Description()
            sents, unraw_words = Preprocess_Raw(the_text)
            the_length, sents_word_frequency, topic_title_frequency, description_frequency, headline_frequency, \
            sentence_length, sentence_binary, sentence_value \
                = Word_Frequency(sents, unraw_words, the_title, the_description, the_headline)
            the_content_word_frequency = content_word_frequency(sents,corpus_root)
            sents_score = overlap_score(sents, model_raw)
            if(j == 0 and i == 0 ):
                combine_sents_word_frequency = sents_word_frequency
                combine_topic_title_frequency = topic_title_frequency
                combine_description_frequency = description_frequency
                combine_headline_frequency = headline_frequency
                combine_sentence_length = sentence_length
                combine_sentence_binary = sentence_binary
                combine_sentence_value = sentence_value
                combine_the_content_word_frequency = the_content_word_frequency
                combine_sents_score = sents_score
            else:
                combine_sents_word_frequency.extend(sents_word_frequency)
                combine_topic_title_frequency.extend(topic_title_frequency)
                combine_description_frequency.extend(description_frequency)
                combine_headline_frequency.extend(headline_frequency)
                combine_sentence_length.extend(sentence_length)
                combine_sentence_binary.extend(sentence_binary)
                combine_sentence_value.extend(sentence_value)
                combine_the_content_word_frequency.extend(the_content_word_frequency)
                combine_sents_score.extend(sents_score)

    all_sample = turn_to_sample(len(combine_sents_word_frequency), combine_sents_word_frequency, combine_topic_title_frequency, combine_description_frequency, \
                                combine_headline_frequency, combine_sentence_length, \
                                combine_sentence_binary, combine_sentence_value,combine_the_content_word_frequency)

    #test_data = [[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1]]

    dir_names=[]
    # all_test_data[15] = []
    k = 0
    str_file="D:\\李卓聪\\DUC\\DUC data\\DUC2006_Summarization_Documents\\duc2006_docs"
    #corpus_root_model = "D:\李卓聪\DUC\DUC results\2006-NISTeval\ROUGE\models"
    for dirpaths, dirnames, filenames in os.walk(str_file):
        for dirname in dirnames:
            #print(dirname)
            dir_names.insert(k,dirname)
            k += 1
    #print("dir_names",dir_names)
    data_predict = [[]] * len(dir_names)
    all_test_data = [[]] * len(dir_names)
    sents_together = [[]] * len(dir_names)
    for i in range(len(dir_names)):
        #i_num = 0
        print("the i of 2006 is = ",i)
        corpus_root = "D:\\李卓聪\\DUC\\DUC data\\DUC2006_Summarization_Documents\\duc2006_docs\\" + dir_names[i]
        wordlist = PlaintextCorpusReader(corpus_root, '.*')
        all_raw = wordlist.fileids()
        #model_raw = Data_Model(corpus_root_model,i_num)
        for j in range(len(all_raw)):
            the_raw = wordlist.raw(all_raw[j])
            the_headline, the_text = String_Headline_Text(the_raw)
            the_title, the_description = String_Title_Description()
            sents, unraw_words = Preprocess_Raw(the_text)
            the_length, sents_word_frequency, topic_title_frequency, description_frequency, headline_frequency, \
            sentence_length, sentence_binary, sentence_value \
                = Word_Frequency(sents, unraw_words, the_title, the_description, the_headline)
            the_content_word_frequency = content_word_frequency(sents,corpus_root)
            #sents_score = overlap_score(sents, model_raw)
            #将每个document的句子合在一起
            if(j == 0  ):
                combine_sents_word_frequency = sents_word_frequency
                combine_topic_title_frequency = topic_title_frequency
                combine_description_frequency = description_frequency
                combine_headline_frequency = headline_frequency
                combine_sentence_length = sentence_length
                combine_sentence_binary = sentence_binary
                combine_sentence_value = sentence_value
                combine_the_content_word_frequency = the_content_word_frequency
                #combine_sents_score = sents_score
                sents_together[i] = sents
            else:
                combine_sents_word_frequency.extend(sents_word_frequency)
                combine_topic_title_frequency.extend(topic_title_frequency)
                combine_description_frequency.extend(description_frequency)
                combine_headline_frequency.extend(headline_frequency)
                combine_sentence_length.extend(sentence_length)
                combine_sentence_binary.extend(sentence_binary)
                combine_sentence_value.extend(sentence_value)
                combine_the_content_word_frequency.extend(the_content_word_frequency)
                #combine_sents_score.extend(sents_score)
                sents_together[i].extend(sents)  #将每个document的句子合在一起

            all_test_data[i] = turn_to_sample(len(combine_sents_word_frequency), combine_sents_word_frequency, combine_topic_title_frequency, combine_description_frequency, \
                                combine_headline_frequency, combine_sentence_length, \
                                combine_sentence_binary, combine_sentence_value,combine_the_content_word_frequency)

    data_predict = SVR_data(all_sample,combine_sents_score,all_test_data)
    return data_predict
#642-661  用于训练
# data_predict = getsomething()
#
# print("len(data_predict) = ",len(data_predict))
# for i in range(len(data_predict)):
#     for j in range(len(data_predict[i])):
#         print("len(data_predict[i][j]) = ", data_predict[i][j],"j = ",j)
#         # print(data_predict[i])
#     print("the document is ", i,"!!!!!!!!!!!!!!!!!!!!!!" )
#
# #print(data_predict)
#
# fl = open('D:\\李卓聪\\save_list.txt', 'w')
# for j in range(len(data_predict)):
#     save_file = "D:\\李卓聪\\save_list_2_" + str(j) + ".txt"
#     fl = open(save_file, 'w')
#     for i in data_predict[j]:
#         fl.write(str(i))
#         fl.write("\n")
#     fl.close()


print("finish saving!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# print(sents_together)

# 读取文档中的weight值，输入第i篇document，返回float型的数组
def read_weight_file(i):
    save_file = "D:\\李卓聪\\save_list_2_" + str(i) + ".txt"
    f = open(save_file, 'r')
    weight_list = f.readlines()
    weight_list_adjust = [0] * len(weight_list)
    for j in range(len(weight_list)):
        just = float(weight_list[j].replace('\n', ''))
        weight_list_adjust[j] = just
    return weight_list_adjust



def get_sents_together():
    dir_names=[]
    k = 0
    str_file="D:\\李卓聪\\DUC\\DUC data\\DUC2006_Summarization_Documents\\duc2006_docs"
    for dirpaths, dirnames, filenames in os.walk(str_file):
        for dirname in dirnames:
            dir_names.insert(k,dirname)
            k += 1
    sents_together = [[]] * len(dir_names)
    for i in range(len(dir_names)):
        corpus_root = "D:\\李卓聪\\DUC\\DUC data\\DUC2006_Summarization_Documents\\duc2006_docs\\" + dir_names[i]
        wordlist = PlaintextCorpusReader(corpus_root, '.*')
        all_raw = wordlist.fileids()
        for j in range(len(all_raw)):
            the_raw = wordlist.raw(all_raw[j])
            the_headline, the_text = String_Headline_Text(the_raw)
            sents, unraw_words = Preprocess_Raw(the_text)
            if (j == 0):
                sents_together[i] = sents
            else:
                sents_together[i].extend(sents)  # 将每个document的句子合在一起
    return sents_together

# def get_summary(k):
#     sents_together = get_sents_together()
#     weight_list = read_weight_file(k)
#     weight_sorted = sorted(weight_list,reverse = True)
#     num = weight_list.index(weight_sorted[0])
#     summary = [[]] * 1
#     summary[0] = sents_together[k][num]
#     summary_word = sents_together[k][num]
#     total_word = len(nltk.word_tokenize(sents_together[k][num]))
#     print(total_word)
#     i = 1
#     while(total_word <= 250):
#         num = weight_list.index(weight_sorted[i])
#         summary.append(sents_together[k][num])
#         summary_word += sents_together[k][num]
#         total_word += len(nltk.word_tokenize(sents_together[k][num]))
#         print(total_word)
#         i += 1
#     #print(summary)
#     return summary,summary_word

def get_summary(k):
    sents_together = get_sents_together()
    weight_list = read_weight_file(k)
    weight_sorted = sorted(weight_list,reverse = True)
    num = random.randint(0, len(weight_list))
    summary = [[]] * 1
    summary[0] = sents_together[k][num]
    summary_word = sents_together[k][num]
    total_word = len(nltk.word_tokenize(sents_together[k][num]))
    print(total_word)
    i = 1
    while(total_word <= 250):
        num = random.randint(0,len(weight_list))
        summary.append(sents_together[k][num])
        summary_word += sents_together[k][num]
        total_word += len(nltk.word_tokenize(sents_together[k][num]))
        #print(total_word)
        i += 1
    #print(summary)
    return summary,summary_word

# All summaries were truncated to 250 words before being evaluated
rouge_all = 0
for i in range(50):
    summary,summary_word = get_summary(i)
    useless,summary_word = Preprocess_Raw_one(summary_word)
    #print(summary)
    #print(summary_word)

    model_raw = Data_Model(i)
    print(model_raw)
    print(summary_word)
    #print(len(model_raw),model_raw)

    print("i = ",i)
    # ll = ['dd','ff','rr']
    # kk = [['d','ff','rr'],['dd','ff','rr']]
    rouge_new_one = Rouge_new(model_raw, summary_word)
    rouge_all +=rouge_new_one
print("results = ",rouge_all/50)


