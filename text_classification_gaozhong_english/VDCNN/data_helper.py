import csv
import numpy as np
from collections import defaultdict

class data_helper():
    def __init__(self, sequence_max_length=1014):
#        self.alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} '
        self.alphabet = " etaoinsrhldc.umwgfypbv,k)(-102x\"j53?46:的7_9z/8;词q’一改你在处修下文写空格加!“中不错个有出学每多以语题误要作”该面并请意第用线增英上注行生划对内分及数容后同其除\\是删给人假计为求段和或间仅校横最小从可听右桌选句信如号节者两答只短缺时起他国开掉话字回交把符能使…点了余定单言共读名许漏参斜篇师活们已适课限项老%入结会据左连子当提贯之均自•允头所动料好根到材括李来现涉发地表填华于我∧大换将标高考此这﹩阅年成总包家得习细经‘至情主理法关事[实准示方应]三正*看息少感相完●封设报谈明合日书友边章与想原己就说约件外物过位做部议图电述&力体况白前心问美独教各因目试等解钟期班全建评介构观尾但化论由画真汇某秒确很照去天游长也绍按周重进历性讲达姓列无身业新称概通佳受车本助举社向邮次象些£展二简接何月￡望流没式影引都然她近工网机认演$公置母稿么朋种记整比常利遍卷例满赛星知任水道园乐保手规@持打父度故直→必】【告②①安市须越果◆则几品欢样征卡③系响放决程旅传什导而组气反帮难更亲＄广才平先里着被排食馆四取算务愿复调范世希代识爱员收环需片④"
        self.char_dict = {}
        self.sequence_max_length = sequence_max_length
        for i,c in enumerate(self.alphabet):
            self.char_dict[c] = i+1

    def char2vec(self, text):
        data = np.zeros(self.sequence_max_length)
        for i in range(0, len(text)):
            if i >= self.sequence_max_length:
                return data
            elif text[i] in self.char_dict:
                data[i] = self.char_dict[text[i]]
            else:
                    # unknown character set to be 68
                data[i] = 490
        return data
    

    def load_csv_file(self, data_source, label_source, num_classes):
        """
        	Load CSV file, generate one-hot labels and process text data as Paper did.
		  """
        x_text = []
        y = []
        doc_count = 0
        
        with open(data_source, encoding = "utf-8") as csvFile:
            readCSV = csv.reader(csvFile, delimiter = ",")
            for row in readCSV:
                row = "".join(row)
                x_text.append(self.char2vec(row.lower()))   
                doc_count = doc_count + 1
        
        with open(label_source, encoding = "utf-8") as csvFile2:
            readCSV = csv.reader(csvFile2, delimiter = ",")
            for row in readCSV:
                d = defaultdict(list)
                for k,va in [(v,i) for i,v in enumerate(row)]:
                    d[k].append(va)
            
                for k in range(len(d.get("1.0"))):
                    index = d.get("1.0")[k]
                    row[index] = 1
                for k in range(len(d.get("0.0"))):
                    index = d.get("0.0")[k]
                    row[index] = 0
                
                y.append(row)
        
        return np.array(x_text), np.array(y)

    def load_dataset(self, dataset_path):
        # Read Classes Info
        num_classes = 236
        # Read CSV Info
        a, y = self.load_csv_file(dataset_path+'gaozhong_english_preprocessed.csv', dataset_path+'labels_english.csv', num_classes)
        
        # Randomly shuffle data
        np.random.seed(11) # 使得随机数列可预测
        # 产生一个array：起始点0，结束点len(y)，步长1。然后将其打乱。
        shuffle_indices = np.random.permutation(np.arange(len(y))) 
        x_shuffled = a[shuffle_indices] # 将文件句子和标签以同样的方式打乱
        y_shuffled = y[shuffle_indices]
        
        
        # Split train/test set
        #直接把dev_sample_index前用于训练,dev_sample_index后的用于训练;
        dev_sample_index = -1 * int(0.1 * float(len(y))) # -1:代表从后往前取
        train_data, test_data = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        train_label, test_label = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        
        
        return train_data, train_label, test_data, test_label

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
                
                
                
    def dev_batch_iter(self, data, batch_size, num_epochs, shuffle=False):
        """
        Generates a batch iterator for a dataset.
        """
        batch_size = 128
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/(batch_size)) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * (batch_size)
                end_index = min((batch_num + 1) * (batch_size), data_size)
                yield shuffled_data[start_index:end_index]