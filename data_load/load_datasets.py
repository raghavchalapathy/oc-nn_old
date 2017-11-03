import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib

dataPath = './data/'




def prepare_usps_mlfetch():

    import tempfile
    import pickle
    # print "importing usps from pickle file ....."

    with open(dataPath+'usps_data.pkl', "rb") as fp:
          loaded_data1 = pickle.load(fp)

    # test_data_home = tempfile.mkdtemp()
    # from sklearn.datasets.mldata import fetch_mldata
    # usps = fetch_mldata('usps', data_home=test_data_home)
    # print usps.target.shape
    # print type(usps.target)
    labels = loaded_data1['target']
    data = loaded_data1['data']
    # print "******",labels

    k_ones = np.where(labels == 2)
    label_ones = labels[k_ones]
    data_ones = data[k_ones]

    k_sevens = np.where(labels == 8)
    label_sevens = labels[k_sevens]
    data_sevens = data[k_sevens]
    #
    # print "data_sevens:",data_sevens.shape
    # print "label_sevens:",label_sevens.shape
    # print "data_ones:",data_ones.shape
    # print "label_ones:",label_ones.shape
    #
    data_ones = data_ones[:220]
    label_ones= label_ones[:220]
    data_sevens = data_sevens[:11]
    label_sevens = label_sevens[:11]

    data = np.concatenate((data_ones,data_sevens),axis=0)
    label = np.concatenate((label_ones,label_sevens),axis=0)
    label[0:220] = 1
    label[220:231] = -1
    # print "1-s",data[0]
    # print label
    # print "7-s",data[230]
    # print label
    # print "data:",data.shape
    # print "label:",label.shape

    # import matplotlib.pyplot as plt
    # plt.hist(label,bins=5)
    # plt.title("Count of  USPS Normal(1's) and Anomalous datapoints(7's) in training set")
    # plt.show()

    return [data,label]






def prepare_fake_news_data():

    import pandas as pd
    import numpy as np
    import spacy
    import scattertext as st
    #import imp; imp.reload(st)
    from IPython.display import IFrame
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:98% !important; }</style>"))
    import pickle
    from nltk.corpus import reuters

    nlp = spacy.en.English()
    uci_df = pd.read_csv(dataPath+'/uci-news-aggregator.csv.gz')
    traditional_publishers = ['Forbes','Bloomberg','Los Angeles Times','TIME','Wall Street Journal']
    repubable_celebrity_gossip = ['TheCelebrityCafe.com', 'PerezHilton.com']
    real_df = uci_df[uci_df['PUBLISHER'].isin(traditional_publishers)]
    real_df.columns = [x.lower() for x in real_df.columns]
    real_df['type'] = 'traditional'
    df = pd.read_csv(dataPath+'/fake.csv.gz')
    df = df.append(real_df)
    df = df[df['title'].apply(lambda x: type(x) == str)]
    df['clean_title'] = df['title'].apply(lambda x: ' '.join(x.split('>>')[0].split('>>')[0].split('[')[0].split('(')[0].split('|')[0].strip().split()))
    df = df.ix[df['clean_title'].drop_duplicates().index]
    # df['parsed_title'] = df['clean_title'].apply(nlp)
    df['meta'] = df['author'].fillna('') + df['publisher'].fillna('') + ' ' + df['site_url'].fillna('')
    df['category'] = df['type'].apply(lambda x: 'Real' if x == 'traditional' else 'Fake')
    fake_df = df[df['category'] == 'Fake']

    df_hate = fake_df[fake_df['type']=="hate"]
    df_fake = fake_df[fake_df['type']=="fake"]

    df_hate = df_hate[['text','type']]
    df_fake = df_fake[['text','type']]


    # let's take a look at the types of labels  are present in the data.
    # The +1 correspond to label hate and -1(outliers) correspond to label fake
    # print df_hate.head(5)
    # print df_fake.head(5)

    df_hate['type'][df_hate.type == "hate"] = 1
    df_fake['type'][df_fake.type == "fake"] = -1

    Xdata_hate = df_hate['text'].values 
    Xdata_fake = df_fake['text'].values 
    
    Xlabels_hate = df_hate['type'].values 
    Xlabels_fake = df_fake['type'].values 

    Xlabels = np.concatenate((Xlabels_hate,Xlabels_fake),axis=0)

    # print "Hate Class Shape",df_hate.shape
    # print "Fake Class shape",df_fake.shape
    # # print Xlabels.shape
    # # print Xlabels
    # import matplotlib.pyplot as plt
    # plt.hist(Xlabels,bins=5)
    # plt.title("Count of Hate and Fake class datapoints in data set")
    # plt.show()

    ### text vectorization--go from strings to lists of numbers
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectPercentile, f_classif

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    # data_train_transformed = vectorizer.fit_transform(df_hate['text']).toarray()
    # data_test_transformed  = vectorizer.transform(df_fake['text']).toarray()
    data_train_transformed = vectorizer.fit_transform(df_hate['text'])
    data_test_transformed  = vectorizer.transform(df_fake['text'])

    # print "Hate News",(data_train_transformed[:10])
    # print "Fake News",(data_test_transformed[:10])
    # print data_train_transformed.shape
    # print data_test_transformed.shape

    # print type(data_train_transformed)
    # print type(data_test_transformed)

    # # slim the data for training and testing
    selector = SelectPercentile(f_classif, percentile=1)
    labels_train = Xlabels_hate
    labels_test = Xlabels_fake
    selector.fit(data_train_transformed, labels_train)
    data_train_transformed = selector.transform(data_train_transformed).toarray()
    data_test_transformed  = selector.transform(data_test_transformed).toarray()

    # print data_train_transformed.shape
    # print data_test_transformed.shape

    # print type(data_train_transformed)
    # print type(data_test_transformed)

    data_train_transformed
    data_test_transformed
    labels_train = Xlabels_hate
    labels_test = Xlabels_fake

    data_train    = data_train_transformed
    data_test     = data_test_transformed
    targets_train = labels_train
    targets_test  = labels_test

    train_X    = data_train_transformed
    test_X     = data_test_transformed
    train_y = labels_train
    test_y  = labels_test
    train_y = train_y.tolist()
    train_y = [[i] for i in train_y]

    test_y = test_y.tolist()
    test_y = [[i] for i in test_y]

    return [train_X,train_y,test_X,test_y]



def prepare_cifar_10_data():

    from tflearn.datasets import cifar10
   
    image_and_anamolies = {'image': 5,'anomalies1':3,'anomalies2':3,'imagecount': 220,'anomaliesCount':11}

    def prepare_cifar_data_with_anamolies(original,original_labels,image_and_anamolies):

        imagelabel = image_and_anamolies['image']
        imagecnt = image_and_anamolies['imagecount']

        idx = np.where(original_labels ==imagelabel)

        idx = idx[0][:imagecnt]


        data = original[idx]
        data_labels = original_labels[idx]
        
        anamoliescnt = image_and_anamolies['anomaliesCount']
        anamolieslabel1 = image_and_anamolies['anomalies1']

        anmolies_idx1 = np.where(original_labels ==anamolieslabel1)
        anmolies_idx1 = anmolies_idx1[0][:(anamoliescnt)]
        
        ana_images1 = original[anmolies_idx1]
        ana_images1_labels = original_labels[anmolies_idx1]

        anamolieslabel2 = image_and_anamolies['anomalies2']
        anmolies_idx2 = np.where(original_labels ==anamolieslabel2)
        anmolies_idx2 = anmolies_idx2[0][:(anamoliescnt)]
        
        anomalies = original[anmolies_idx2]
        anomalies_labels = original_labels[anmolies_idx2]


        return [data,data_labels,anomalies,anomalies_labels]

    # load cifar-10 data
    ROOT = dataPath+'cifar-10_data/'
    (X, Y), (testX, testY) = cifar10.load_data(ROOT)
    testX = np.asarray(testX)
    testY = np.asarray(testY)


    [train_X,train_Y,testX,testY]=prepare_cifar_data_with_anamolies(X,Y,image_and_anamolies)

    # Make the Dog samples as positive and cat samples as negative
    testY[testY ==5] = 1
    testY[testY ==3] =-1
    test_y = [[i] for i in testY]

    ## Modify to suite the tensorflow code 
    train_Y[train_Y ==5]= 1
    train_Y[train_Y ==3]= -1
    train_y = [[i] for i in train_Y]

    train_X = np.reshape(train_X, (len(train_X),3072))
    testX = np.reshape(testX, (len(testX),3072))

    # print "Data Train Shape",train_X.shape
    # print "Data Test Shape",testX.shape

    # print "Label Train Shape",train_Y.shape
    # print "Label Test Shape",testY.shape

    data_train    = train_X
    data_test     = testX
    targets_train = train_Y
    targets_test  = testY

    train_X    = train_X
    test_X     = testX
    train_y = train_Y
    test_y  = testY
    train_y = train_y.tolist()
    train_y = [[i] for i in train_y]

    test_y = test_y.tolist()
    test_y = [[i] for i in test_y]
    
    return [train_X,train_y,test_X,test_y]



def prepare_spam_vs_ham_data():
 
    import pandas as pd
    df = pd.read_csv('./data/spam.csv', encoding='latin-1')
    df.head()
    df.shape
    # split into train and test

    df_ham = df[df['v1']=="ham"]
    df_spam = df[df['v1']=="spam"]




    df_ham['v1'][df_ham.v1 == "ham"] = 1
    df_spam['v1'][df_spam.v1 == "spam"] = -1

    df_ham = df_ham[0:220]
    df_spam = df_spam[0:11]

    Xlabels_ham = df_ham['v1'].values 
    labels_train = Xlabels_ham
    Xlabels_spam = df_spam['v1'].values 

    Xlabels = np.concatenate((Xlabels_ham,Xlabels_spam),axis=0)

    # print "Ham Class Shape",df_ham.shape
    # print "Spam Class shape",df_spam.shape
    # print Xlabels.shape
    # print Xlabels
    # import matplotlib.pyplot as plt
    # plt.hist(Xlabels,bins=5)
    # plt.title("Count of HAM and SPAM class datapoints in data set")
    # plt.show()


    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectPercentile, f_classif

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    data_train_transformed = vectorizer.fit_transform(df_ham['v2'])
    data_test_transformed  = vectorizer.transform(df_spam['v2'])


    # print "HAM: ",(data_train_transformed[:10])
    # print "SPAM:",(data_test_transformed[:10])

    # # # slim the data for training and testing
    # selector = SelectPercentile(f_classif, percentile=0.6)
    # labels_train = Xlabels_ham
    # labels_test = Xlabels_spam
    # selector.fit(data_train_transformed, labels_train)
    # data_train_transformed = selector.transform(data_train_transformed)
    # data_test_transformed  = selector.transform(data_test_transformed)

    # print data_train_transformed.shape
    # print data_test_transformed.shape

    # print type(data_train_transformed)
    # print type(data_test_transformed)

    labels_train = Xlabels_ham
    labels_test = Xlabels_spam


    data_train    = data_train_transformed
    data_test     = data_test_transformed
    targets_train = labels_train
    targets_test  = labels_test

    train_X    = data_train_transformed
    test_X     = data_test_transformed
    train_y = labels_train
    test_y  = labels_test
    train_y = train_y.tolist()
    train_y = [[i] for i in train_y]

    test_y = test_y.tolist()
    test_y = [[i] for i in test_y]

    

    return [train_X,train_y,test_X,test_y]
