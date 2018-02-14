import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib
import dataset
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from imutils import paths
import os
from keras.models import load_model
from keras.models import Model
import keras
from keras.layers import Dense, GlobalAveragePooling2D,Activation
nu = 0.001
activations = ["linear", "rbf"]
# dataPath = './data/'
dataPath = "/Users/raghav/Documents/Uni/oc-nn/data/"
from keras import backend as K

from keras.datasets import mnist
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.datasets import make_blobs

def readjpegimages2Array(filepath):
    from PIL import Image
    import os, numpy as np
    import matplotlib.pyplot as plt
    folder = filepath
    read = lambda imname: np.asarray(Image.open(imname))
    ims = [np.array(Image.open(os.path.join(folder, filename))) for filename in os.listdir(folder)]
    imageList = []
    for x in range(0,len(ims)):

        if(ims[x].shape ==(256,256)):
            imageList.append(ims[x])
    result = np.asarray(imageList)

    return result
def func_slice_stich_scene(X1_images,X2_images,Y1_labels,Y2_labels):


    # data_labels = np.concatenate((images_labels1, images_labels2), axis=0)

    res1 = np.array_split(X1_images, 2, axis=1)
    res2 = np.array_split(X2_images, 2, axis=1)
    data = np.hstack((res1[0], res2[0]))
    reslabel1 = np.array_split(Y1_labels, 2)
    reslabel2 = np.array_split(Y2_labels, 2)
    data_labels = np.hstack((reslabel1[0], reslabel2[0]))

    # imgs =  np.concatenate((images1, images2), axis=0)

    return [data,data_labels]
def prepare_scene_data_with_anamolies():



    images1 = readjpegimages2Array("/Users/raghav/Documents/Uni/oc-nn/data/scene/mountains/")
    images3 = readjpegimages2Array("/Users/raghav/Documents/Uni/oc-nn/data/scene/insidecity/")

    print images1.shape
    print images3.shape
    import numpy as np
    print len(images1)
    print len(images3)


    images1 = np.reshape(images1,(len(images1),256*256))
    images3 = np.reshape(images3,(len(images3), 256*256))
    temp_images = np.concatenate((images1,images3), axis=0)

    images_labels1 =  np.full(len(images1), 0)
    # images_labels2 = np.full(len(images2), 1)
    images_labels3 = np.full(len(images3), 1)
    temp_images_label = np.concatenate((images_labels1,images_labels3), axis=0)


    data_train = images1
    data_test = images3

    data_train_label = images_labels1
    data_test_label = images_labels3

    return [data_train,data_test,data_train_label,data_test_label]


def prepare_mnist_mlfetch():
    (x_train, x_trainLabels), (x_test, x_testLabels) = mnist.load_data()
    labels = x_trainLabels
    data = x_train
    # print "******",labels

    ## 4859 digit- 4
    k_four = np.where(labels == 4)
    label_four  = labels[k_four]
    data_four = data[k_four]

    k_zeros = np.where(labels == 0)
    k_sevens = np.where(labels == 7)
    k_nine = np.where(labels == 9)

    ## 265 (0,7,9)

    label_zeros = labels[k_zeros]
    data_zeros = data[k_zeros]

    label_sevens = labels[k_sevens]
    data_sevens = data[k_sevens]

    label_nine = labels[k_nine]
    data_nines = data[k_nine]


    data_four = data_four[:220]

    data_zeros = data_zeros[:5]
    data_sevens = data_sevens[:3]
    data_nines = data_nines[:3]

    data_sevens = data_sevens[:11]

    normal = data_four
    anomalies = np.concatenate((data_zeros, data_sevens,data_nines), axis=0)


    normal = np.reshape(normal,(len(normal),784))
    anomalies = np.reshape(anomalies, (len(anomalies), 784))

    return [normal,anomalies]

def prepare_synthetic_data():
    n_samples = 190
    centers = 1
    num_features = 512
    X, y = make_blobs(n_samples=n_samples, n_features=num_features, cluster_std=1.0,
                      centers=centers, shuffle=False, random_state=42)
    ## Add 10 Normally distributed anomalous points
    mu, sigma = 0, 2
    a = np.random.normal(mu, sigma, (10, num_features))
    X = np.concatenate((X, a), axis=0)
    y_a = np.empty(10)
    y_a.fill(5)
    y = np.concatenate((y, y_a), axis=0)
    print X.shape, y.shape
    data_train = X[0:190]
    label_train = y[0:190]
    data_test = X[190:200]
    label_test = y[190:200]
    return [data_train, label_train,data_test,label_test]

def prepare_mnist_with_anomalies():

    (x_train, x_trainLabels), (x_test, x_testLabels) = mnist.load_data()
    import tempfile
    import pickle
    # print "importing usps from pickle file ....."

    # with open(dataPath + 'usps_data.pkl', "rb") as fp:
    #     loaded_data1 = pickle.load(fp)

    # test_data_home = tempfile.mkdtemp()
    # from sklearn.datasets.mldata import fetch_mldata
    # usps = fetch_mldata('usps', data_home=test_data_home)
    # print usps.target.shape
    # print type(usps.target)
    labels = x_trainLabels
    data = x_train
    # print "******",labels

    k_ones = np.where(labels == 1)
    label_ones = labels[k_ones]
    data_ones = data[k_ones]

    k_sevens = np.where(labels == 7)
    label_sevens = labels[k_sevens]
    data_sevens = data[k_sevens]
    #
    # print "data_sevens:",data_sevens.shape
    # print "label_sevens:",label_sevens.shape
    # print "data_ones:",data_ones.shape
    # print "label_ones:",label_ones.shape
    #
    data_ones = data_ones[:220]
    label_ones = label_ones[:220]
    data_sevens = data_sevens[:11]
    label_sevens = label_sevens[:11]

    data = np.concatenate((data_ones, data_sevens), axis=0)
    label = np.concatenate((label_ones, label_sevens), axis=0)
    label[0:220] = 1
    label[220:231] = 7
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

    return [data, label]


def prepare_usps_mlfetch():

    import tempfile
    import pickle
    # print "importing usps from pickle file ....."

    with open(dataPath + 'usps_data.pkl', "rb") as fp:
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
    label_ones = label_ones[:220]
    data_sevens = data_sevens[:11]
    label_sevens = label_sevens[:11]

    data = np.concatenate((data_ones, data_sevens), axis=0)
    label = np.concatenate((label_ones, label_sevens), axis=0)
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

    return [data, label]


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
    uci_df = pd.read_csv(dataPath + '/uci-news-aggregator.csv.gz')
    traditional_publishers = ['Forbes', 'Bloomberg', 'Los Angeles Times', 'TIME', 'Wall Street Journal']
    repubable_celebrity_gossip = ['TheCelebrityCafe.com', 'PerezHilton.com']
    real_df = uci_df[uci_df['PUBLISHER'].isin(traditional_publishers)]
    real_df.columns = [x.lower() for x in real_df.columns]
    real_df['type'] = 'traditional'
    df = pd.read_csv(dataPath + '/fake.csv.gz')
    df = df.append(real_df)
    df = df[df['title'].apply(lambda x: type(x) == str)]
    df['clean_title'] = df['title'].apply(lambda x: ' '.join(x.split('>>')[0].split('>>')[0].split('[')[0].split('(')[0].split('|')[0].strip().split()))
    df = df.ix[df['clean_title'].drop_duplicates().index]
    # df['parsed_title'] = df['clean_title'].apply(nlp)
    df['meta'] = df['author'].fillna('') + df['publisher'].fillna('') + ' ' + df['site_url'].fillna('')
    df['category'] = df['type'].apply(lambda x: 'Real' if x == 'traditional' else 'Fake')
    fake_df = df[df['category'] == 'Fake']

    df_hate = fake_df[fake_df['type'] == "hate"]
    df_fake = fake_df[fake_df['type'] == "fake"]

    df_hate = df_hate[['text', 'type']]
    df_fake = df_fake[['text', 'type']]

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

    Xlabels = np.concatenate((Xlabels_hate, Xlabels_fake), axis=0)

    # print "Hate Class Shape",df_hate.shape
    # print "Fake Class shape",df_fake.shape
    # # print Xlabels.shape
    # # print Xlabels
    # import matplotlib.pyplot as plt
    # plt.hist(Xlabels,bins=5)
    # plt.title("Count of Hate and Fake class datapoints in data set")
    # plt.show()

    # text vectorization--go from strings to lists of numbers
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectPercentile, f_classif

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    # data_train_transformed = vectorizer.fit_transform(df_hate['text']).toarray()
    # data_test_transformed  = vectorizer.transform(df_fake['text']).toarray()
    data_train_transformed = vectorizer.fit_transform(df_hate['text'])
    data_test_transformed = vectorizer.transform(df_fake['text'])

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
    data_test_transformed = selector.transform(data_test_transformed).toarray()

    # print data_train_transformed.shape
    # print data_test_transformed.shape

    # print type(data_train_transformed)
    # print type(data_test_transformed)

    data_train_transformed
    data_test_transformed
    labels_train = Xlabels_hate
    labels_test = Xlabels_fake

    data_train = data_train_transformed
    data_test = data_test_transformed
    targets_train = labels_train
    targets_test = labels_test

    train_X = data_train_transformed
    test_X = data_test_transformed
    train_y = labels_train
    test_y = labels_test
    train_y = train_y.tolist()
    train_y = [[i] for i in train_y]

    test_y = test_y.tolist()
    test_y = [[i] for i in test_y]

    return [train_X, train_y, test_X, test_y]


def prepare_cifar_10_data():

    from tflearn.datasets import cifar10

    image_and_anamolies = {'image': 5, 'anomalies1': 3, 'anomalies2': 3, 'imagecount': 220, 'anomaliesCount': 11}

    def prepare_cifar_data_with_anamolies(original, original_labels, image_and_anamolies):

        imagelabel = image_and_anamolies['image']
        imagecnt = image_and_anamolies['imagecount']

        idx = np.where(original_labels == imagelabel)

        idx = idx[0][:imagecnt]

        data = original[idx]
        data_labels = original_labels[idx]

        anamoliescnt = image_and_anamolies['anomaliesCount']
        anamolieslabel1 = image_and_anamolies['anomalies1']

        anmolies_idx1 = np.where(original_labels == anamolieslabel1)
        anmolies_idx1 = anmolies_idx1[0][:(anamoliescnt)]

        ana_images1 = original[anmolies_idx1]
        ana_images1_labels = original_labels[anmolies_idx1]

        anamolieslabel2 = image_and_anamolies['anomalies2']
        anmolies_idx2 = np.where(original_labels == anamolieslabel2)
        anmolies_idx2 = anmolies_idx2[0][:(anamoliescnt)]

        anomalies = original[anmolies_idx2]
        anomalies_labels = original_labels[anmolies_idx2]

        return [data, data_labels, anomalies, anomalies_labels]

    # load cifar-10 data
    ROOT = dataPath + 'cifar-10_data/'
    (X, Y), (testX, testY) = cifar10.load_data(ROOT)
    testX = np.asarray(testX)
    testY = np.asarray(testY)

    [train_X, train_Y, testX, testY] = prepare_cifar_data_with_anamolies(X, Y, image_and_anamolies)

    # Make the Dog samples as positive and cat samples as negative
    testY[testY == 5] = 1
    testY[testY == 3] = -1
    test_y = [[i] for i in testY]

    # Modify to suite the tensorflow code
    train_Y[train_Y == 5] = 1
    train_Y[train_Y == 3] = -1
    train_y = [[i] for i in train_Y]

    train_X = np.reshape(train_X, (len(train_X), 3072))
    testX = np.reshape(testX, (len(testX), 3072))

    # print "Data Train Shape",train_X.shape
    # print "Data Test Shape",testX.shape

    # print "Label Train Shape",train_Y.shape
    # print "Label Test Shape",testY.shape

    data_train = train_X
    data_test = testX
    targets_train = train_Y
    targets_test = testY

    train_X = train_X
    test_X = testX
    train_y = train_Y
    test_y = testY
    train_y = train_y.tolist()
    train_y = [[i] for i in train_y]

    test_y = test_y.tolist()
    test_y = [[i] for i in test_y]

    return [train_X, train_y, test_X, test_y]


def prepare_spam_vs_ham_data():

    import pandas as pd
    df = pd.read_csv('./data/spam.csv', encoding='latin-1')
    df.head()
    df.shape
    # split into train and test

    df_ham = df[df['v1'] == "ham"]
    df_spam = df[df['v1'] == "spam"]

    df_ham['v1'][df_ham.v1 == "ham"] = 1
    df_spam['v1'][df_spam.v1 == "spam"] = -1

    df_ham = df_ham[0:220]
    df_spam = df_spam[0:11]

    Xlabels_ham = df_ham['v1'].values
    labels_train = Xlabels_ham
    Xlabels_spam = df_spam['v1'].values

    Xlabels = np.concatenate((Xlabels_ham, Xlabels_spam), axis=0)

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
    data_test_transformed = vectorizer.transform(df_spam['v2'])

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

    data_train = data_train_transformed
    data_test = data_test_transformed
    targets_train = labels_train
    targets_test = labels_test

    train_X = data_train_transformed
    test_X = data_test_transformed
    train_y = labels_train
    test_y = labels_test
    train_y = train_y.tolist()
    train_y = [[i] for i in train_y]

    test_y = test_y.tolist()
    test_y = [[i] for i in test_y]

    return [train_X, train_y, test_X, test_y]


def prepare_cifar_10_data_for_conv_net():

    # Prepare input data
    classes = ['dogs', 'cats']
    num_classes = len(classes)
    # 20% of the data will automatically be used for validation
    validation_size = 0.0
    img_size = 128
    num_channels = 3
    train_path = dataPath + '/cifar-10_data/train/'
    test_path = dataPath + '/cifar-10_data/test/'

    # We shall load all the training and validation images and labels into memory using openCV and use that during training
    data = dataset.read_train_sets(train_path, img_size, classes, validation_size=0)
    test_images, test_ids = dataset.read_test_set(test_path, img_size)
    # print("Size of:")
    # print("- Training-set:\t\t{}".format(len(data.train.images)))
    # print("- Test-set:\t\t{}".format(len(test_images)))
    # print("- Validation-set:\t{}".format(len(data.valid.labels)))

    return [data.train.images, data.train.labels, test_images, test_ids, data]

def add_new_last_layer(base_model_output,base_model_input):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model_output
  print "base_model.output",x.shape
  inp = base_model_input
  print "base_model.input",inp.shape
  dense1 = Dense(512, name="dense_output1")(x)  # new sigmoid layer
  dense1out = Activation("relu", name="output_activation1")(dense1)
  dense2 = Dense(1, name="dense_output2")(dense1out) #new sigmoid layer
  dense2out = Activation("relu",name="output_activation2")(dense2)  # new sigmoid layer
  model = Model(inputs=inp, outputs=dense2out)
  return model


def prepare_cifar_data_for_rpca_forest_ocsvm(train_path,test_path):

    print("[INFO] loading images for training...")
    data = []
    data_test = []
    labels = []
    labels_test = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(train_path)))
    # loop over the input training images
    image_dict = {}
    test_image_dict = {}
    i = 0
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        # image = cv2.resize(image, (1, 3072))
        image = img_to_array(image)
        data.append(image)
        image_dict.update({i: imagePath})
        i = i + 1

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        label = 1 if label == "dogs" else 0
        labels.append(label)
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data)

    y = keras.utils.to_categorical(labels, num_classes=2)
    # print image_dict

    print("[INFO] preparing test data (anomalous )...")
    testimagePaths = sorted(list(paths.list_images(test_path)))
    # loop over the test images
    j = 0
    for imagePath in testimagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        # image = cv2.resize(image, (1, 3072))
        image = img_to_array(image)
        data_test.append(image)
        test_image_dict.update({j: imagePath})
        j = j + 1
        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        label = 0 if label == "cats" else 1
        labels_test.append(label)
    # scale the raw pixel intensities to the range [0, 1]
    data_test = np.array(data_test)

    ## Normalise the data
    data = np.array(data) / 255.0
    data = np.reshape(data,(len(data),3072))
    data_test = np.array(data_test) / 255.0
    data_test = np.reshape(data_test, (len(data_test), 3072))




    return [data,data_test]

def prepare_cifar_data_for_cae_ocsvm(train_path,test_path,modelpath):
    ### Declare the training and test paths
    # modelpath = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/trained_models/cifar10_conv_3_id_32_e_1000_encoder.model"
    #
    # train_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/data/cifar-10/normal/"
    # test_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/data/cifar-10/anoma/"
    NB_IV3_LAYERS_TO_FREEZE = 23
    side = 32
    side = 32
    channel = 3
    keras.backend.clear_session()
    ## Declare the scoring functions
    g = lambda x: 1 / (1 + tf.exp(-x))

    # g  = lambda x : x # Linear
    def nnScore(X, w, V, g):

        # print "X",X.shape
        # print "w",w[0].shape
        # print "v",V[0].shape
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu(x):
        y = x
        y[y < 0] = 0
        return y

    def add_new_last_layer(base_model_output, base_model_input):
        """Add last layer to the convnet
        Args:
          base_model: keras model excluding top
          nb_classes: # of classes
        Returns:
          new keras model with last layer
        """
        x = base_model_output
        print "base_model.output", x.shape
        inp = base_model_input
        print "base_model.input", inp.shape
        dense1 = Dense(512, name="dense_output1")(x)  # new sigmoid layer
        dense1out = Activation("relu", name="output_activation1")(dense1)
        dense2 = Dense(1, name="dense_output2")(dense1out)  # new sigmoid layer
        dense2out = Activation("relu", name="output_activation2")(dense2)  # new sigmoid layer
        model = Model(inputs=inp, outputs=dense2out)
        return model

    # load the trained convolutional neural network freeze all the weights except for last four layers
    print("[INFO] loading network...")
    model = load_model(modelpath)
    model = add_new_last_layer(model.output, model.input)
    print(model.summary())
    print "Length of model layers...", len(model.layers)

    # Freeze the weights of untill the last four layers
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True

    print("[INFO] loading images for training...")
    data = []
    data_test = []
    labels = []
    labels_test = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(train_path)))
    # loop over the input training images
    image_dict = {}
    test_image_dict = {}
    i = 0
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (side, side))
        image = img_to_array(image)
        data.append(image)
        image_dict.update({i: imagePath})
        i = i + 1

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        label = 1 if label == "dogs" else 0
        labels.append(label)
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data)

    y = keras.utils.to_categorical(labels, num_classes=2)
    # print image_dict

    print("[INFO] preparing test data (anomalous )...")
    testimagePaths = sorted(list(paths.list_images(test_path)))
    # loop over the test images
    j = 0
    for imagePath in testimagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (side, side))
        image = img_to_array(image)
        data_test.append(image)
        test_image_dict.update({j: imagePath})
        j = j + 1
        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        label = 0 if label == "cats" else 1
        labels_test.append(label)
    # scale the raw pixel intensities to the range [0, 1]
    data_test = np.array(data_test)

    ## Normalise the data
    data = np.array(data) / 255.0
    data_test = np.array(data_test) / 255.0



    # Preprocess the inputs
    x_train = data
    x_test = data_test

    ## Obtain the intermediate output
    layer_name = 'dense_1'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)

    layer1 = model.get_layer("dense_output1")
    layer2 = model.get_layer("dense_output2")
    intermediate_output = intermediate_layer_model.predict(x_train)
    X = intermediate_output
    intermediate_output = intermediate_layer_model.predict(x_test)
    X_test = intermediate_output

    print X.shape
    print X_test.shape

    K.clear_session()

    return [X,X_test]


# USAGE
# python test_network.py --model dog_not_dog.model --image images/examples/dog_01.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from imutils import paths
import os
from keras.optimizers import SGD
import keras.backend as K
from keras.models import load_model
from keras.losses import customLoss
import tensorflow as tf
from keras.models import Model
import keras
from keras.layers import Dense
from keras.layers import Dense, GlobalAveragePooling2D,Activation
from keras.applications.vgg16 import preprocess_input

def prepare_pfam_data_for_ocsvm_isolationForest(data_path,biovecModelPath):
    import biovec
    import numpy as np
    # pv = biovec.ProtVec("some_fasta_file.fasta", out="output_corpusfile_path.txt")
    # pv["QAT"]
    # pv.to_vecs("ATATQSQSMTEEL")
    # pv.save('model_file_path')
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    # train_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/lstm_autoencoders/data/testout.csv"
    train_path = data_path
    protein_data = pd.read_csv(train_path, header=None, sep="\t")
    result = protein_data[0]
    print len(result)

    pv2 = biovec.models.load_protvec(biovecModelPath)
    protein_vectors = []
    for i in range(0, len(result)):
        print result[i]
        vec = pv2.to_vecs(result[i])
        protein_vectors.append(vec[0])

    x_train = np.asarray(protein_vectors)

    print x_train.shape

    train_encoded = x_train[0:50]
    test_encoded = x_train[54:59]
    print "Encoded Training samples:", train_encoded.shape
    print "Encoded Testing samples:", test_encoded.shape

    # Preprocess the inputs
    X = train_encoded
    X_test = test_encoded

    print X.shape
    print X_test.shape

    # Preprocess the inputs
    X = train_encoded
    X_test = test_encoded

    print X.shape
    print X_test.shape

    return [X,X_test]



def prepare_pfam_data_for_lstm_ae(data_path,modelpath):
    # train_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/lstm_autoencoders/data/testout.csv"


    # load the trained convolutional neural network freeze all the weights except for last four layers
    print("[INFO] loading network...")
    model = load_model(modelpath)
    print(model.summary())
    print "Length of model layers...", len(model.layers)

    # data_path = '/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/lstm_autoencoders/data/test_sol2sol.txt'
    data_path = data_path
    train_num_samples = 64
    laten_dim = 50
    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    lines = open(data_path).read().split('\n')
    for line in lines[: min(train_num_samples, len(lines) - 1)]:
        input_text, target_text = line.split('\t')
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)
    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())

    def encode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = model.predict(input_seq)
        return states_value

    encoded = []
    ## Encode the Train sequences
    for seq_index in range(train_num_samples):
        # Take one sequence (part of the training test)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        encoded_seq = encode_sequence(input_seq)
        # print('-')
        # print('Input sequence:', input_texts[seq_index])
        # print('Encoded sequence:', encoded_seq)
        encoded_state_h = np.reshape(encoded_seq[0], laten_dim)
        encoded_state_c = np.reshape(encoded_seq[1], laten_dim)
        encoded_state = np.concatenate((encoded_state_h, encoded_state_c), axis=0)
        encoded.append(encoded_state_c)

    x_train = np.asarray(encoded)
    X = x_train



    data_train = X[0:54]
    data_test = X[54:64]
    print "Data Train:", data_train.shape
    print "Data Test:", data_test.shape

    return [data_train,data_test]



def prepare_cifar_data_for_ae2_svdd(train_path,test_path,modelpath):
    ### Declare the training and test paths
    # modelpath = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/trained_models/cifar10_conv_3_id_32_e_1000_encoder.model"
    #
    # train_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/data/cifar-10/normal/"
    # test_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/data/cifar-10/anoma/"
    NB_IV3_LAYERS_TO_FREEZE = 4
    side = 32
    side = 32
    channel = 3

    ## Declare the scoring functions
    g = lambda x: 1 / (1 + tf.exp(-x))

    # g  = lambda x : x # Linear
    def nnScore(X, w, V, g):

        # print "X",X.shape
        # print "w",w[0].shape
        # print "v",V[0].shape
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu(x):
        y = x
        y[y < 0] = 0
        return y

    def add_new_last_layer(base_model_output, base_model_input):
        """Add last layer to the convnet
        Args:
          base_model: keras model excluding top
          nb_classes: # of classes
        Returns:
          new keras model with last layer
        """
        x = base_model_output
        print "base_model.output", x.shape
        inp = base_model_input
        print "base_model.input", inp.shape
        dense1 = Dense(128, name="dense_output1")(x)  # new sigmoid layer
        dense1out = Activation("relu", name="output_activation1")(dense1)
        dense2 = Dense(1, name="dense_output2")(dense1out)  # new sigmoid layer
        dense2out = Activation("relu", name="output_activation2")(dense2)  # new sigmoid layer
        model = Model(inputs=inp, outputs=dense2out)
        return model

    # load the trained convolutional neural network freeze all the weights except for last four layers
    print("[INFO] loading network...")
    model = load_model(modelpath)
    model = add_new_last_layer(model.output, model.input)
    print(model.summary())
    print "Length of model layers...", len(model.layers)

    # Freeze the weights of untill the last four layers
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True

    print("[INFO] loading images for training...")
    data = []
    data_test = []
    labels = []
    labels_test = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(train_path)))
    # loop over the input training images
    image_dict = {}
    test_image_dict = {}
    i = 0
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (side, side))
        image = img_to_array(image)
        data.append(image)
        image_dict.update({i: imagePath})
        i = i + 1

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        label = 1 if label == "dogs" else 0
        labels.append(label)
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data)

    y = keras.utils.to_categorical(labels, num_classes=2)
    # print image_dict

    print("[INFO] preparing test data (anomalous )...")
    testimagePaths = sorted(list(paths.list_images(test_path)))
    # loop over the test images
    j = 0
    for imagePath in testimagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (side, side))
        image = img_to_array(image)
        data_test.append(image)
        test_image_dict.update({j: imagePath})
        j = j + 1
        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        label = 0 if label == "cats" else 1
        labels_test.append(label)
    # scale the raw pixel intensities to the range [0, 1]
    data_test = np.array(data_test)

    ## Normalise the data
    data = np.array(data) / 255.0
    data_test = np.array(data_test) / 255.0



    # Preprocess the inputs
    x_train = data
    x_test = data_test

    ## Obtain the intermediate output
    layer_name = 'dense_1'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)

    layer1 = model.get_layer("dense_output1")
    layer2 = model.get_layer("dense_output2")
    intermediate_output = intermediate_layer_model.predict(x_train)
    X = intermediate_output
    intermediate_output = intermediate_layer_model.predict(x_test)
    X_test = intermediate_output

    print X.shape
    print X_test.shape

    K.clear_session()

    return [X,X_test]




def prepare_usps_data_for_ae2_svdd(data_train,data_test,modelpath):
    ### Declare the training and test paths
    # modelpath = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/trained_models/cifar10_conv_3_id_32_e_1000_encoder.model"
    #
    # train_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/data/cifar-10/normal/"
    # test_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/data/cifar-10/anoma/"
    NB_IV3_LAYERS_TO_FREEZE = 4
    side = 32
    side = 32
    channel = 3

    ## Declare the scoring functions
    g = lambda x: 1 / (1 + tf.exp(-x))

    # g  = lambda x : x # Linear
    def nnScore(X, w, V, g):

        # print "X",X.shape
        # print "w",w[0].shape
        # print "v",V[0].shape
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu(x):
        y = x
        y[y < 0] = 0
        return y

    def add_new_last_layer(base_model_output, base_model_input):
        """Add last layer to the convnet
        Args:
          base_model: keras model excluding top
          nb_classes: # of classes
        Returns:
          new keras model with last layer
        """
        x = base_model_output
        print "base_model.output", x.shape
        inp = base_model_input
        print "base_model.input", inp.shape
        dense1 = Dense(128, name="dense_output1")(x)  # new sigmoid layer
        dense1out = Activation("relu", name="output_activation1")(dense1)
        dense2 = Dense(1, name="dense_output2")(dense1out)  # new sigmoid layer
        dense2out = Activation("relu", name="output_activation2")(dense2)  # new sigmoid layer
        model = Model(inputs=inp, outputs=dense2out)
        return model

    # load the trained convolutional neural network freeze all the weights except for last four layers
    print("[INFO] loading network...")
    model = load_model(modelpath)
    model = add_new_last_layer(model.output, model.input)
    print(model.summary())
    print "Length of model layers...", len(model.layers)

    # Freeze the weights of untill the last four layers
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True


    # Preprocess the inputs
    X_train = data_train
    X_test = data_test
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    print "X_train.shape", X_train.shape
    print "X_test.shape", X_test.shape
    X_train = np.reshape(X_train, (len(X_train), 16, 16, 1))
    X_test = np.reshape(X_test, (len(X_test), 16, 16, 1))

    img_shape = X_train.shape[1:]

    print "Image Shape", img_shape


    ## Obtain the intermediate output
    layer_name = 'dense_2'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)

    layer1 = model.get_layer("dense_output1")
    layer2 = model.get_layer("dense_output2")
    intermediate_output = intermediate_layer_model.predict(X_train)
    X = intermediate_output
    intermediate_output = intermediate_layer_model.predict(X_test)
    X_test = intermediate_output

    print X.shape
    print X_test.shape

    K.clear_session()

    return [X,X_test]



def prepare_usps_data_for_cae_svdd(data_train,data_test,modelpath):
    ### Declare the training and test paths
    # modelpath = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/trained_models/cifar10_conv_3_id_32_e_1000_encoder.model"
    #
    # train_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/data/cifar-10/normal/"
    # test_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/data/cifar-10/anoma/"
    NB_IV3_LAYERS_TO_FREEZE = 4
    side = 32
    side = 32
    channel = 3

    ## Declare the scoring functions
    g = lambda x: 1 / (1 + tf.exp(-x))

    # g  = lambda x : x # Linear
    def nnScore(X, w, V, g):

        # print "X",X.shape
        # print "w",w[0].shape
        # print "v",V[0].shape
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu(x):
        y = x
        y[y < 0] = 0
        return y

    def add_new_last_layer(base_model_output, base_model_input):
        """Add last layer to the convnet
        Args:
          base_model: keras model excluding top
          nb_classes: # of classes
        Returns:
          new keras model with last layer
        """
        x = base_model_output
        print "base_model.output", x.shape
        inp = base_model_input
        print "base_model.input", inp.shape
        dense1 = Dense(128, name="dense_output1")(x)  # new sigmoid layer
        dense1out = Activation("relu", name="output_activation1")(dense1)
        dense2 = Dense(1, name="dense_output2")(dense1out)  # new sigmoid layer
        dense2out = Activation("relu", name="output_activation2")(dense2)  # new sigmoid layer
        model = Model(inputs=inp, outputs=dense2out)
        return model

    # load the trained convolutional neural network freeze all the weights except for last four layers
    print("[INFO] loading network...")
    model = load_model(modelpath)
    model = add_new_last_layer(model.output, model.input)
    print(model.summary())
    print "Length of model layers...", len(model.layers)

    # Freeze the weights of untill the last four layers
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True


    # Preprocess the inputs

    # Preprocess the inputs
    X_train = data_train
    X_test = data_test
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    print "X_train.shape", X_train.shape
    print "X_test.shape", X_test.shape
    X_train = np.reshape(X_train, (len(X_train), 16, 16, 1))
    X_test = np.reshape(X_test, (len(X_test), 16, 16, 1))

    img_shape = X_train.shape[1:]

    print "Image Shape", img_shape

    ## Obtain the intermediate output
    layer_name = 'dense_1'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)

    layer1 = model.get_layer("dense_output1")
    layer2 = model.get_layer("dense_output2")
    intermediate_output = intermediate_layer_model.predict(X_train)
    X = intermediate_output
    intermediate_output = intermediate_layer_model.predict(X_test)
    X_test = intermediate_output


    K.clear_session()

    return [X,X_test]





def prepare_mnist_data_for_ae2_svdd(data_train,data_test,modelpath):
    ### Declare the training and test paths
    # modelpath = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/trained_models/cifar10_conv_3_id_32_e_1000_encoder.model"
    #
    # train_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/data/cifar-10/normal/"
    # test_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/data/cifar-10/anoma/"
    NB_IV3_LAYERS_TO_FREEZE = 4
    side = 32
    side = 32
    channel = 3

    ## Declare the scoring functions
    g = lambda x: 1 / (1 + tf.exp(-x))

    # g  = lambda x : x # Linear
    def nnScore(X, w, V, g):

        # print "X",X.shape
        # print "w",w[0].shape
        # print "v",V[0].shape
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu(x):
        y = x
        y[y < 0] = 0
        return y

    def add_new_last_layer(base_model_output, base_model_input):
        """Add last layer to the convnet
        Args:
          base_model: keras model excluding top
          nb_classes: # of classes
        Returns:
          new keras model with last layer
        """
        x = base_model_output
        print "base_model.output", x.shape
        inp = base_model_input
        print "base_model.input", inp.shape
        dense1 = Dense(128, name="dense_output1")(x)  # new sigmoid layer
        dense1out = Activation("relu", name="output_activation1")(dense1)
        dense2 = Dense(1, name="dense_output2")(dense1out)  # new sigmoid layer
        dense2out = Activation("relu", name="output_activation2")(dense2)  # new sigmoid layer
        model = Model(inputs=inp, outputs=dense2out)
        return model

    # load the trained convolutional neural network freeze all the weights except for last four layers
    print("[INFO] loading network...")
    model = load_model(modelpath)
    model = add_new_last_layer(model.output, model.input)
    print(model.summary())
    print "Length of model layers...", len(model.layers)

    # Freeze the weights of untill the last four layers
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True


    # Preprocess the inputs
    X_train = data_train
    X_test = data_test
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    print "X_train.shape", X_train.shape
    print "X_test.shape", X_test.shape
    X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
    X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))

    img_shape = X_train.shape[1:]

    print "Image Shape", img_shape


    ## Obtain the intermediate output
    layer_name = 'dense_1'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)

    layer1 = model.get_layer("dense_output1")
    layer2 = model.get_layer("dense_output2")
    intermediate_output = intermediate_layer_model.predict(X_train)
    X = intermediate_output
    intermediate_output = intermediate_layer_model.predict(X_test)
    X_test = intermediate_output

    print X.shape
    print X_test.shape

    K.clear_session()

    return [X,X_test]

def prepare_mnist_data_for_cae_svdd(data_train,data_test,modelpath):
    ### Declare the training and test paths
    # modelpath = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/trained_models/cifar10_conv_3_id_32_e_1000_encoder.model"
    #
    # train_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/data/cifar-10/normal/"
    # test_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/data/cifar-10/anoma/"
    NB_IV3_LAYERS_TO_FREEZE = 4
    side = 32
    side = 32
    channel = 3

    ## Declare the scoring functions
    g = lambda x: 1 / (1 + tf.exp(-x))

    # g  = lambda x : x # Linear
    def nnScore(X, w, V, g):

        # print "X",X.shape
        # print "w",w[0].shape
        # print "v",V[0].shape
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu(x):
        y = x
        y[y < 0] = 0
        return y

    def add_new_last_layer(base_model_output, base_model_input):
        """Add last layer to the convnet
        Args:
          base_model: keras model excluding top
          nb_classes: # of classes
        Returns:
          new keras model with last layer
        """
        x = base_model_output
        print "base_model.output", x.shape
        inp = base_model_input
        print "base_model.input", inp.shape
        dense1 = Dense(128, name="dense_output1")(x)  # new sigmoid layer
        dense1out = Activation("relu", name="output_activation1")(dense1)
        dense2 = Dense(1, name="dense_output2")(dense1out)  # new sigmoid layer
        dense2out = Activation("relu", name="output_activation2")(dense2)  # new sigmoid layer
        model = Model(inputs=inp, outputs=dense2out)
        return model

    # load the trained convolutional neural network freeze all the weights except for last four layers
    print("[INFO] loading network...")
    model = load_model(modelpath)
    model = add_new_last_layer(model.output, model.input)
    print(model.summary())
    print "Length of model layers...", len(model.layers)

    # Freeze the weights of untill the last four layers
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True


    # Preprocess the inputs

    # Preprocess the inputs
    X_train = data_train
    X_test = data_test
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    print "X_train.shape", X_train.shape
    print "X_test.shape", X_test.shape
    X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
    X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))

    img_shape = X_train.shape[1:]

    print "Image Shape", img_shape

    ## Obtain the intermediate output
    layer_name = 'dense_1'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)

    layer1 = model.get_layer("dense_output1")
    layer2 = model.get_layer("dense_output2")
    intermediate_output = intermediate_layer_model.predict(X_train)
    X = intermediate_output
    intermediate_output = intermediate_layer_model.predict(X_test)
    X_test = intermediate_output


    K.clear_session()

    return [X,X_test]




