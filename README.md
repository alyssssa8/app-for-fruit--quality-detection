# app-for-fruit--quality-detection
python, java, tensor flow

more details at project 3 summary.docx 

videos for the project:

    https://www.youtube.com/watch?v=bq43vdvVWnc
    
    https://www.youtube.com/watch?v=bClPtZGUh-s

 A describing how code is structured and the state of how it works. Give a description for each filename listed. 
•	In Android Project:  The whole project is based on Google’s TensorFlow Android example. The main source files are ClassifierActivity.java and TensorFlowImageClassifier.java; the main model files are mytest.pb and mylabel.txt.
o	ClassifierActivity.java: Activity class. Initialized the tensorflow utility. Communicate with tensorflow utility. Handle result with TTS (Text-to-Speech)engine. 
o	TensorFlowImageClassifier.java: the tensorflow utility. Communicate with the real tensorflow library. Prepare data and get result from the tensorflow library.
o	mytest.pb: model file used for tensorflow library.
o	mylabel.txt: dictionary file to translate number to the label.
•	In Python Project: The training part is based on the TensorFlow example MNIST at https://www.tensorflow.org/get_started/mnist/pros . The banana images are from http://image-net.org/ . There are two main source files in the project: TrainerFruitGrade.py and two_layer_fc.py.
o	TrainerFruitGrade.py: load images, build the network, save models.
o	Two_layer_fc.py: implementation of the layers for the neural network.
