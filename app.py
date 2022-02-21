from flask import Flask, render_template, request, redirect, send_file
from werkzeug.utils import secure_filename
import pickle
import mapper
import pandas as pd
import small_text_convert
import pre_trained_model
from sklearn.metrics import accuracy_score
from sklearn import metrics 
from small_text.active_learner import PoolBasedActiveLearner

app = Flask(__name__, template_folder='templates')
app.config["UPLOAD_FOLDER"] = "uploadedfiles/"
app.config["DOWNLOAD_FOLDER"] = "downloadedfiles/"
app.config["LOG_FOLDER"] = "log/"

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

#load model using pickle
#model = pickle.load(open('nowal.pkl', 'rb'))
model = PoolBasedActiveLearner.load('nowal.pkl')

def log_file(active_learner, test, f):
    y_pred_test = active_learner.classifier.predict(test)
    y_pred_test_prob = active_learner.classifier.predict_proba(test)  
    test_acc = accuracy_score(y_pred_test, test.y)
    # fpr, tpr, th = metrics.roc_curve(test.y, y_pred_test_prob[:,1])
    # test_auc = metrics.auc(fpr, tpr)
    #log_loss = metrics.log_loss(test.y, y_pred_test_prob)

    f.write("\n############### Test Classifier ########################\n")
    f.write(metrics.classification_report(test.y, y_pred_test))
    f.write("\n############### Test Confusion Matrix ##################\n")
    cm = metrics.confusion_matrix(test.y, y_pred_test)
    f.write(str(cm))
    f.write("\nTest accuracy: ")
    f.write(str(test_acc))
    # f.write("\nTest AUC: ")
    # f.write(str(test_auc))
    #f.write("\nLoss Fun: {}".format(str(log_loss)))
    # f.write("\nfptr: {}  tpr: {}   threshold: {}  Loss Fun: {}".format(
    #     str(fpr),str(tpr),str(th),str(log_loss)))

    f.write("\n############### Test Data ########################\n")
    f.write(str(y_pred_test))
    f.write("\n############### Test Data Prob ########################\n")
    f.write(str(y_pred_test_prob))

def file_predict(file_path, filename):
    #sentence = pd.DataFrame()
    df = pd.read_csv(file_path + filename)
    sentence_list = df["Questions"].to_list()
    ptm = pre_trained_model.PreTrainedModel()
    ptm.set_tokenizer('bert-base-uncased')
    tokenizer = ptm.get_set_tokenizer()
    label = [999] * len(sentence_list)
    #label = df["Intents"].to_list()
    converted_sentence = small_text_convert.get_transformers_dataset(
                                                            tokenizer, 
                                                            sentence_list, 
                                                            label)
    prediction = model.classifier.predict(converted_sentence)
    #df["Predicted_Intents_en"] = tokenizer.decode(prediction)
    labeled_prediction = []
    for i in prediction:
        labeled_prediction.append(mapper.label_names[i])
    df["Predicted_Intents"] = prediction
    df["Predicted_Intents_eng"] = labeled_prediction
    acc = accuracy_score(df["Predicted_Intents"].to_list(),df["Intents"].to_list())
    df.to_csv(app.config["DOWNLOAD_FOLDER"] + filename, index = False)
    f = open(app.config["LOG_FOLDER"]+"log.txt","w")
    log_file(model, converted_sentence, f)
    del df['Predicted_Intents']
    del df['Intents']
    return (df, acc)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    sentence = [str(x) for x in request.form.values()]
    print(sentence)
    # sentence = ["an edgy thriller that delivers a surprising punch",
    #     "lurid and less than lucid work"]
    ptm = pre_trained_model.PreTrainedModel()
    ptm.set_tokenizer('bert-base-uncased')
    tokenizer = ptm.get_set_tokenizer()
    label = [999] * len(sentence)
    converted_senteance = small_text_convert.get_transformers_dataset(
                                                            tokenizer, 
                                                            sentence, 
                                                            label)
    prediction = model.classifier.predict(converted_senteance)
    return render_template('index.html', 
                prediction_text='The labels for {} is {}'.format(sentence[0], 
                                                                mapper.label_names[prediction[0]]))
@app.route('/upload')
def upload_file1():
   return render_template('upload.html')
	
@app.route('/predictor', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('no file')
            return redirect(request.url)
        f = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if f.filename == '':
            print('no filename')
            return redirect(request.url)
        else:
            filename = secure_filename(f.filename)
            file_path = app.config["UPLOAD_FOLDER"]
            f.save(file_path + filename)
            result = file_predict(file_path, filename)
            send_file(file_path + filename, as_attachment=True, attachment_filename='')
            return render_template('result.html', acc = result[1], 
                        tables=[result[0].to_html(classes='data',
                                                header="true", 
                                                index=False, 
                                                justify="center")])
    return render_template('upload.html')

@app.route('/api')
def build_api():
    filename = "test.csv"
    file_path = app.config["UPLOAD_FOLDER"]
    result = file_predict(file_path, filename)
    send_file(file_path + filename, as_attachment=True, attachment_filename='')
    return render_template('result.html', acc = result[1], 
                tables=[result[0].to_html(classes='data',
                                        header="true", 
                                        index=False, 
                                        justify="center")])
if __name__ == '__main__':
    app.run(debug=True)