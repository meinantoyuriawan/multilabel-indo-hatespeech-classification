from crypt import methods
from flask import Flask
import dill as pickle

from flask import request

app = Flask(__name__)
def importModel():
    # just need change model directory
    modelA = pickle.load(open('model/model_abusive.sav', 'rb'))
    modelC = pickle.load(open('model/model_Category.sav', 'rb'))
    modelT = pickle.load(open('model/model_Target.sav', 'rb'))
    modelL = pickle.load(open('model/model_Level.sav', 'rb'))
    return modelA, modelC, modelT, modelL

def importVector():
    # just need chane vectorizer directory
    vecA = pickle.load(open('vectorization/vec_abusive.pickle', 'rb'))
    vecC = pickle.load(open('vectorization/vec_Category.pickle', 'rb'))
    vecT = pickle.load(open('vectorization/vec_Target.pickle', 'rb'))
    vecL = pickle.load(open('vectorization/vec_Level.pickle', 'rb'))
    return vecA, vecC, vecT, vecL

def predict_abusive(model, vec):
    output = model.predict_proba(vec)
    hs = output.toarray()[0][0]
    ab = output.toarray()[0][1]
    hs = "{:.3f}".format(hs)
    ab = "{:.3f}".format(ab)
    # print('Hate Speech: ', hs)
    # print('Abusive: ', ab)
    return hs, ab

def predict_category(model, vec):
    output = model.predict_proba(vec)
    religion = output.toarray()[0][0]
    race = output.toarray()[0][1]
    physical = output.toarray()[0][2]
    gender = output.toarray()[0][3]
    other = output.toarray()[0][4]

    religion = "{:.3f}".format(religion)
    race = "{:.3f}".format(race)
    physical = "{:.3f}".format(physical)
    gender = "{:.3f}".format(gender)
    other = "{:.3f}".format(other)

    # print('Religion: ', religion)
    # print('Race: ', race)
    # print('Physical: ', physical)
    # print('Gender: ', gender)
    # print('Other: ', other)

    return religion, race, physical, gender, other

def predict_target(model, vec):
    output = model.predict_proba(vec)
    ind = output.toarray()[0][0]
    gr = output.toarray()[0][1]

    ind = "{:.3f}".format(ind)
    gr = "{:.3f}".format(gr)
    
    # print('Individual: ', ind)
    # print('Group: ', gr)

    return ind, gr

def predict_level(model, vec):
    output = model.predict_proba(vec)
    weak = output.toarray()[0][0]
    moderate = output.toarray()[0][1]
    strong = output.toarray()[0][2]
    weak = "{:.3f}".format(weak)
    moderate = "{:.3f}".format(moderate)
    strong = "{:.3f}".format(strong)    

    # print('Weak: ', weak)
    # print('Moderate: ', moderate)
    # print('Strong: ', strong)

    return weak, moderate, strong

@app.route('/input', methods=['POST'])
def predictAll():

    feature_list = request.form.to_dict()
    text = [list(feature_list.values())[0]]
    print(text)
    modelA, modelC, modelT, modelL = importModel()
    vecA, vecC, vecT, vecL = importVector()

    resA = vecA.transform([text[0]])
    resC = vecC.transform([text[0]])
    resT = vecT.transform([text[0]])
    resL = vecL.transform([text[0]])
    hs, ab = predict_abusive(modelA, resA)
    religion, race, physical, gender, other = predict_category(modelC, resC)
    ind, gr = predict_target(modelT, resT)
    weak, moderate, strong = predict_level(modelL, resL)

    # new_segments = []
    obj_segments = {
        'Hate Speech': hs,
        'Abusive' : ab,
        'Religion' : religion,
        'Race' : race,
        'Physical' : physical,
        'Gender' : gender,
        'Other' : other,
        'Individual' : ind,
        'Group' : gr,
        'Weak' : weak,
        'Moderate' : moderate,
        'Strong' : strong,
    }

    
    return obj_segments