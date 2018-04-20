
class Word_Analysis:
    syllable_name = ""
    proba = []
    result = []
    prediction_values = []
    all_label = []

    def __init__(self,syllable_name, proba,result,prediction_values, all_label):
        self.syllable_name = syllable_name
        self.proba = proba
        self.result = result
        self.prediction_values = prediction_values
        self.all_label = all_label