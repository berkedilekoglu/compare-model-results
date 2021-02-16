import numpy as np
import matplotlib.pyplot as plt


class comparison:

    def __init__(self,X_test,y_test):

        
        self.X_test = X_test
        self.y_test = y_test
        self.predictions_dict = {"True Labels":{"predictions": self.y_test,"threshold": 0.5}}
        self.labels_dict = {"True Labels":{"labels": self.y_test,"x_all":0, "y_all":0,"true_x":0,"true_y":0}}
        self.figSize_x = 20
        self.figSize_y = 10
        self.figureName = 'Comparison of Predictions'
        self.bottomSpace = None
        self.topSpace = None
        self.hspace = 0.2
        self.wspace = None


        self.set_x_y_all(self.y_test,"True Labels")

    def set_figSize(self,x,y):
        """This function gets width and height to set figure size"""

        self.figSize_x = x
        self.figSize_y = y

    def set_figName(self,name):
        """This function get a name to set Main Plot Name"""

        self.figureName = name

    def set_spaces(self,bottomSpace = None,topSpace = None,hspace = 0.2,wspace = None):
        """This function get variables to set subplot spaces
        
        bottomSpace  # the bottom of the subplots of the figure
        topSpace     # the top of the subplots of the figure
        wspace      # the amount of width reserved for space between subplots,expressed as a fraction of the average axis width
        hspace      # the amount of height reserved for space between subplots,expressed as a fraction of the average axis height
        
        """
        self.bottomSpace = bottomSpace
        self.topSpace = topSpace
        self.hspace = hspace
        self.wspace = wspace

    
    def update(self):
        self.labels_dict["True Labels"]["labels"] = self.y_test
        self.predictions_dict["True Labels"]["labels"] = self.y_test
        self.find_true_index_predictedLabels("True Labels")

    def oneHot_to_integer(self): 
        """If your labels are one hot encoded use that function
        Basicly from [[0,1],[1,0]] -> [1,0]"""
        self.y_test = [np.where(r==1)[0][0] for r in self.y_test]
    
    def order_test_samples(self): 
    
        """This function for ordering indexes of positive and negative test examples.
        It helps us to get more clear illustration for predictions
        Use output of that function for your prediction"""

        #unique_elements, counts_elements = np.unique(y, return_counts=True)
        negative_indexes = list(np.where(self.y_test==0)[0])
        positive_indexes = list(np.where(self.y_test==1)[0])
        
        positive_samples = self.X_test[positive_indexes]
        negative_samples = self.X_test[negative_indexes]
        
        
        negative_labels = np.zeros((len(negative_indexes)))
        positive_labels = np.ones((len(positive_indexes)))
        
        self.y_test = np.concatenate([positive_labels,negative_labels])
        self.X_test = np.concatenate([positive_samples,negative_samples],axis=0)
        self.update()
        

        return self.X_test, self.y_test

    def set_x_y_all(self,y,modelName):
        """This function set x and y arrays for creating a black background space in our plot"""
        y_position = list(range(len(y)))
        x_position = np.ones((len(y)))

        self.labels_dict[modelName]["y_all"] = y_position
        self.labels_dict[modelName]["x_all"] = x_position

    def predicted_labels(self, y_probs,threshold):
        """This function takes probabilities and threshold as inputs
        Determine labels by using threshold"""

        labels = np.zeros((len(y_probs))) #Create a zero array, thus we can look at probabilities for 1's
        
        for index in range(len(y_probs)):
            if y_probs[index][1] >= threshold: #Look at probs for 1's. If prob is larger than threshold predict it as 1.
                labels[index] = 1
        
        return labels

    def arrenge_x_axes(self,true_index):
        """This function determines hight of the true predictions -> 1s"""
        return np.ones((len(true_index)))

    def find_true_index_predictedLabels(self,modelName):

        """This function determines indexes of 1 in our predictions"""

        y_pred = self.labels_dict[modelName]["labels"]

        true_index = []
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                true_index.append(i)
        
        true_x = self.arrenge_x_axes(true_index)

        self.labels_dict[modelName]["true_y"] = true_index
        self.labels_dict[modelName]["true_x"] = true_x


    def set_prob_predictions(self,modelName,preds,threshold=0.5):

        """Each prediction will be saved in dictionary, thus we can use them later.
        This function also set all necessary indexes for plotting step"""

        self.predictions_dict[modelName] = {"predictions": preds,"threshold": threshold}
        

        pred_labels = self.predicted_labels(preds,threshold)

        self.set_label_predictions(modelName,pred_labels)
        self.set_x_y_all(pred_labels,modelName)
        self.find_true_index_predictedLabels(modelName)
    
    def set_label_predictions(self,modelName,labels):

        """Label version of set_prob_predictions function. """

        self.labels_dict[modelName] = {"labels": labels,"x_all":0, "y_all":0,"true_x":0,"true_y":0}
        self.set_x_y_all(labels,modelName)
        self.find_true_index_predictedLabels(modelName)

    """def set_model_threshold(self,modelName,threshold):
        
        self.predictions_dict[modelName]["threshold"] = threshold

        pred_labels = self.predicted_labels(self.predictions_dict[modelName]["predictions"],threshold)

        self.set_label_predictions(modelName,pred_labels)
        self.set_x_y_all(pred_labels,modelName)
        self.find_true_index_predictedLabels(modelName)"""

    
    def clear_all(self):

        """This function can be called to erase all instances for dictionaries"""

        print("Saved Predictions will be cleaned!")
        self.labels_dict.clear()
        self.predictions_dict.clear()
        print("Cleaning was done!")

    def delete_element(self,modelName):

        """This function deletes dictionary elements with respect to model name input"""

        if modelName not in self.labels_dict and modelName not in self.predictions_dict:
            raise Exception(f"{modelName} is not an element of any dictionary!")
        else:
            print(f"Saved Predictions for model {modelName} will be cleaned!")
            if modelName in self.labels_dict:
                self.labels_dict.pop(modelName)
            
            if modelName in self.predictions_dict:
                self.predictions_dict.pop(modelName)
            print("Cleaning was done!")



    def compare_3_prediction(self,modelName1,modelName2,modelName3):

        """If you want to take detailed report for comparison of 3 models this helper function will be called in compare_predictions
        It takes names of the 3 models and examine common predictions
        Common predictions and mistakes are important
        Individual mistakes can be exported from that report for further investigation to use Voting
        """
        predicted_labels1 = self.labels_dict[modelName1]["labels"]
        predicted_labels2 = self.labels_dict[modelName2]["labels"]
        predicted_labels3 = self.labels_dict[modelName3]["labels"]

        correct_labels1 = np.where(self.y_test == predicted_labels1)[0]
        incorrect_labels1 = np.where(self.y_test != predicted_labels1)[0]

        correct_labels2 = np.where(self.y_test == predicted_labels2)[0]
        incorrect_labels2 = np.where(self.y_test != predicted_labels2)[0]

        correct_labels3 = np.where(self.y_test == predicted_labels3)[0]
        incorrect_labels3 = np.where(self.y_test != predicted_labels3)[0]

        same_correct1_2 = np.intersect1d(correct_labels1, correct_labels2)
        same_correct_123 = np.intersect1d(same_correct1_2 , correct_labels3)

        same_incorrect1_2 = np.intersect1d(incorrect_labels1 , incorrect_labels2)
        same_incorrect_123 = np.intersect1d(same_incorrect1_2 , incorrect_labels3)

        common_predictions_len = len(same_incorrect_123) + len(same_correct_123)

        print(f"{len(same_correct_123)} common samples were correctly predicted by Three predictor")
        print(f"{len(same_incorrect_123)} common samples were wrongly predicted by Three predictor")
        print(f"{len(predicted_labels1)-common_predictions_len} samples were predicted differently")


    def compare_2_prediction(self,modelName1,modelName2):
        """If you want to take detailed report for comparison of 2 models this helper function will be called in compare_predictions
        It takes names of the 2 models and examine common predictions
        Common predictions and mistakes are important
        Individual mistakes can be exported from that report for further investigation to use Voting
        """
        predicted_labels1 = self.labels_dict[modelName1]["labels"]
        predicted_labels2 = self.labels_dict[modelName2]["labels"]

        correct_labels1 = np.where(self.y_test == predicted_labels1)[0]
        incorrect_labels1 = np.where(self.y_test != predicted_labels1)[0]

        correct_labels2 = np.where(self.y_test == predicted_labels2)[0]
        incorrect_labels2 = np.where(self.y_test != predicted_labels2)[0]

        same_correct = np.isin(correct_labels1, correct_labels2)
        common_correct = correct_labels1[same_correct]

        same_incorrect = np.isin(incorrect_labels1, incorrect_labels2)
        common_incorrect = incorrect_labels1[same_incorrect]

        common_predictions_len = len(common_incorrect) + len(common_correct)

        print(f"{len(common_correct)} common samples were correctly predicted by both predictor")
        print(f"{len(common_incorrect)} common samples were wrongly predicted by both predictor")
        print(f"{len(predicted_labels1)-common_predictions_len} samples were predicted differently")


    def compare_with_golds(self,modelName):
        """This function compares individual models with gold standarts"""

        predicted_labels = self.labels_dict[modelName]["labels"]

        true_predictions = np.where(self.y_test == predicted_labels)[0]
        false_predictions = np.where(self.y_test != predicted_labels)[0]

        print(f"{len(true_predictions)} samples were correctly predicted and {len(false_predictions)} samples were falsely predicted out of {len(self.y_test)} samples by Model: {modelName}")


    def compare_predictions(self,modelName1=None,modelName2=None,modelName3=None):
        """If you want to take detailed explanation of comparison you can use that function.
        This function take 1 to 3 models
        Each model will be compared with Gold Standarts
        Each model compared with each other to find individual mistakes
        After that Voting can be applied
        """
        if modelName1 != None:

            if modelName1 not in self.labels_dict:
                raise Exception(f"{modelName1} is not an element of any dictionary!")
            else:
                counts_elements = np.unique(self.y_test, return_counts=True)[1]
                print(f"There are {counts_elements[0]} negative and {counts_elements[1]} positive samples in labels")

                self.compare_with_golds(modelName1)

            if modelName2 != None and modelName3 == None:

                if modelName2 not in self.labels_dict:
                    raise Exception(f"{modelName2} is not an element of any dictionary!")
                else:
                    self.compare_with_golds(modelName2)
                    self.compare_2_prediction(modelName1,modelName2)


            if modelName3 != None:

                if modelName3 not in self.labels_dict:
                    raise Exception(f"{modelName3} is not an element of any dictionary!")
                else:
                    self.compare_with_golds(modelName2)
                    self.compare_with_golds(modelName3)
                    self.compare_3_prediction(modelName1,modelName2,modelName3)

    
    def plot_predictions(self):

        """To plot predictions of each model and Gold Standarts use that function"""

        model_numbers = len(self.labels_dict)

        fig, ax = plt.subplots(model_numbers,figsize=(self.figSize_x, self.figSize_y))
        fig.suptitle(self.figureName)
        
        model_index = 0
        for model_name,values in self.labels_dict.items(): #Create subplots
            
            X_all = values["x_all"]
            y_all = values["y_all"]

            X_true = values["true_x"]
            y_true = values["true_y"]
            
            ax[model_index].bar(y_all,X_all,width=1,color='black')
            ax[model_index].bar(y_true,X_true,width=1,color='#DAA520')
            ax[model_index].set_title(model_name)
                    
            model_index += 1

        hspace_ = self.hspace + model_index*0.1 #Arrange space between submodels

        plt.subplots_adjust(bottom=self.bottomSpace, top=self.topSpace, hspace=hspace_ , wspace=self.wspace)
        plt.show()