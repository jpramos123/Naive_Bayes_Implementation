import numpy as np

#TODO:
"""
Mepear e armazenar as features (colunas) em indices unicos
Para cada feature, mepear e armazenar os diferentes tipos de dados e realizar a contagem comparando com o target (Quantos em Yes e quantos em No)
Criar dicionarios com os mapeamentos?

features = {0: {"sunny" : 0, "overcast": 1, "rain": 2},
            1: {"hot": 0, "mild": 1, "cool": 2} 
}
"""


class NaiveBayes:

    def __init__(self, features, targets, model_type):
        self.features = features
        self.targets = targets
        self.inverted_targets = {}
        self.features_mapping = {}
        self.targets_mapping = {}
        self.targets_amounts = {}
        self.learn_matrix = []
        self.type = model_type
        self.classes_mean = {}
        self.classes_deviation = {}
        self.features_mean = {}
        self.features_deviation = {}

    def create_feature_dict(self):

        features_arr = [[] for i in range(len(self.features[0]))]
        features_dict = {}

        for i in range(len(self.features[0])):
            for j in range(len(self.features)):
                if self.features[j][i] not in features_arr[i]:
                    features_arr[i].append(self.features[j][i])

        for i in range(len(features_arr)):
            features_dict[i] = {}
            for j in range(len(features_arr[i])):
                features_dict[i][features_arr[i][j]] = j

        
        self.features_mapping = features_dict
    
    def create_targets_dict(self):

        targets_dict = {}
        inverted_targets = {}
        targets_amount = {}
        
        for i in range(len(np.unique(self.targets))):
            targets_dict[np.unique(self.targets)[i]] = i
        
        for i in range(len(np.unique(self.targets))):
            inverted_targets[i] = np.unique(self.targets)[i]

        for t in targets_dict.keys():
            targets_amount[t] = len(np.where(self.targets == t)[0])
            
        
        self.targets_mapping = targets_dict
        self.inverted_targets = inverted_targets
        self.targets_amounts = targets_amount

    def build_learn_table_structure(self):

        learn_matrix = [[[0 for k in range(len(self.features_mapping[i].keys()))] for j in range(len(self.targets_mapping.keys()))] for i in range(len(self.features_mapping.keys()))]
        
        self.learn_matrix = learn_matrix

    def learn(self):
        
        for i in range(len(self.features)):
            for j in range(len(self.features[i])):
                self.learn_matrix[j][self.targets_mapping[self.targets[i]]][self.features_mapping[j][self.features[i][j]]] += 1

        for i in range(len(self.learn_matrix)):
           for j in range(len(self.learn_matrix[i])):
               self.learn_matrix[i][j] = np.array(self.learn_matrix[i][j])/len(np.where(self.targets == self.inverted_targets[j])[0])

    def fit(self):


        if self.type == 'nc':
            self.create_feature_dict()
            self.create_targets_dict()
            self.build_learn_table_structure()
            self.learn()

        elif self.type == 'c':
            self.create_targets_dict()
            self.means()
            self.standard_deviation()
        else:
            print("Method not available")

    def means(self):
        
        # Features Mean
        for i in range(self.features.shape[1]):
            self.features_mean[i] = np.mean(self.features[:,i])

        # Targets Mean        
        for i in range(len(self.features)):
            if self.targets[i] not in self.classes_mean.keys():
                self.classes_mean[self.targets[i]] = np.sum(self.features[i])
            else:
                self.classes_mean[self.targets[i]] += np.sum(self.features[i])
    
        for i in self.classes_mean.keys():
            self.classes_mean[i] = self.classes_mean[i]/len(np.where(self.targets == i)[0])
        
        #print(self.classes_mean)

    def standard_deviation(self):

        # Features Standard Deviation
        for i in range(self.features.shape[1]):
            self.features_deviation[i] = np.std(self.features[:,i])
            
        # Targets Standard Deviation

        for i in range(len(self.features)):
            if self.targets[i] not in self.classes_deviation.keys():
                self.classes_deviation[self.targets[i]] = [self.features[i]]
            else:
                self.classes_deviation[self.targets[i]].append(self.features[i])
        
        #print(self.classes_deviation)
        #print(self.classes_deviation[0])
        for i in self.classes_deviation.keys():
            self.classes_deviation[i] = np.std(self.classes_deviation[i])
            
        #print(self.classes_deviation)

    def predict(self, arr_input):

        if self.type == 'nc':
            result = [1 for _ in self.inverted_targets.keys()]
            for targets_idx in self.inverted_targets:
                for f in range(len(self.learn_matrix)):
                    result[targets_idx] = result[targets_idx] * self.learn_matrix[f][targets_idx][self.features_mapping[f][arr_input[f]]]
                result[targets_idx] = result[targets_idx] * self.targets_amounts[self.inverted_targets[targets_idx]]/len(self.targets)
            
            max_res = 0
            max_idx = 0
            for i in range(len(result)):
                if result[i] > max_res:
                    max_res = result[i]
                    max_idx = i
            
            print("Probability: ", round(max_res, 4), "\nResult: ", self.inverted_targets[max_idx])
        
        elif self.type == 'c':

            result = [1 for _ in self.targets_mapping.keys()]

            for i in range(len(self.targets_mapping.keys())):
                for j in range(len(arr_input)):
                    result[i] = result[i] * (1/((self.classes_mean[i])*np.sqrt(2*np.pi))*np.exp(-((arr_input[j])**2)) / 2*self.classes_deviation[i]**2)

            max_res = 0
            max_idx = 0
            for i in range(len(result)):
                if result[i] > max_res:
                    max_res = result[i]
                    max_idx = i
            
            print("Probability: ", round(max_res, 4), "\nResult: ", self.inverted_targets[max_idx])