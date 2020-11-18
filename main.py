import numpy as np

from NaiveBayes import NaiveBayes


if __name__ == "__main__":
    
    print(" --- Exemplo dos slides: Tennis Dataset --- ")


    features_tennis = np.array([['sunny', 'hot', 'high', 'weak'],
                ['sunny', 'hot', 'high', 'strong'],
                ['overcast', 'hot', 'high', 'weak'],
                ['rain', 'mild', 'high', 'weak'],
                ['rain', 'cool', 'normal', 'weak'],
                ['rain', 'cool', 'normal', 'strong'],
                ['overcast', 'cool', 'normal', 'strong'],
                ['sunny', 'mild', 'high', 'weak'],
                ['sunny', 'cool', 'normal', 'weak'],
                ['rain', 'mild', 'normal', 'weak'],
                ['sunny', 'mild', 'normal', 'strong'],
                ['overcast', 'mild', 'high', 'strong'],
                ['overcast', 'hot', 'normal', 'weak'],
                ['rain', 'mild', 'high', 'strong']])

    target_tennis = np.array(['No',
                     'No', 
                     'Yes',
                     'Yes',
                     'Yes',
                     'No',
                     'Yes',
                     'No',
                     'Yes',
                     'Yes',
                     'Yes',
                     'Yes',
                     'Yes',
                     'No'
                     ])

    nb = NaiveBayes(features_tennis, target_tennis)
 
    nb.fit()

    nb.predict(['sunny', 'cool', 'high', 'strong'])
    

    print("\n --- Exercicio 1 --- ")

    features_1 = np.array([["<=30", "High", "No", "Fair"],
                           ["<=30", "High", "No", "Excellent" ],
                           ["31...40", "High", "No", "Fair"],
                           [">40", "Medium", "No", "Fair"],
                           [">40", "Low", "Yes", "Fair"],
                           [">40", "Low", "Yes", "Excellent"],
                           ["31...40", "Low", "Yes", "Excellent"],
                           ["<=30", "Medium", "No", "Fair"],
                           ["<=30", "Low", "Yes", "Fair"],
                           [">40", "Medium", "Yes", "Fair"],
                           ["<=30", "Medium", "Yes", "Excellent"],
                           ["31...40", "Medium", "No", "Excellent"],
                           ["31...40", "High", "Yes", "Fair"],
                           [">40", "Medium", "No", "Excellent"],
    ])

    targets_1 = np.array(["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"])

    nb_1 = NaiveBayes(features_1, targets_1)
 
    nb_1.fit()

    print("Age: 31...40, Income: Medium, Student: No, Credit: Excellent")
    nb_1.predict(["31...40", 'Medium', 'No', 'Excellent'])
    
    print("\nAge: <=30, Income: Low, Student: Yes, Credit: Fair")
    nb_1.predict(["<=30", 'Low', 'Yes', 'Fair'])