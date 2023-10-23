#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

np.random.seed(42)

x = np.random.rand(1000)*3

moy = np.mean(x)
moy = round(moy,2)
moyStr = str(moy)
print("la moyenne de x est "+ moyStr)
ecaTyp = np.std(x)
ecaTyp = round(ecaTyp,2)
ecaTypStr = str(ecaTyp)
print("l'écart type de x est "+ ecaTypStr)
med = np.median(x)
med = round(med,2)
medStr = str(med)
print("la médiane de x est "+ medStr)

x_bis = np.random.rand(1000)*3

moy = np.mean(x_bis)
moy = round(moy,2)
moyStr = str(moy)
print("la moyenne de x_bis est "+ moyStr)
ecaTyp = np.std(x_bis)
ecaTyp = round(ecaTyp,2)
ecaTypStr = str(ecaTyp)
print("l'écart type de x_bis est "+ ecaTypStr)
med = np.median(x_bis)
med = round(med,2)
medStr = str(med)
print("la médiane de x_bis est "+ medStr)


y = np.sin(x)
noise = np.random.randn(1000)*0.1

y =+ noise

plt.figure(figsize=(20,15))
plt.scatter(x, y)
plt.show()
plt.hist(noise, bins=50)
plt.show()

bike_folder = "./data1/computer_vision_tp1/data1/bike"
elements_bike = os.listdir(bike_folder)
num_bike = len(elements_bike)
print(num_bike)
car_folder = "./data1/computer_vision_tp1/data1/car"
elements_car = os.listdir(car_folder)
num_car = len(elements_car)
print(num_car)
num_total = num_bike + num_car
print("le nombre de fichier total est " + str(num_total))

image = img.imread("./data1/computer_vision_tp1/data1/bike/Bike (1).png")
plt.imshow(image[:,:,1], cmap='gray', origin='lower')
plt.show()

target_size = (224,224)

def peuplate_images_and_labels_lists(image_folder_path):
    images= []
    labels = []
    for filename in os.listdir(image_folder_path):
        image = cv2.imread(os.path.join(image_folder_path, filename))

        image = cv2.resize(image,target_size)
        images.append(image)
        labels.append(filename.split(" ")[0])
    return images,labels

    
images_bike, labels_bike = peuplate_images_and_labels_lists(bike_folder)
images_car, labels_car = peuplate_images_and_labels_lists(car_folder)

images = np.array(images_bike+images_car)
labels = np.array(labels_bike+labels_car)

images = np.array([image.flatten() for image in images])
print(str(len(images)))
x_train, x_test, y_train, y_test = train_test_split(images, labels,
test_size=0.2, random_state=0)


#Partie 3

#a) II.
clf_model = DecisionTreeClassifier(random_state=0)

#a) III.
clf_model.fit(x_train,y_train)
#a) IV.
prediction_clt = clf_model.predict([x_test[0]])

#b)
svm_model = SVC(random_state=0)
svm_model.fit(x_train, y_train)
prediction_svm = svm_model.predict([x_test[0]])

#c) I.
# Prédictions du modèle DecisionTreeClassifier
print(x_test.ndim)
accuracy_clt = accuracy_score(y_test, clf_model.predict(x_test))

# Prédictions du modèle SVC
accuracy_svm = accuracy_score(y_test, clf_model.predict(x_test))

#c) II.
# Matrice de confusion du modèle DecisionTreeClassifier
confusion_matrix_clf = confusion_matrix(y_test, clf_model.predict(x_test))
print(confusion_matrix_clf)

# Matrice de confusion du modèle SVC
confusion_matrix_svm = confusion_matrix(y_test, svm_model.predict(x_test))
print(confusion_matrix_svm)

#Partie 4
#a) a.
tree_size = clf_model.get_depth()
tree_size = str(tree_size)
print("la profondeur de l'arbre de décision est à :"+ tree_size)

#a) b. I.
max_depth_list = list(range(1, 13))

#a) b. II.
train_accuracy = []
test_accuracy = []

for depth in max_depth_list:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
    clf.fit(x_train, y_train)  # Entraînement du modèle
    train_pred = clf.predict(x_train)  # Prédictions sur le set d'entraînement
    test_pred = clf.predict(x_test)  # Prédictions sur le set de test
    train_acc = accuracy_score(y_train, train_pred)  # Exactitude du set d'entraînement
    test_acc = accuracy_score(y_test, test_pred)  # Exactitude du set de test
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)
    depth_str=str(depth)
    print("profondeur : " +depth_str+" terminé")
print("end")
#%%
for depth_len in test_accuracy:
    depth_str=str(depth_len)
    print(depth_str)

plt.plot(max_depth_list, train_accuracy, label='Train Accuracy', marker='o')
plt.plot(max_depth_list, test_accuracy, label='Test Accuracy', marker='o')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs. max_depth')
plt.show()
#%%