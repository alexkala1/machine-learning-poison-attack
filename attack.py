from secml.data import CDataset
from secml.data.loader import CDLRandomBlobs
from secml.data.splitter import CTrainTestSplit
from secml.ml.features import CNormalizerMinMax
from secml.ml.peval.metrics import CMetricAccuracy
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernels import CKernelRBF
from secml.adv.attacks import CAttackPoisoningSVM
from dataset1 import read_data


random_state = 999

featuresNumber = 2  # Number of features
samplesNumber = 300  # Number of samples
centers = [[-1, -1], [+1, +1]]  # Centers of the clusters
deviationOfClusters = 0.9  # Standard deviation of the clusters

# dataset = CDLRandomBlobs(n_features=featuresNumber,
#                          centers=centers,
#                          cluster_std=deviationOfClusters,
#                          n_samples=samplesNumber,
#                          random_state=random_state).load()

incdataset = read_data()

dataset = CDataset(incdataset[0], incdataset[1])

setSamplesTrainingNumber = 100  # Number of training set samples
setSamplesValidationNumber = 100  # Number of validation set samples
setSampleTestNumber = 100  # Number of test set samples

# Split in training, validation and test

splitter = CTrainTestSplit(
    train_size=setSamplesTrainingNumber + setSamplesValidationNumber, test_size=setSampleTestNumber,
    random_state=random_state)

trainingValidation, test = splitter.split(dataset)

splitter = CTrainTestSplit(
    train_size=setSamplesTrainingNumber, test_size=setSamplesValidationNumber, random_state=random_state)

training, validation = splitter.split(dataset)

# Normalize the data
normalizer = CNormalizerMinMax()
training.X = normalizer.fit_transform(training.X)
validation.X = normalizer.transform(validation.X)
test.X = normalizer.transform(test.X)

# Metric to use for training and performance evaluation
metric = CMetricAccuracy()

# Creation of the multiclass classifier
classifier = CClassifierSVM(kernel=CKernelRBF(gamma=10), C=1)

# We can now fit the classifier
classifier.fit(training.X, training.Y)
print("Training of classifier complete!")

# Compute predictions on a test set
predictionY = classifier.predict(test.X)

# Bounds of the attack space. Can be set to `None` for unbounded
lowerBound, upperBound = validation.X.min(), validation.X.max()

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 0.05,
    'eta_min': 0.05,
    'eta_max': None,
    'max_iter': 100,
    'eps': 1e-6
}

poisonAttack = CAttackPoisoningSVM(classifier=classifier,
                                   training_data=training,
                                   val=validation,
                                   lb=lowerBound, ub=upperBound,
                                   solver_params=solver_params,
                                   random_seed=random_state)

# chose and set the initial poisoning sample features and label
choiceX = training[0, :].X
choiceY = training[0, :].Y
poisonAttack.x0 = choiceX
poisonAttack.xc = choiceX
poisonAttack.yc = choiceY

print("Initial poisoning sample features: {:}".format(choiceX.ravel()))
print("Initial poisoning sample label: {:}".format(choiceY.item()))

# Number of poisoning points to generate
poisoningPointsNumber = 20
poisonAttack.n_points = poisoningPointsNumber

# Run the poisoning attack
print("Attack started...")
poisonYPrediction, poisonScores, poisoningPoints, f_opt = poisonAttack.run(test.X, test.Y)
print("Attack complete!")

# Number of poisoning points to generate
poisoningPointsNumber = 40
poisonAttack.n_points = poisoningPointsNumber

# Run the poisoning attack
print("Attack started...")
poisonYPrediction, poisonScores, poisoningPoints, f_opt = poisonAttack.run(test.X, test.Y)
print("Attack complete!")

# Evaluate the accuracy of the original classifier
originalClassifierAccuracy = metric.performance_score(y_true=test.Y, y_pred=predictionY)

# Evaluate the accuracy after the poisoning attack
poisonedAccuracy = metric.performance_score(y_true=test.Y, y_pred=poisonYPrediction)

print("Original accuracy on test set: {:.2%}".format(originalClassifierAccuracy))
print("Accuracy after attack on test set: {:.2%}".format(poisonedAccuracy))

# Training of the poisoned classifier
poisonedClassifier = classifier.deepcopy()
poisonedClassifierTraining = training.append(poisoningPoints)  # Join the training set with the poisoning points
poisonedClassifier.fit(poisonedClassifierTraining.X, poisonedClassifierTraining.Y)

