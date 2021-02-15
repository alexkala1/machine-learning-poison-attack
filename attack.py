from secml.data.loader import CDLRandomBlobs
from secml.data.splitter import CTrainTestSplit
from secml.ml.features import CNormalizerMinMax
from secml.ml.peval.metrics import CMetricAccuracy
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernels import CKernelRBF
from secml.adv.attacks import CAttackPoisoningSVM
from secml.figure import CFigure
from secml.optim.constraints import CConstraintBox

random_state = 999

featuresNumber = 2  # Number of features
samplesNumber = 300  # Number of samples
centers = [[-1, -1], [+1, +1]]  # Centers of the clusters
deviationOfClusters = 0.9  # Standard deviation of the clusters

dataset = CDLRandomBlobs(n_features=featuresNumber,
                         centers=centers,
                         cluster_std=deviationOfClusters,
                         n_samples=samplesNumber,
                         random_state=random_state).load()

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

# First plot
figure = CFigure(4, 5)

grid_limits = [(lowerBound - 0.1, upperBound + 0.1),
               (lowerBound - 0.1, upperBound + 0.1)]

figure.sp.plot_ds(training)

# highlight the initial poisoning sample showing it as a star
figure.sp.plot_ds(training[0, :], markers='*', markersize=16)

figure.sp.title('Attacker objective and gradients')
figure.sp.plot_fun(
    func=poisonAttack.objective_function,
    grid_limits=grid_limits, plot_levels=False,
    n_grid_points=10, colorbar=True)

# plot the box constraint

box = fbox = CConstraintBox(lb=lowerBound, ub=upperBound)
figure.sp.plot_constraint(box, grid_limits=grid_limits,
                          n_grid_points=10)

figure.tight_layout()
figure.show()

# Number of poisoning points to generate
poisoningPointsNumber = 20
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

# Define common bounds for the subplots
min_limit = min(poisonedClassifierTraining.X.min(), test.X.min())
max_limit = max(poisonedClassifierTraining.X.max(), test.X.max())
grid_limits = [[min_limit, max_limit], [min_limit, max_limit]]


# Last plot prints
figure = CFigure(10, 10)

figure.subplot(2, 2, 1)
figure.sp.title("Original classifier (training set)")
figure.sp.plot_decision_regions(
    classifier, n_grid_points=200, grid_limits=grid_limits)
figure.sp.plot_ds(training, markersize=5)
figure.sp.grid(grid_on=False)

figure.subplot(2, 2, 2)
figure.sp.title("Poisoned classifier (training set + poisoning points)")
figure.sp.plot_decision_regions(
    poisonedClassifier, n_grid_points=200, grid_limits=grid_limits)
figure.sp.plot_ds(training, markersize=5)
figure.sp.plot_ds(poisoningPoints, markers=['*', '*'], markersize=12)
figure.sp.grid(grid_on=False)

figure.subplot(2, 2, 3)
figure.sp.title("Original classifier (test set)")
figure.sp.plot_decision_regions(
    classifier, n_grid_points=200, grid_limits=grid_limits)
figure.sp.plot_ds(test, markersize=5)
figure.sp.text(0.05, -0.25, "Accuracy on test set: {:.2%}".format(originalClassifierAccuracy),
               bbox=dict(facecolor='white'))
figure.sp.grid(grid_on=False)

figure.subplot(2, 2, 4)
figure.sp.title("Poisoned classifier (test set)")
figure.sp.plot_decision_regions(
    poisonedClassifier, n_grid_points=200, grid_limits=grid_limits)
figure.sp.plot_ds(test, markersize=5)
figure.sp.text(0.05, -0.25, "Accuracy on test set: {:.2%}".format(poisonedAccuracy),
               bbox=dict(facecolor='white'))
figure.sp.grid(grid_on=False)

figure.show()
