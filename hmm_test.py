import numpy as np
import joblib
import pickle

with open('model.pickle', 'rb') as f:
    hmm_model = pickle.load(f)
 
with open('labels.pickle','rb') as f:
    testing_labels = pickle.load(f)
    

test_data = np.load('test_features.npy', allow_pickle=True)
print(type(test_data))

print(test_data.shape)
# print(test_data[0])

# Compute the log-likelihood of each test sequence using the model
log_likelihoods = []
for seq in test_data:
    log_likelihoods.append(hmm_model.score(seq.reshape(-1, 512)))
    
log_likelihoods = np.array(log_likelihoods).reshape(-1, 1)
# print(log_likelihoods)

test_states = hmm_model.predict(log_likelihoods)
# test_states = hmm_model.predict(test_data)

print("Test States:",test_states)
    

# Compute the accuracy of the model
# testing_labels = np.load('labels.npy', allow_pickle = True)
# testing_labels = np.argmax(labels)
print("Testing Labels:", testing_labels)

predicted_labels = test_states
print("Predicted Labels:", predicted_labels.shape)



accuracy = np.mean(testing_labels == predicted_labels)
print("Accuracy:", accuracy*100)



