import os
import gc
import numpy as np
import torch
from pathlib import Path
from os import walk  
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


            
def perform_mc_dropout_prediction(model, dataset_path, X_mean, X_std):
    # Load files in the directory and sort them
    files = []

    for (dirpath, dirnames, filenames) in walk(dataset_path):
        files.extend(filenames)
        break

    files.sort()
    
    chunk_shape = [6, 4320]  # 360 chunks
    num_samples = 100  # Adjust based on requirements
    first_run = True
    ii = 0
    
    for count, file in enumerate(files):
        #if count > 2:
        #    break
        features = np.load(Path(dataset_path / file))
        features = np.divide((features - X_mean), X_std)

        features = torch.tensor(features)
        features = features.to(device)

        # Perform multiple forward passes with dropout
        num_samples = 100  # Adjust this based on your requirements
        predictions_sum = np.zeros((num_samples, chunk_shape[0], chunk_shape[1]))

        with torch.no_grad():
            for i in range(num_samples):
                predictions = model(features)

                # Store predictions for later averaging
                predictions_sum[i] = predictions.cpu().detach().numpy().reshape(chunk_shape)

        # Average predictions over all samples
        predictions_avg = np.mean(predictions_sum, axis=0)
        predictions_var = np.var(predictions_sum, axis=0)
        save_path = os.path.join(prediction_sum_save , f'predictions_sum_{ii}.npy')
        #np.save(save_path, predictions_sum)
        #fit_params, failed_fits, failed_fit_details = fit_gaussian_to_predictions(predictions_sum)
        #print(failed_fits)

        if first_run:
            prediction_map2 = predictions_avg
            prediction_map2_var = predictions_var
            first_run = False
        else:
            prediction_map2 = np.append(prediction_map2, predictions_avg, axis=0)
            prediction_map2_var = np.append(prediction_map2_var, predictions_var, axis=0)

        ii += 1
        print("Prediction " + str(ii) + " done!")

        del features
        gc.collect()



    
    return prediction_map2, prediction_map2_var