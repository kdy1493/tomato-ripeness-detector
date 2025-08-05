import kagglehub
import dataset_tools as dtools

# Download kaggle dataset
# 1. laboro-tomato
# path = kagglehub.dataset_download("nexuswho/laboro-tomato" , path = "data")

# 2. andrewmvd/tomato-detection
# path = kagglehub.dataset_download("andrewmvd/tomato-detection")

# Download datasetninja dataset 
dtools.download(dataset='AgRobTomato Dataset', dst_dir='~/dataset-ninja/')

# print("Path to dataset files:", path)
