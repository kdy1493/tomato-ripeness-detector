import kagglehub

# Download latest version
# path = kagglehub.dataset_download("nexuswho/laboro-tomato" , path = "data")

path = kagglehub.dataset_download("andrewmvd/tomato-detection")

print("Path to dataset files:", path)