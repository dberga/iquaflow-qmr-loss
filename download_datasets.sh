#USGS (download not available), visor online
#URL: https://earthexplorer.usgs.gov/

#xview dataset (you need to manually be logged in the website and access the download page)
#URL: https://challenge.xviewdataset.org/download-links
wget https://d307kc0mrhucc3.cloudfront.net/train_images.zip
wget https://d307kc0mrhucc3.cloudfront.net/train_labels.zip
wget https://d307kc0mrhucc3.cloudfront.net/val_images.zip

#INRIA Aerial Image Labeling Dataset
curl -k https://files.inria.fr/aerialimagelabeling/getAerial.sh | bash;
