DATAFILE=image_CLEF.zip
wget https://transferlearningdrive.blob.core.windows.net/teamdrive/dataset/$DATAFILE
mkdir -p data
unzip $DATAFILE -d data/
mv $DATAFILE data/

