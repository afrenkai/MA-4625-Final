echo "starting the process of getting data"

pip install -r requirements.txt
python get_data.py
python hierarchical_cv.py
python get_sample.py

echo "should have finished getting data, and printed a sample of the train split"