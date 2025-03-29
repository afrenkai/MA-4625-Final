echo "starting the process of getting data"

python3 get_data.py
python3 hierarchical_cv.py
python3 get_sample.py

echo "should have finished getting data, and printed a sample of the train split"